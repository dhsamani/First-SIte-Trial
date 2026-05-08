# Databricks notebook source
# MAGIC %md
# MAGIC # Asset Intelligence & Recommendation Pipeline
# MAGIC ### Reads `asset_health_score_features` → Aggregates per company → Calls LLM → Materializes drill-down assets
# MAGIC **Run Mode:** Daily batch via Databricks Workflow | Cluster: DBR 13.3 LTS+

# COMMAND ----------
# MAGIC %md ## 0. Bootstrap — Install dependencies & configure environment

# COMMAND ----------
# %pip install anthropic tenacity  # Uncomment on first run or pin in cluster init script
# dbutils.library.restartPython()  # Uncomment after pip install

# COMMAND ----------

import os
import json
import uuid
import logging
import traceback
from datetime import date, datetime
from functools import reduce
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, TimestampType, DateType, LongType
)
from delta.tables import DeltaTable

import mlflow

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("AssetIntelligencePipeline")

# COMMAND ----------
# MAGIC %md ## 1. Configuration

# COMMAND ----------

class PipelineConfig:
    """
    Central configuration.  Override via Databricks Widgets or environment variables.
    All catalog/schema references follow Unity Catalog: catalog.schema.table
    """

    # ── Databricks Widgets (set at runtime) ───────────────────────────────
    @staticmethod
    def _widget(name: str, default: str) -> str:
        try:
            val = dbutils.widgets.get(name)
            return val if val else default
        except Exception:
            return os.getenv(name.upper().replace("-", "_"), default)

    # ── Source ────────────────────────────────────────────────────────────
    SOURCE_TABLE: str           = "catalog.schema.asset_health_score_features"

    # ── Output catalog / schema ───────────────────────────────────────────
    TARGET_CATALOG: str         = "catalog"
    TARGET_SCHEMA: str          = "asset_intelligence"

    # Derived output table FQNs
    @classmethod
    def tbl(cls, name: str) -> str:
        return f"{cls.TARGET_CATALOG}.{cls.TARGET_SCHEMA}.{name}"

    FEAT_TABLE: str             = "fact_asset_risk_features"      # per-asset, per-day
    AGG_TABLE: str              = "agg_company_metrics"           # aggregated, per-company, per-day
    REC_TABLE: str              = "recommendations"               # LLM output
    REC_ASSETS_TABLE: str       = "recommendation_assets"         # drill-down
    AUDIT_TABLE: str            = "llm_audit_log"                 # every LLM call

    # ── LLM ───────────────────────────────────────────────────────────────
    ANTHROPIC_SECRET_SCOPE: str = "anthropic"
    ANTHROPIC_SECRET_KEY: str   = "api_key"
    LLM_MODEL: str              = "claude-opus-4-5"
    LLM_MAX_TOKENS: int         = 4096
    LLM_TEMPERATURE: float      = 0.2      # low for determinism
    RECS_PER_PERSONA: int       = 6

    # ── Risk thresholds (days) ────────────────────────────────────────────
    CRITICAL_DAYS: int          = 30
    HIGH_DAYS: int              = 90
    MEDIUM_DAYS: int            = 180
    LOW_DAYS: int               = 365

    # ── Energy / CO2 percentile threshold for "high risk" flag ───────────
    ENERGY_RISK_PERCENTILE: float = 0.75   # top 25% within company = high risk
    CO2_RISK_PERCENTILE: float    = 0.75

    # ── Parallelism ───────────────────────────────────────────────────────
    LLM_WORKER_THREADS: int     = 4        # parallel LLM calls across companies
    SPARK_SHUFFLE_PARTITIONS: int = 200

    # ── Source column name map (raw → canonical) ──────────────────────────
    COLUMN_MAP: dict = {
        "company_id":                   "company_id",
        "serial_number":                "serial_number",
        "end_of_support_days_remaining": "eos_support_days_remaining",
        "endofsaledaysremainig":        "eos_days_remaining",          # source typo handled
        "endoflifedaysremaining":       "eol_days_remaining",
        "contractdaysremaining":        "contract_days_remaining",
        "estimated energy use":         "estimated_energy_use_kwh",
        "carbon usage":                 "carbon_usage_kg",
        "ceo score":                    "ceo_score",
        "cfo score":                    "cfo_score",
        "cio score":                    "cio_score",                   # normalises CIO/cio
        "security score":               "security_score",
        "sustainability score":         "sustainability_score",
    }

    # ── Persona definitions ───────────────────────────────────────────────
    PERSONAS: list = [
        {
            "code":         "CEO",
            "name":         "Chief Executive Officer",
            "tone":         "STRATEGIC",
            "score_col":    "ceo_score",
            "primary_focus": (
                "Business continuity, strategic investment decisions, board-level risk visibility, "
                "capital expenditure prioritisation, vendor concentration risk."
            ),
        },
        {
            "code":         "CFO",
            "name":         "Chief Financial Officer",
            "tone":         "FINANCIAL",
            "score_col":    "cfo_score",
            "primary_focus": (
                "Cost avoidance, ROI analysis, contract spend optimisation, emergency vs. planned "
                "replacement cost delta, energy cost reduction, carbon financial liability."
            ),
        },
        {
            "code":         "CIO",
            "name":         "Chief Information Officer",
            "tone":         "TECHNICAL_STRATEGIC",
            "score_col":    "cio_score",
            "primary_focus": (
                "Technology debt, security patch coverage, architecture modernisation, "
                "procurement lead times, platform compatibility, EOS-driven vulnerability exposure."
            ),
        },
        {
            "code":         "SECURITY",
            "name":         "Security Officer",
            "tone":         "RISK_OPERATIONAL",
            "score_col":    "security_score",
            "primary_focus": (
                "Unpatched CVE exposure on EOS devices, zero-day attack surface, devices running "
                "without active support contracts, shadow infrastructure, compliance gaps."
            ),
        },
        {
            "code":         "SUSTAINABILITY",
            "name":         "Sustainability Officer",
            "tone":         "ENVIRONMENTAL",
            "score_col":    "sustainability_score",
            "primary_focus": (
                "Fleet CO2 emissions, energy inefficiency from aging devices, ESG reporting "
                "baselines, carbon reduction potential, Scope 2 GHG protocol alignment."
            ),
        },
    ]


# Register Databricks widgets for runtime overrides
try:
    dbutils.widgets.text("run_date",     "",       "Run Date (YYYY-MM-DD, blank=today)")
    dbutils.widgets.text("company_filter", "",     "Single company_id to process (blank=all)")
    dbutils.widgets.text("dry_run",      "false",  "Dry run — skip LLM calls and writes")
except Exception:
    pass  # running outside Databricks notebook context


def get_run_date() -> date:
    raw = PipelineConfig._widget("run_date", "")
    return datetime.strptime(raw, "%Y-%m-%d").date() if raw else date.today()


def is_dry_run() -> bool:
    return PipelineConfig._widget("dry_run", "false").lower() == "true"


def get_company_filter() -> Optional[str]:
    val = PipelineConfig._widget("company_filter", "")
    return val if val else None


# COMMAND ----------
# MAGIC %md ## 2. Spark Session & Schema Initialisation

# COMMAND ----------

class SparkManager:
    """Configures and returns the active SparkSession with Databricks-optimised settings."""

    @staticmethod
    def get_session() -> SparkSession:
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.shuffle.partitions",
                       str(PipelineConfig.SPARK_SHUFFLE_PARTITIONS))
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
        spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
        return spark

    @staticmethod
    def ensure_schema(spark: SparkSession) -> None:
        spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS "
            f"{PipelineConfig.TARGET_CATALOG}.{PipelineConfig.TARGET_SCHEMA}"
        )
        logger.info(
            f"Schema ready: {PipelineConfig.TARGET_CATALOG}.{PipelineConfig.TARGET_SCHEMA}"
        )


# COMMAND ----------
# MAGIC %md ## 3. Data Loading & Column Normalisation

# COMMAND ----------

class DataLoader:
    """
    Reads the source table and normalises every column name to the canonical
    form defined in PipelineConfig.COLUMN_MAP.  Handles:
      - Spaces in column names     ("estimated energy use")
      - Capitalisation variants    ("CIO score" / "cio score")
      - Known source typos         ("endofsaledaysremainig")
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load(self, company_filter: Optional[str] = None) -> DataFrame:
        logger.info(f"Reading source table: {PipelineConfig.SOURCE_TABLE}")
        df = self.spark.table(PipelineConfig.SOURCE_TABLE)

        df = self._normalise_columns(df)
        df = self._cast_types(df)
        df = self._filter_active(df, company_filter)

        logger.info(f"Loaded {df.count()} rows after normalisation & filtering.")
        return df

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: DataFrame) -> DataFrame:
        """
        Lowercase + strip all column names, then apply the canonical mapping.
        Any column not in the map is kept with its whitespace-cleaned name.
        """
        # Step 1: lowercase + strip whitespace from all column names
        renamed = df
        for col in df.columns:
            clean = col.lower().strip()
            if clean != col:
                renamed = renamed.withColumnRenamed(col, clean)

        # Step 2: apply canonical map
        cmap = {k.lower().strip(): v for k, v in PipelineConfig.COLUMN_MAP.items()}
        for raw_name, canonical in cmap.items():
            if raw_name in renamed.columns and raw_name != canonical:
                renamed = renamed.withColumnRenamed(raw_name, canonical)

        return renamed

    @staticmethod
    def _cast_types(df: DataFrame) -> DataFrame:
        """
        Enforce correct data types.  Source may store all columns as strings
        depending on ETL origin.
        """
        int_cols = [
            "eos_support_days_remaining", "eos_days_remaining",
            "eol_days_remaining", "contract_days_remaining",
        ]
        double_cols = [
            "estimated_energy_use_kwh", "carbon_usage_kg",
            "ceo_score", "cfo_score", "cio_score",
            "security_score", "sustainability_score",
        ]
        for c in int_cols:
            if c in df.columns:
                df = df.withColumn(c, F.col(c).cast(IntegerType()))
        for c in double_cols:
            if c in df.columns:
                df = df.withColumn(c, F.col(c).cast(DoubleType()))

        return df

    @staticmethod
    def _filter_active(df: DataFrame, company_filter: Optional[str]) -> DataFrame:
        """Apply company filter if specified; drop rows with null primary key."""
        df = df.filter(
            F.col("company_id").isNotNull() & F.col("serial_number").isNotNull()
        )
        if company_filter:
            logger.info(f"Filtering to single company: {company_filter}")
            df = df.filter(F.col("company_id") == company_filter)
        return df


# COMMAND ----------
# MAGIC %md ## 4. Feature Engineering — Risk Bands, Percentile Ranks

# COMMAND ----------

class FeatureEngineer:
    """
    Computes derived risk features on top of the raw source columns:
      - Risk bands   (EXPIRED / CRITICAL / HIGH / MEDIUM / LOW / NA) for each days-remaining metric
      - Percentile ranks for energy and CO2 within each company
      - High-risk flags (above company 75th percentile)
      - Composite missing-data flag
    Returns a DataFrame ready to write to fact_asset_risk_features.
    """

    BAND_THRESHOLDS = {
        "EXPIRED":  (None, 0),     # days_remaining < 0
        "CRITICAL": (0,   30),
        "HIGH":     (30,  90),
        "MEDIUM":   (90,  180),
        "LOW":      (180, 365),
        "FUTURE":   (365, None),   # days > 365 → very low urgency
    }

    def __init__(self, run_date: date):
        self.run_date = run_date

    def transform(self, df: DataFrame) -> DataFrame:
        df = self._add_risk_bands(df)
        df = self._add_percentile_ranks(df)
        df = self._add_high_risk_flags(df)
        df = self._add_missing_data_flag(df)
        df = df.withColumn("computed_date", F.lit(self.run_date).cast(DateType()))
        df = df.withColumn("pipeline_run_id", F.lit(str(uuid.uuid4())))
        return df

    # ── Risk band helper ──────────────────────────────────────────────────

    @staticmethod
    def _band_expr(col_name: str) -> F.Column:
        """
        Vectorised risk-band classification for a days-remaining column.
        Returns a string column: EXPIRED | CRITICAL | HIGH | MEDIUM | LOW | FUTURE | NA
        """
        return (
            F.when(F.col(col_name).isNull(), F.lit("NA"))
             .when(F.col(col_name) < 0,   F.lit("EXPIRED"))
             .when(F.col(col_name) <= 30,  F.lit("CRITICAL"))
             .when(F.col(col_name) <= 90,  F.lit("HIGH"))
             .when(F.col(col_name) <= 180, F.lit("MEDIUM"))
             .when(F.col(col_name) <= 365, F.lit("LOW"))
             .otherwise(F.lit("FUTURE"))
        )

    def _add_risk_bands(self, df: DataFrame) -> DataFrame:
        days_cols = {
            "eos_support_days_remaining": "eos_support_risk_band",
            "eos_days_remaining":         "eos_risk_band",
            "eol_days_remaining":         "eol_risk_band",
            "contract_days_remaining":    "contract_risk_band",
        }
        for src, tgt in days_cols.items():
            if src in df.columns:
                df = df.withColumn(tgt, self._band_expr(src))
        return df

    def _add_percentile_ranks(self, df: DataFrame) -> DataFrame:
        """
        Compute within-company percentile rank (0–100) for energy and CO2.
        Uses percent_rank() window function partitioned by company_id.
        """
        w = Window.partitionBy("company_id").orderBy("estimated_energy_use_kwh")
        df = df.withColumn(
            "energy_pct_rank",
            F.round(F.percent_rank().over(w) * 100, 2)
        )

        w2 = Window.partitionBy("company_id").orderBy("carbon_usage_kg")
        df = df.withColumn(
            "co2_pct_rank",
            F.round(F.percent_rank().over(w2) * 100, 2)
        )
        return df

    def _add_high_risk_flags(self, df: DataFrame) -> DataFrame:
        """
        Flag devices above the configured percentile threshold within their company.
        """
        threshold = PipelineConfig.ENERGY_RISK_PERCENTILE * 100
        df = df.withColumn(
            "energy_high_risk_flag",
            F.col("energy_pct_rank") >= F.lit(threshold)
        )
        df = df.withColumn(
            "co2_high_risk_flag",
            F.col("co2_pct_rank") >= F.lit(threshold)
        )
        return df

    @staticmethod
    def _add_missing_data_flag(df: DataFrame) -> DataFrame:
        """
        True if ANY of the key date-derived columns is NULL — asset is invisible to risk scoring.
        """
        date_cols = [
            "eos_support_days_remaining", "eos_days_remaining",
            "eol_days_remaining", "contract_days_remaining",
        ]
        missing_cond = reduce(
            lambda a, b: a | b,
            [F.col(c).isNull() for c in date_cols if c in df.columns]
        )
        return df.withColumn("missing_date_flag", missing_cond)


# COMMAND ----------
# MAGIC %md ## 5. Company-Level Aggregation

# COMMAND ----------

class CompanyAggregator:
    """
    Aggregates the feature-enriched per-asset data into one summary row per company.
    This summary is the direct input to the LLM prompt.
    Output schema matches agg_company_metrics Delta table.
    """

    def aggregate(self, feat_df: DataFrame) -> DataFrame:
        logger.info("Computing company-level aggregations...")

        agg = feat_df.groupBy("company_id", "computed_date").agg(
            # ── Fleet counts ─────────────────────────────────────────────
            F.count("*")                                    .alias("total_assets"),
            F.sum(F.col("missing_date_flag").cast("int"))  .alias("assets_missing_dates"),

            # ── EOL bands ────────────────────────────────────────────────
            F.sum((F.col("eol_risk_band") == "EXPIRED") .cast("int")).alias("eol_expired_count"),
            F.sum((F.col("eol_risk_band") == "CRITICAL").cast("int")).alias("eol_critical_count"),
            F.sum((F.col("eol_risk_band") == "HIGH")    .cast("int")).alias("eol_high_count"),
            F.sum((F.col("eol_risk_band") == "MEDIUM")  .cast("int")).alias("eol_medium_count"),
            F.sum((F.col("eol_risk_band") == "LOW")     .cast("int")).alias("eol_low_count"),
            F.sum((F.col("eol_risk_band") == "NA")      .cast("int")).alias("eol_na_count"),

            # ── EOS (end-of-sale) bands ───────────────────────────────────
            F.sum((F.col("eos_risk_band") == "EXPIRED") .cast("int")).alias("eos_expired_count"),
            F.sum((F.col("eos_risk_band") == "CRITICAL").cast("int")).alias("eos_critical_count"),
            F.sum((F.col("eos_risk_band") == "HIGH")    .cast("int")).alias("eos_high_count"),
            F.sum((F.col("eos_risk_band") == "MEDIUM")  .cast("int")).alias("eos_medium_count"),

            # ── EOS Support bands ─────────────────────────────────────────
            F.sum((F.col("eos_support_risk_band") == "EXPIRED") .cast("int")).alias("eos_support_expired_count"),
            F.sum((F.col("eos_support_risk_band") == "CRITICAL").cast("int")).alias("eos_support_critical_count"),
            F.sum((F.col("eos_support_risk_band") == "HIGH")    .cast("int")).alias("eos_support_high_count"),

            # ── Contract bands ────────────────────────────────────────────
            F.sum((F.col("contract_risk_band") == "EXPIRED") .cast("int")).alias("contract_expired_count"),
            F.sum((F.col("contract_risk_band") == "CRITICAL").cast("int")).alias("contract_critical_count"),
            F.sum((F.col("contract_risk_band") == "HIGH")    .cast("int")).alias("contract_high_count"),
            F.sum((F.col("contract_risk_band") == "MEDIUM")  .cast("int")).alias("contract_medium_count"),

            # ── Energy metrics ────────────────────────────────────────────
            F.sum("estimated_energy_use_kwh")                    .alias("total_energy_kwh"),
            F.avg("estimated_energy_use_kwh")                    .alias("avg_energy_kwh_per_device"),
            F.max("estimated_energy_use_kwh")                    .alias("max_energy_kwh_device"),
            F.sum(F.col("energy_high_risk_flag").cast("int"))    .alias("energy_high_risk_count"),

            # ── CO2 metrics ───────────────────────────────────────────────
            F.sum("carbon_usage_kg")                             .alias("total_co2_kg"),
            F.avg("carbon_usage_kg")                             .alias("avg_co2_kg_per_device"),
            F.max("carbon_usage_kg")                             .alias("max_co2_kg_device"),
            F.sum(F.col("co2_high_risk_flag").cast("int"))       .alias("co2_high_risk_count"),

            # ── Persona scores (avg across fleet) ─────────────────────────
            F.avg("ceo_score")            .alias("avg_ceo_score"),
            F.avg("cfo_score")            .alias("avg_cfo_score"),
            F.avg("cio_score")            .alias("avg_cio_score"),
            F.avg("security_score")       .alias("avg_security_score"),
            F.avg("sustainability_score") .alias("avg_sustainability_score"),

            # ── Worst-case persona scores ─────────────────────────────────
            F.min("ceo_score")            .alias("min_ceo_score"),
            F.min("cfo_score")            .alias("min_cfo_score"),
            F.min("cio_score")            .alias("min_cio_score"),
            F.min("security_score")       .alias("min_security_score"),
            F.min("sustainability_score") .alias("min_sustainability_score"),
        )

        agg = self._add_percentage_metrics(agg)
        agg = agg.withColumn("agg_id", F.expr("uuid()"))
        return agg

    @staticmethod
    def _add_percentage_metrics(agg: DataFrame) -> DataFrame:
        total = F.col("total_assets")

        def pct(col_name: str) -> F.Column:
            return F.round(F.col(col_name) / total * 100, 2)

        return (
            agg
            .withColumn("eol_pct_expired",  pct("eol_expired_count"))
            .withColumn("eol_pct_critical", pct("eol_critical_count"))
            .withColumn("eol_pct_high",     pct("eol_high_count"))
            .withColumn("eos_pct_expired",  pct("eos_expired_count"))
            .withColumn("eos_pct_critical", pct("eos_critical_count"))
            .withColumn("eos_pct_high",     pct("eos_high_count"))
            .withColumn("eos_support_pct_expired",  pct("eos_support_expired_count"))
            .withColumn("eos_support_pct_critical", pct("eos_support_critical_count"))
            .withColumn("contract_pct_expired",  pct("contract_expired_count"))
            .withColumn("contract_pct_critical", pct("contract_critical_count"))
        )


# COMMAND ----------
# MAGIC %md ## 6. LLM Prompt Builder

# COMMAND ----------

class PromptBuilder:
    """
    Builds a structured system + user prompt for each company × persona combination.
    The user prompt is populated entirely from the aggregation row — no raw asset data
    crosses the LLM boundary (keeps tokens low and data governance clean).
    """

    SYSTEM_TEMPLATE = """You are an enterprise asset intelligence advisor generating recommendations for the {persona_name}.

PERSONA CONTEXT:
- Role: {persona_name}
- Primary concerns: {primary_focus}
- Communication tone: {tone} — adjust language accordingly.
  (STRATEGIC = board-level language, FINANCIAL = cost/ROI focus, 
   TECHNICAL_STRATEGIC = architecture + risk, RISK_OPERATIONAL = security/compliance,
   ENVIRONMENTAL = ESG/sustainability)

INSTRUCTIONS:
1. Generate exactly {n_recs} recommendations ranked by urgency/business impact.
2. Each recommendation must be directly actionable within this persona's scope.
3. Reference SPECIFIC metric values from the company data provided.
4. For EVERY recommendation include a filter_definition JSON that describes which assets 
   are affected (this drives the drill-down in the UI).
5. Return ONLY a valid JSON array — no prose, no markdown, no code fences.

filter_definition SCHEMA (use exact field names below):
{{
  "filters": [
    {{"field": "<field_name>", "operator": "<|>|<=|>=|=|IS NULL|IS NOT NULL", "value": <value>}}
  ]
}}
Valid field names for filters:
  eol_days_remaining, eos_days_remaining, eos_support_days_remaining,
  contract_days_remaining, estimated_energy_use_kwh, carbon_usage_kg,
  energy_high_risk_flag, co2_high_risk_flag, missing_date_flag,
  eol_risk_band, eos_risk_band, eos_support_risk_band, contract_risk_band

REQUIRED JSON FIELDS PER RECOMMENDATION:
  title               (str, max 80 chars)
  recommendation_text (str, 2-4 sentences, specific numbers, actionable)
  action_label        (str, CTA text, e.g. "View 47 critical devices")
  priority            (str, one of: CRITICAL | HIGH | MEDIUM | LOW)
  category            (str, one of: EOL | EOS | EOS_SUPPORT | CONTRACT | ENERGY | CO2 | COMPOSITE)
  metric_value        (number — the headline metric, e.g. 47)
  metric_unit         (str — e.g. "devices", "USD", "kWh", "kg CO2", "%")
  metric_label        (str — context for the number, e.g. "past end-of-life")
  filter_definition   (object — as specified above)
"""

    USER_TEMPLATE = """
COMPANY ASSET INTELLIGENCE REPORT
Company ID  : {company_id}
Report Date : {computed_date}
Persona     : {persona_name} ({persona_code})

════════════════════════════════════════
FLEET OVERVIEW
════════════════════════════════════════
Total Active Assets         : {total_assets}
Assets with Missing Dates   : {assets_missing_dates}  ← cannot be risk-scored

════════════════════════════════════════
END-OF-LIFE (EOL)
════════════════════════════════════════
Expired (past EOL)          : {eol_expired_count}  ({eol_pct_expired}% of fleet)
Critical  (1–30 days)       : {eol_critical_count}  ({eol_pct_critical}%)
High      (31–90 days)      : {eol_high_count}     ({eol_pct_high}%)
Medium    (91–180 days)     : {eol_medium_count}
Low       (181–365 days)    : {eol_low_count}

════════════════════════════════════════
END-OF-SALE (EOS)
════════════════════════════════════════
Expired (past EOS)          : {eos_expired_count}  ({eos_pct_expired}%)
Critical  (1–30 days)       : {eos_critical_count}  ({eos_pct_critical}%)
High      (31–90 days)      : {eos_high_count}     ({eos_pct_high}%)

════════════════════════════════════════
END-OF-SUPPORT (EOS Support / Security Patches)
════════════════════════════════════════
Expired (no patches)        : {eos_support_expired_count}  ({eos_support_pct_expired}%)
Critical  (1–30 days)       : {eos_support_critical_count}  ({eos_support_pct_critical}%)
High      (31–90 days)      : {eos_support_high_count}

════════════════════════════════════════
CONTRACT STATUS
════════════════════════════════════════
Expired contracts           : {contract_expired_count}  ({contract_pct_expired}%)
Critical  (≤30 days)        : {contract_critical_count}  ({contract_pct_critical}%)
High      (31–90 days)      : {contract_high_count}
Medium    (91–180 days)     : {contract_medium_count}

════════════════════════════════════════
ENERGY & CO2
════════════════════════════════════════
Total Annual Energy         : {total_energy_kwh:.0f} kWh
Avg Energy per Device       : {avg_energy_kwh_per_device:.1f} kWh
Max Energy (single device)  : {max_energy_kwh_device:.1f} kWh
High-Energy Risk Devices    : {energy_high_risk_count}  (above 75th pct within fleet)
Total Annual CO2            : {total_co2_kg:.0f} kg
Avg CO2 per Device          : {avg_co2_kg_per_device:.1f} kg
High-CO2 Risk Devices       : {co2_high_risk_count}

════════════════════════════════════════
PERSONA HEALTH SCORES (0=best, 100=worst risk)
════════════════════════════════════════
Avg {persona_code} Score across fleet : {avg_persona_score:.1f}
Min {persona_code} Score (worst asset): {min_persona_score:.1f}

Generate exactly {n_recs} recommendations for the {persona_name}. Return ONLY the JSON array.
"""

    def build(self, agg_row: dict, persona: dict) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) for a single company × persona call."""

        system = self.SYSTEM_TEMPLATE.format(
            persona_name=persona["name"],
            primary_focus=persona["primary_focus"],
            tone=persona["tone"],
            n_recs=PipelineConfig.RECS_PER_PERSONA,
        )

        # Safely pull persona-specific scores (column names built dynamically)
        score_col  = persona["score_col"].replace(" ", "_")
        avg_col    = f"avg_{score_col}"
        min_col    = f"min_{score_col}"

        user = self.USER_TEMPLATE.format(
            company_id=agg_row.get("company_id", "UNKNOWN"),
            computed_date=agg_row.get("computed_date", str(date.today())),
            persona_name=persona["name"],
            persona_code=persona["code"],
            total_assets=agg_row.get("total_assets", 0),
            assets_missing_dates=agg_row.get("assets_missing_dates", 0),

            eol_expired_count=agg_row.get("eol_expired_count", 0),
            eol_pct_expired=agg_row.get("eol_pct_expired", 0.0),
            eol_critical_count=agg_row.get("eol_critical_count", 0),
            eol_pct_critical=agg_row.get("eol_pct_critical", 0.0),
            eol_high_count=agg_row.get("eol_high_count", 0),
            eol_pct_high=agg_row.get("eol_pct_high", 0.0),
            eol_medium_count=agg_row.get("eol_medium_count", 0),
            eol_low_count=agg_row.get("eol_low_count", 0),

            eos_expired_count=agg_row.get("eos_expired_count", 0),
            eos_pct_expired=agg_row.get("eos_pct_expired", 0.0),
            eos_critical_count=agg_row.get("eos_critical_count", 0),
            eos_pct_critical=agg_row.get("eos_pct_critical", 0.0),
            eos_high_count=agg_row.get("eos_high_count", 0),
            eos_pct_high=agg_row.get("eos_pct_high", 0.0),

            eos_support_expired_count=agg_row.get("eos_support_expired_count", 0),
            eos_support_pct_expired=agg_row.get("eos_support_pct_expired", 0.0),
            eos_support_critical_count=agg_row.get("eos_support_critical_count", 0),
            eos_support_pct_critical=agg_row.get("eos_support_pct_critical", 0.0),
            eos_support_high_count=agg_row.get("eos_support_high_count", 0),

            contract_expired_count=agg_row.get("contract_expired_count", 0),
            contract_pct_expired=agg_row.get("contract_pct_expired", 0.0),
            contract_critical_count=agg_row.get("contract_critical_count", 0),
            contract_pct_critical=agg_row.get("contract_pct_critical", 0.0),
            contract_high_count=agg_row.get("contract_high_count", 0),
            contract_medium_count=agg_row.get("contract_medium_count", 0),

            total_energy_kwh=agg_row.get("total_energy_kwh") or 0,
            avg_energy_kwh_per_device=agg_row.get("avg_energy_kwh_per_device") or 0,
            max_energy_kwh_device=agg_row.get("max_energy_kwh_device") or 0,
            energy_high_risk_count=agg_row.get("energy_high_risk_count", 0),

            total_co2_kg=agg_row.get("total_co2_kg") or 0,
            avg_co2_kg_per_device=agg_row.get("avg_co2_kg_per_device") or 0,
            co2_high_risk_count=agg_row.get("co2_high_risk_count", 0),

            avg_persona_score=agg_row.get(avg_col) or 0.0,
            min_persona_score=agg_row.get(min_col) or 0.0,
            n_recs=PipelineConfig.RECS_PER_PERSONA,
        )

        return system, user


# COMMAND ----------
# MAGIC %md ## 7. LLM Client with Retry Logic

# COMMAND ----------

class LLMClient:
    """
    Calls the Anthropic Messages API with exponential backoff retry.
    Fetches the API key from Databricks Secret Scope.
    All calls are logged to llm_audit_log for governance.
    """

    def __init__(self):
        self._api_key = self._get_api_key()
        self._base_url = "https://api.anthropic.com/v1/messages"

    @staticmethod
    def _get_api_key() -> str:
        try:
            return dbutils.secrets.get(
                scope=PipelineConfig.ANTHROPIC_SECRET_SCOPE,
                key=PipelineConfig.ANTHROPIC_SECRET_KEY,
            )
        except Exception:
            # Fallback to environment variable for local testing
            key = os.getenv("ANTHROPIC_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "Anthropic API key not found. Set it in Databricks Secret Scope "
                    f"'{PipelineConfig.ANTHROPIC_SECRET_SCOPE}' / "
                    f"'{PipelineConfig.ANTHROPIC_SECRET_KEY}' or ANTHROPIC_API_KEY env var."
                )
            return key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
        reraise=True,
    )
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        company_id: str,
        persona_code: str,
    ) -> tuple[list, dict]:
        """
        Returns (parsed_recommendations_list, audit_metadata_dict).
        Raises on unrecoverable errors.
        """
        start = datetime.utcnow()

        payload = {
            "model": PipelineConfig.LLM_MODEL,
            "max_tokens": PipelineConfig.LLM_MAX_TOKENS,
            "temperature": PipelineConfig.LLM_TEMPERATURE,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = requests.post(
            self._base_url, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()

        resp_json = response.json()
        raw_text  = resp_json["content"][0]["text"].strip()
        latency_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

        # Parse JSON — strip any accidental code fences from LLM
        clean_text = raw_text
        for fence in ["```json", "```JSON", "```"]:
            clean_text = clean_text.replace(fence, "")
        clean_text = clean_text.strip()

        try:
            recommendations = json.loads(clean_text)
            if not isinstance(recommendations, list):
                raise ValueError("LLM returned non-list JSON root")
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}\nRaw: {raw_text[:500]}")

        usage = resp_json.get("usage", {})
        audit = {
            "log_id":             str(uuid.uuid4()),
            "company_id":         company_id,
            "persona_code":       persona_code,
            "system_prompt":      system_prompt,
            "user_prompt":        user_prompt,
            "raw_llm_response":   raw_text,
            "parsed_successfully": True,
            "error_message":      None,
            "tokens_input":       usage.get("input_tokens", 0),
            "tokens_output":      usage.get("output_tokens", 0),
            "latency_ms":         latency_ms,
            "model_version":      PipelineConfig.LLM_MODEL,
            "created_at":         datetime.utcnow().isoformat(),
        }

        logger.info(
            f"LLM call OK | {company_id} × {persona_code} | "
            f"{len(recommendations)} recs | {latency_ms}ms | "
            f"in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)}"
        )
        return recommendations, audit


# COMMAND ----------
# MAGIC %md ## 8. Drill-Down Materialiser — Filter Evaluation Engine

# COMMAND ----------

class DrilldownMaterialiser:
    """
    Given a filter_definition from the LLM, evaluates it against the
    feature DataFrame and returns the matching (serial_number, asset_name, ...) rows.

    This is the mechanism that makes clicking "40 devices" return the exact asset list.

    Design principle:
      - Filter definition is the LLM's intent.
      - We re-evaluate against fact_asset_risk_features for the run_date.
      - The resulting asset list is stored in recommendation_assets (point-in-time snapshot).
      - Future API calls simply SELECT from recommendation_assets — no re-filtering needed.
    """

    # All columns that may appear in filter_definitions (from the LLM system prompt)
    ALLOWED_FILTER_FIELDS = {
        "eol_days_remaining", "eos_days_remaining", "eos_support_days_remaining",
        "contract_days_remaining", "estimated_energy_use_kwh", "carbon_usage_kg",
        "energy_high_risk_flag", "co2_high_risk_flag", "missing_date_flag",
        "eol_risk_band", "eos_risk_band", "eos_support_risk_band", "contract_risk_band",
    }

    OPERATOR_MAP = {
        "<":          lambda col, val: F.col(col) < val,
        ">":          lambda col, val: F.col(col) > val,
        "<=":         lambda col, val: F.col(col) <= val,
        ">=":         lambda col, val: F.col(col) >= val,
        "=":          lambda col, val: F.col(col) == val,
        "!=":         lambda col, val: F.col(col) != val,
        "IS NULL":    lambda col, _:   F.col(col).isNull(),
        "IS NOT NULL":lambda col, _:   F.col(col).isNotNull(),
    }

    def evaluate(
        self,
        feat_df: DataFrame,
        company_id: str,
        filter_definition: dict,
        sort_metric_col: Optional[str] = None,
    ) -> DataFrame:
        """
        Returns a DataFrame of matching assets for a given company + filter definition.
        Columns returned: serial_number, company_id, and all feature columns.
        """
        result = feat_df.filter(F.col("company_id") == company_id)

        filters = filter_definition.get("filters", [])
        if not filters:
            logger.warning(f"Empty filter_definition for company {company_id} — returning no assets.")
            return result.limit(0)

        conditions = []
        for f in filters:
            field    = f.get("field", "")
            operator = f.get("operator", "")
            value    = f.get("value")

            # Security: only allow known fields
            if field not in self.ALLOWED_FILTER_FIELDS:
                logger.warning(f"Skipping unknown filter field: {field}")
                continue

            if field not in result.columns:
                logger.warning(f"Filter field '{field}' not in DataFrame — skipping.")
                continue

            op_fn = self.OPERATOR_MAP.get(operator)
            if op_fn is None:
                logger.warning(f"Unknown operator '{operator}' — skipping filter.")
                continue

            conditions.append(op_fn(field, value))

        if conditions:
            combined = reduce(lambda a, b: a & b, conditions)
            result = result.filter(combined)

        # Sort by sort_metric_col ascending (most urgent first)
        if sort_metric_col and sort_metric_col in result.columns:
            result = result.orderBy(F.col(sort_metric_col).asc_nulls_last())

        return result


# COMMAND ----------
# MAGIC %md ## 9. Delta Table Writers

# COMMAND ----------

class DeltaWriter:
    """
    Handles all Delta table writes with MERGE (upsert) semantics.
    - fact_asset_risk_features : merge on (company_id, serial_number, computed_date)
    - agg_company_metrics      : merge on (company_id, computed_date)
    - recommendations          : merge on (recommendation_id)
    - recommendation_assets    : merge on (recommendation_id, serial_number)
    - llm_audit_log            : append-only insert
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # ── Generic helpers ───────────────────────────────────────────────────

    def _table_exists(self, fqn: str) -> bool:
        try:
            self.spark.table(fqn)
            return True
        except Exception:
            return False

    def _merge_or_create(
        self,
        df: DataFrame,
        fqn: str,
        merge_keys: list[str],
        partition_cols: Optional[list[str]] = None,
    ) -> None:
        if not self._table_exists(fqn):
            writer = df.write.format("delta").mode("overwrite")
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            writer.saveAsTable(fqn)
            logger.info(f"Created table {fqn} ({df.count()} rows)")
        else:
            dt = DeltaTable.forName(self.spark, fqn)
            match_cond = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
            update_cols = {c: f"source.{c}" for c in df.columns if c not in merge_keys}
            (
                dt.alias("target")
                  .merge(df.alias("source"), match_cond)
                  .whenMatchedUpdate(set=update_cols)
                  .whenNotMatchedInsertAll()
                  .execute()
            )
            logger.info(f"Merged into {fqn} | keys={merge_keys}")

    def write_features(self, df: DataFrame) -> None:
        fqn = PipelineConfig.tbl(PipelineConfig.FEAT_TABLE)
        self._merge_or_create(
            df, fqn,
            merge_keys=["company_id", "serial_number", "computed_date"],
            partition_cols=["computed_date"],
        )

    def write_aggregations(self, df: DataFrame) -> None:
        fqn = PipelineConfig.tbl(PipelineConfig.AGG_TABLE)
        self._merge_or_create(
            df, fqn,
            merge_keys=["company_id", "computed_date"],
            partition_cols=["computed_date"],
        )

    def write_recommendations(self, rows: list[dict]) -> None:
        if not rows:
            logger.warning("No recommendations to write.")
            return
        df = self.spark.createDataFrame(rows)
        fqn = PipelineConfig.tbl(PipelineConfig.REC_TABLE)
        self._merge_or_create(df, fqn, merge_keys=["recommendation_id"])

    def write_recommendation_assets(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = self.spark.createDataFrame(rows)
        fqn = PipelineConfig.tbl(PipelineConfig.REC_ASSETS_TABLE)
        self._merge_or_create(
            df, fqn,
            merge_keys=["recommendation_id", "serial_number"],
        )

    def append_audit_log(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = self.spark.createDataFrame(rows)
        fqn = PipelineConfig.tbl(PipelineConfig.AUDIT_TABLE)
        df.write.format("delta").mode("append").saveAsTable(fqn)
        logger.info(f"Appended {len(rows)} audit rows to {fqn}")


# COMMAND ----------
# MAGIC %md ## 10. Recommendation Row Builder

# COMMAND ----------

class RecommendationBuilder:
    """
    Validates and flattens raw LLM recommendation dicts into structured rows
    ready for the recommendations Delta table.  Also derives the sort_metric_col
    so the DrilldownMaterialiser knows which column to order by.
    """

    REQUIRED_FIELDS = {
        "title", "recommendation_text", "action_label", "priority",
        "category", "metric_value", "metric_unit", "metric_label", "filter_definition",
    }

    VALID_PRIORITIES  = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    VALID_CATEGORIES  = {"EOL", "EOS", "EOS_SUPPORT", "CONTRACT", "ENERGY", "CO2", "COMPOSITE"}

    # Maps category → the feature column that best sorts the drill-down assets
    SORT_METRIC_MAP = {
        "EOL":         "eol_days_remaining",
        "EOS":         "eos_days_remaining",
        "EOS_SUPPORT": "eos_support_days_remaining",
        "CONTRACT":    "contract_days_remaining",
        "ENERGY":      "estimated_energy_use_kwh",
        "CO2":         "carbon_usage_kg",
        "COMPOSITE":   "eol_days_remaining",
    }

    @classmethod
    def validate_and_build(
        cls,
        raw_rec: dict,
        company_id: str,
        persona_code: str,
        run_date: date,
    ) -> Optional[dict]:
        """
        Returns a clean recommendation dict, or None if validation fails.
        """
        missing = cls.REQUIRED_FIELDS - set(raw_rec.keys())
        if missing:
            logger.warning(f"Recommendation missing fields {missing} — skipping.")
            return None

        priority = str(raw_rec.get("priority", "")).upper()
        category = str(raw_rec.get("category", "")).upper()

        if priority not in cls.VALID_PRIORITIES:
            logger.warning(f"Invalid priority '{priority}' — defaulting to MEDIUM.")
            priority = "MEDIUM"
        if category not in cls.VALID_CATEGORIES:
            logger.warning(f"Invalid category '{category}' — defaulting to COMPOSITE.")
            category = "COMPOSITE"

        rec_id = str(uuid.uuid4())

        return {
            "recommendation_id":  rec_id,
            "company_id":         company_id,
            "persona_code":       persona_code,
            "generated_at":       datetime.utcnow().isoformat(),
            "computed_date":      str(run_date),
            "title":              str(raw_rec.get("title", ""))[:80],
            "recommendation_text": str(raw_rec.get("recommendation_text", "")),
            "action_label":       str(raw_rec.get("action_label", "")),
            "priority":           priority,
            "category":           category,
            "metric_value":       float(raw_rec.get("metric_value") or 0),
            "metric_unit":        str(raw_rec.get("metric_unit", "")),
            "metric_label":       str(raw_rec.get("metric_label", "")),
            "filter_definition":  json.dumps(raw_rec.get("filter_definition", {})),
            "sort_metric_col":    cls.SORT_METRIC_MAP.get(category, "eol_days_remaining"),
            "asset_count_snapshot": 0,   # populated after drill-down materialisation
            "status":             "ACTIVE",
            "llm_model_version":  PipelineConfig.LLM_MODEL,
        }


# COMMAND ----------
# MAGIC %md ## 11. Main Orchestrator

# COMMAND ----------

class AssetIntelligencePipeline:
    """
    Top-level orchestrator.  Wires all components together and runs the pipeline.
    """

    def __init__(self, spark: SparkSession, run_date: date, dry_run: bool = False):
        self.spark      = spark
        self.run_date   = run_date
        self.dry_run    = dry_run

        self.loader        = DataLoader(spark)
        self.feature_eng   = FeatureEngineer(run_date)
        self.aggregator    = CompanyAggregator()
        self.prompt_builder = PromptBuilder()
        self.llm_client    = LLMClient()
        self.drilldown     = DrilldownMaterialiser()
        self.writer        = DeltaWriter(spark)
        self.rec_builder   = RecommendationBuilder()

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, company_filter: Optional[str] = None) -> None:
        logger.info(f"═══ Pipeline START | run_date={self.run_date} | dry_run={self.dry_run} ═══")

        with mlflow.start_run(run_name=f"asset_intelligence_{self.run_date}"):
            mlflow.log_param("run_date",       str(self.run_date))
            mlflow.log_param("dry_run",        str(self.dry_run))
            mlflow.log_param("company_filter", company_filter or "ALL")
            mlflow.log_param("llm_model",      PipelineConfig.LLM_MODEL)

            # ── Step 1: Load & Feature Engineer ──────────────────────────
            raw_df  = self.loader.load(company_filter)
            feat_df = self.feature_eng.transform(raw_df)
            feat_df.cache()
            total_assets = feat_df.count()
            logger.info(f"Feature engineering complete | {total_assets:,} assets")
            mlflow.log_metric("total_assets", total_assets)

            # ── Step 2: Aggregate ─────────────────────────────────────────
            agg_df = self.aggregator.aggregate(feat_df)
            agg_df.cache()
            n_companies = agg_df.count()
            logger.info(f"Aggregation complete | {n_companies} companies")
            mlflow.log_metric("companies_processed", n_companies)

            # ── Step 3: Write feature & aggregation tables ────────────────
            if not self.dry_run:
                self.writer.write_features(feat_df)
                self.writer.write_aggregations(agg_df)

            # ── Step 4: LLM calls per company × persona ───────────────────
            agg_rows = [row.asDict() for row in agg_df.collect()]
            all_recs, all_rec_assets, all_audits = [], [], []

            for agg_row in agg_rows:
                company_id = agg_row["company_id"]
                company_recs, company_assets, company_audits = (
                    self._process_company(agg_row, feat_df)
                )
                all_recs.extend(company_recs)
                all_rec_assets.extend(company_assets)
                all_audits.extend(company_audits)

            mlflow.log_metric("total_recommendations", len(all_recs))
            mlflow.log_metric("total_drilldown_asset_rows", len(all_rec_assets))

            # ── Step 5: Write recommendations + drill-down ────────────────
            if not self.dry_run:
                self.writer.write_recommendations(all_recs)
                self.writer.write_recommendation_assets(all_rec_assets)
                self.writer.append_audit_log(all_audits)
            else:
                logger.info(f"DRY RUN — would write {len(all_recs)} recs, "
                            f"{len(all_rec_assets)} asset rows.")

            feat_df.unpersist()
            agg_df.unpersist()

        logger.info(f"═══ Pipeline COMPLETE | {len(all_recs)} recommendations generated ═══")

    # ── Private: per-company processing ──────────────────────────────────

    def _process_company(
        self,
        agg_row: dict,
        feat_df: DataFrame,
    ) -> tuple[list, list, list]:
        company_id  = agg_row["company_id"]
        recs, assets, audits = [], [], []

        company_feat_df = feat_df.filter(F.col("company_id") == company_id)

        for persona in PipelineConfig.PERSONAS:
            try:
                company_recs, company_assets, audit = self._process_persona(
                    agg_row, company_feat_df, persona
                )
                recs.extend(company_recs)
                assets.extend(company_assets)
                if audit:
                    audits.append(audit)
            except Exception as e:
                logger.error(
                    f"Failed {company_id} × {persona['code']}: {e}\n{traceback.format_exc()}"
                )
                # Store failed audit record
                audits.append({
                    "log_id":             str(uuid.uuid4()),
                    "company_id":         company_id,
                    "persona_code":       persona["code"],
                    "system_prompt":      "",
                    "user_prompt":        "",
                    "raw_llm_response":   "",
                    "parsed_successfully": False,
                    "error_message":      str(e)[:2000],
                    "tokens_input":       0,
                    "tokens_output":      0,
                    "latency_ms":         0,
                    "model_version":      PipelineConfig.LLM_MODEL,
                    "created_at":         datetime.utcnow().isoformat(),
                })

        return recs, assets, audits

    def _process_persona(
        self,
        agg_row: dict,
        company_feat_df: DataFrame,
        persona: dict,
    ) -> tuple[list, list, Optional[dict]]:
        company_id   = agg_row["company_id"]
        persona_code = persona["code"]

        # Build prompts
        system_prompt, user_prompt = self.prompt_builder.build(agg_row, persona)

        # Call LLM (skip in dry_run)
        if self.dry_run:
            logger.info(f"DRY RUN: skipping LLM call for {company_id} × {persona_code}")
            return [], [], None

        raw_recs, audit = self.llm_client.call(
            system_prompt, user_prompt, company_id, persona_code
        )

        validated_recs, rec_asset_rows = [], []

        for raw_rec in raw_recs:
            rec = self.rec_builder.validate_and_build(
                raw_rec, company_id, persona_code, self.run_date
            )
            if rec is None:
                continue

            # Materialise drill-down assets
            filter_def   = json.loads(rec["filter_definition"])
            sort_col     = rec["sort_metric_col"]
            matched_df   = self.drilldown.evaluate(
                company_feat_df, company_id, filter_def, sort_col
            )
            matched_rows = matched_df.select(
                "serial_number",
                F.coalesce(
                    F.col("asset_name") if "asset_name" in matched_df.columns
                    else F.lit("N/A"),
                    F.lit("N/A")
                ).alias("asset_name"),
                sort_col,
            ).collect()

            rec["asset_count_snapshot"] = len(matched_rows)

            for rank, asset_row in enumerate(matched_rows):
                rec_asset_rows.append({
                    "rec_asset_id":      str(uuid.uuid4()),
                    "recommendation_id": rec["recommendation_id"],
                    "company_id":        company_id,
                    "persona_code":      persona_code,
                    "serial_number":     asset_row["serial_number"],
                    "asset_name":        asset_row.get("asset_name", "N/A"),
                    "metric_value":      float(asset_row[sort_col])
                                         if asset_row[sort_col] is not None else None,
                    "metric_col_name":   sort_col,
                    "sort_rank":         rank,
                    "snapshot_date":     str(self.run_date),
                })

            validated_recs.append(rec)

        logger.info(
            f"  {company_id} × {persona_code}: "
            f"{len(validated_recs)} recs, {len(rec_asset_rows)} asset rows"
        )
        return validated_recs, rec_asset_rows, audit


# COMMAND ----------
# MAGIC %md ## 12. Optimisation & Housekeeping

# COMMAND ----------

class TableOptimiser:
    """
    Runs OPTIMIZE + ZORDER on all output tables after writes.
    Also handles expiry of old recommendations (status → EXPIRED).
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def optimise_all(self) -> None:
        optimisations = [
            (PipelineConfig.FEAT_TABLE,      "computed_date, company_id"),
            (PipelineConfig.AGG_TABLE,       "company_id"),
            (PipelineConfig.REC_TABLE,       "company_id, persona_code"),
            (PipelineConfig.REC_ASSETS_TABLE,"recommendation_id"),
        ]
        for table, zorder_cols in optimisations:
            fqn = PipelineConfig.tbl(table)
            try:
                self.spark.sql(f"OPTIMIZE {fqn} ZORDER BY ({zorder_cols})")
                logger.info(f"OPTIMIZE OK: {fqn}")
            except Exception as e:
                logger.warning(f"OPTIMIZE failed for {fqn}: {e}")

    def expire_old_recommendations(self, run_date: date) -> None:
        """
        Mark recommendations from previous run dates as EXPIRED
        so the UI only shows today's batch.
        """
        fqn = PipelineConfig.tbl(PipelineConfig.REC_TABLE)
        try:
            self.spark.sql(f"""
                UPDATE {fqn}
                SET status = 'EXPIRED'
                WHERE status = 'ACTIVE'
                  AND computed_date < '{run_date}'
            """)
            logger.info(f"Expired old recommendations in {fqn}")
        except Exception as e:
            logger.warning(f"Could not expire old recommendations: {e}")

    def vacuum_tables(self, retain_hours: int = 168) -> None:  # 7 days
        for table in [
            PipelineConfig.FEAT_TABLE, PipelineConfig.AGG_TABLE,
            PipelineConfig.REC_TABLE, PipelineConfig.REC_ASSETS_TABLE,
        ]:
            fqn = PipelineConfig.tbl(table)
            try:
                self.spark.sql(f"VACUUM {fqn} RETAIN {retain_hours} HOURS")
            except Exception as e:
                logger.warning(f"VACUUM failed for {fqn}: {e}")


# COMMAND ----------
# MAGIC %md ## 13. Drill-Down Query Helpers (API layer / Notebook consumers)

# COMMAND ----------

class DrilldownQueryHelper:
    """
    Convenience methods for the API layer (FastAPI / Databricks SQL connector)
    to retrieve recommendations and their asset lists.

    These methods are read-only SELECT queries on the materialised tables.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def get_recommendations(
        self,
        company_id: str,
        persona_code: str,
        run_date: Optional[date] = None,
        status: str = "ACTIVE",
    ) -> DataFrame:
        """Returns all recommendations for a company × persona, latest run date."""
        fqn = PipelineConfig.tbl(PipelineConfig.REC_TABLE)
        df  = self.spark.table(fqn).filter(
            (F.col("company_id")   == company_id)   &
            (F.col("persona_code") == persona_code) &
            (F.col("status")       == status)
        )
        if run_date:
            df = df.filter(F.col("computed_date") == F.lit(str(run_date)))
        else:
            # Latest available date
            latest = df.agg(F.max("computed_date")).collect()[0][0]
            df = df.filter(F.col("computed_date") == F.lit(latest))

        return df.orderBy(
            F.when(F.col("priority") == "CRITICAL", 1)
             .when(F.col("priority") == "HIGH",     2)
             .when(F.col("priority") == "MEDIUM",   3)
             .otherwise(4)
        )

    def get_drilldown_assets(
        self,
        recommendation_id: str,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
    ) -> DataFrame:
        """
        Returns paginated asset list for a specific recommendation.
        One indexed SELECT on recommendation_assets — sub-millisecond at scale.
        """
        fqn = PipelineConfig.tbl(PipelineConfig.REC_ASSETS_TABLE)
        df  = self.spark.table(fqn).filter(
            F.col("recommendation_id") == recommendation_id
        )
        if search:
            df = df.filter(
                F.col("serial_number").contains(search) |
                F.col("asset_name").contains(search)
            )
        df = df.orderBy("sort_rank")
        offset = (page - 1) * page_size
        return df.limit(offset + page_size).subtract(df.limit(offset))

    def export_drilldown_csv(
        self, recommendation_id: str, output_path: str
    ) -> str:
        """
        Exports all assets for a recommendation to a CSV in DBFS / Unity Catalog Volume.
        Returns the output path.
        """
        fqn = PipelineConfig.tbl(PipelineConfig.REC_ASSETS_TABLE)
        df  = (
            self.spark.table(fqn)
                .filter(F.col("recommendation_id") == recommendation_id)
                .orderBy("sort_rank")
                .select("serial_number", "asset_name", "metric_value",
                        "metric_col_name", "sort_rank")
        )
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
        logger.info(f"Exported drill-down CSV to {output_path}")
        return output_path


# COMMAND ----------
# MAGIC %md ## 14. Entry Point — Run the Pipeline

# COMMAND ----------

def main():
    run_date       = get_run_date()
    dry_run        = is_dry_run()
    company_filter = get_company_filter()

    spark = SparkManager.get_session()
    SparkManager.ensure_schema(spark)

    pipeline = AssetIntelligencePipeline(spark, run_date, dry_run)
    pipeline.run(company_filter)

    if not dry_run:
        optimiser = TableOptimiser(spark)
        optimiser.expire_old_recommendations(run_date)
        optimiser.optimise_all()
        # optimiser.vacuum_tables()  # Uncomment if running weekly cleanup

    logger.info("Pipeline finished successfully.")


# Run
main()


# COMMAND ----------
# MAGIC %md
# MAGIC ## 15. Ad-hoc Validation Queries
# MAGIC Run these cells manually to inspect outputs.

# COMMAND ----------

# ── Inspect feature table ─────────────────────────────────────────────────────
# display(spark.table(PipelineConfig.tbl(PipelineConfig.FEAT_TABLE))
#              .filter(F.col("eol_risk_band").isin("CRITICAL", "EXPIRED"))
#              .orderBy("eol_days_remaining")
#              .limit(50))

# ── Inspect aggregations ──────────────────────────────────────────────────────
# display(spark.table(PipelineConfig.tbl(PipelineConfig.AGG_TABLE))
#              .filter(F.col("computed_date") == str(date.today()))
#              .orderBy(F.col("eol_critical_count").desc()))

# ── Inspect recommendations for one company × persona ─────────────────────────
# qh = DrilldownQueryHelper(spark)
# recs_df = qh.get_recommendations(company_id="COMP_001", persona_code="CEO")
# display(recs_df.select("title", "priority", "category",
#                        "metric_value", "metric_unit", "asset_count_snapshot"))

# ── Drill-down for a specific recommendation ──────────────────────────────────
# rec_id   = recs_df.collect()[0]["recommendation_id"]
# assets_df = qh.get_drilldown_assets(rec_id, page=1, page_size=20)
# display(assets_df)

# ── LLM audit log ─────────────────────────────────────────────────────────────
# display(spark.table(PipelineConfig.tbl(PipelineConfig.AUDIT_TABLE))
#              .filter(F.col("parsed_successfully") == False))
