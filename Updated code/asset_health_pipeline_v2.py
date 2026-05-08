# =============================================================================
# Asset Health Score – Feature Engineering, LLM Recommendations & Linkback
# Compatible with: Databricks (DBR 13.x+), PySpark 3.x
# Source Table  : asset_health_score_features
# Unique Key    : company_id + serial_number
#
# Architecture:
#   1.  Load & clean source table
#   2.  Engineer features from lifecycle + energy/carbon columns
#   3.  Build rule-based overlapping recommendation buckets
#       (one asset CAN appear in multiple buckets, e.g. 30-day AND 60-day)
#   4.  Explode buckets → one row per (asset, recommendation_type)
#   5.  Group buckets → LLM generates a narrative per bucket
#   6.  Join LLM recommendations back to original table rows
#   7.  Aggregation: bottom-N by ML score
#   8.  Write all outputs as Delta tables
# =============================================================================

import os
import json
import textwrap
import time
from functools import partial

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, IntegerType, DoubleType,
    ArrayType, StructType, StructField
)

# ── Databricks Foundation Model API (OpenAI-compatible) ─────────────────────
from openai import OpenAI

# =============================================================================
# CONFIG  – edit these before running
# =============================================================================

SOURCE_TABLE      = "asset_health_score_features"
OUTPUT_DATABASE   = "asset_health"          # target Unity Catalog schema / Hive db

# Databricks workspace + token (stored in Databricks secrets – best practice)
# Replace scope/key names with your Databricks secret scope
DATABRICKS_HOST   = dbutils.secrets.get(scope="llm", key="databricks_host")   # e.g. https://adb-xxx.azuredatabricks.net
DATABRICKS_TOKEN  = dbutils.secrets.get(scope="llm", key="databricks_token")

# Foundation Model to use (pay-per-token, no cluster needed)
LLM_MODEL         = "databricks-meta-llama-3-1-70b-instruct"   # or databricks-dbrx-instruct
LLM_MAX_TOKENS    = 512
LLM_TEMPERATURE   = 0.2    # low = deterministic, focused recommendations

# Urgency thresholds (days)
THRESHOLD_30      = 30
THRESHOLD_60      = 60
THRESHOLD_90      = 90
THRESHOLD_180     = 180
THRESHOLD_365     = 365

# Percentile flags
ENERGY_HIGH_PCT   = 0.75
CARBON_HIGH_PCT   = 0.75
BOTTOM_N          = 10

# =============================================================================
# 0. Spark Session
# =============================================================================
spark = SparkSession.builder.appName("AssetHealthPipeline_v2").getOrCreate()
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

spark.sql(f"CREATE DATABASE IF NOT EXISTS {OUTPUT_DATABASE}")

# =============================================================================
# 1. Load & Standardise Source Table
# =============================================================================
df_raw = (
    spark.table(SOURCE_TABLE)
    # Fix column name typo in source
    .withColumnRenamed("endofsaledaysremainig",  "endofsaledaysremaining")
    # Fix spaces in column names (if any)
    .withColumnRenamed("estimated energy use",   "estimated_energy_use")
    .withColumnRenamed("carbon usage",           "carbon_usage")
    .withColumnRenamed("ceo score",              "ceo_score")
    .withColumnRenamed("cfo score",              "cfo_score")
    .withColumnRenamed("CIO score",              "cio_score")
    .withColumnRenamed("security score",         "security_score")
    .withColumnRenamed("sustainability score",   "sustainability_score")
)

# Guarantee numeric types
LIFECYCLE_COLS = [
    "end_of_support_days_remaining",
    "endofsaledaysremaining",
    "endoflifedaysremaining",
    "contractdaysremaining",
    "estimated_energy_use",
    "carbon_usage",
]
for c in LIFECYCLE_COLS:
    df_raw = df_raw.withColumn(c, F.col(c).cast(DoubleType()))

df_raw.cache()
print(f"[INFO] Raw row count: {df_raw.count()}")

# =============================================================================
# 2. Feature Engineering
# =============================================================================

# ── 2a. Urgency bucket per lifecycle column ───────────────────────────────────
def urgency_bucket(col_name: str):
    return (
        F.when(F.col(col_name) < 0,              "ALREADY_EXPIRED")
         .when(F.col(col_name) <= THRESHOLD_30,  "CRITICAL_30D")
         .when(F.col(col_name) <= THRESHOLD_60,  "CRITICAL_60D")
         .when(F.col(col_name) <= THRESHOLD_90,  "CRITICAL_90D")
         .when(F.col(col_name) <= THRESHOLD_180, "WARNING_180D")
         .when(F.col(col_name) <= THRESHOLD_365, "MODERATE_365D")
         .otherwise("OK")
    )

df_feat = (
    df_raw
    # Urgency buckets
    .withColumn("eos_urgency",       urgency_bucket("end_of_support_days_remaining"))
    .withColumn("eosa_urgency",      urgency_bucket("endofsaledaysremaining"))
    .withColumn("eol_urgency",       urgency_bucket("endoflifedaysremaining"))
    .withColumn("contract_urgency",  urgency_bucket("contractdaysremaining"))

    # Binary flags – each threshold band is separate (enables overlap)
    .withColumn("is_eol_expired",    (F.col("endoflifedaysremaining") < 0).cast("int"))
    .withColumn("is_eol_30d",        F.col("endoflifedaysremaining").between(0, THRESHOLD_30).cast("int"))
    .withColumn("is_eol_60d",        F.col("endoflifedaysremaining").between(0, THRESHOLD_60).cast("int"))
    .withColumn("is_eol_90d",        F.col("endoflifedaysremaining").between(0, THRESHOLD_90).cast("int"))

    .withColumn("is_eosa_expired",   (F.col("endofsaledaysremaining") < 0).cast("int"))
    .withColumn("is_eosa_30d",       F.col("endofsaledaysremaining").between(0, THRESHOLD_30).cast("int"))
    .withColumn("is_eosa_60d",       F.col("endofsaledaysremaining").between(0, THRESHOLD_60).cast("int"))
    .withColumn("is_eosa_90d",       F.col("endofsaledaysremaining").between(0, THRESHOLD_90).cast("int"))

    .withColumn("is_eos_expired",    (F.col("end_of_support_days_remaining") < 0).cast("int"))
    .withColumn("is_eos_30d",        F.col("end_of_support_days_remaining").between(0, THRESHOLD_30).cast("int"))
    .withColumn("is_eos_60d",        F.col("end_of_support_days_remaining").between(0, THRESHOLD_60).cast("int"))
    .withColumn("is_eos_90d",        F.col("end_of_support_days_remaining").between(0, THRESHOLD_90).cast("int"))

    .withColumn("is_contract_expired", (F.col("contractdaysremaining") < 0).cast("int"))
    .withColumn("is_contract_30d",   F.col("contractdaysremaining").between(0, THRESHOLD_30).cast("int"))
    .withColumn("is_contract_60d",   F.col("contractdaysremaining").between(0, THRESHOLD_60).cast("int"))
    .withColumn("is_contract_90d",   F.col("contractdaysremaining").between(0, THRESHOLD_90).cast("int"))

    # Composite lifecycle risk (raw sum; lower = more at risk)
    .withColumn(
        "lifecycle_risk_score",
        (
            F.coalesce("end_of_support_days_remaining", F.lit(0)) +
            F.coalesce("endofsaledaysremaining",        F.lit(0)) +
            F.coalesce("endoflifedaysremaining",        F.lit(0)) +
            F.coalesce("contractdaysremaining",         F.lit(0))
        ).cast(DoubleType())
    )
    # How many lifecycle dimensions are in CRITICAL_90D or worse
    .withColumn(
        "critical_dimension_count",
        (
            F.col("is_eol_90d")      + F.col("is_eol_expired")      +
            F.col("is_eosa_90d")     + F.col("is_eosa_expired")     +
            F.col("is_eos_90d")      + F.col("is_eos_expired")      +
            F.col("is_contract_90d") + F.col("is_contract_expired")
        )
    )
    # Energy & carbon transforms
    .withColumn("log_energy_use",    F.log1p(F.col("estimated_energy_use")))
    .withColumn("log_carbon_usage",  F.log1p(F.col("carbon_usage")))
    .withColumn(
        "energy_carbon_ratio",
        F.when(F.col("carbon_usage") > 0,
               F.col("estimated_energy_use") / F.col("carbon_usage"))
         .otherwise(F.lit(None).cast(DoubleType()))
    )
)

# ── 2b. Company-level window features ────────────────────────────────────────
w_co = Window.partitionBy("company_id")

df_feat = (
    df_feat
    .withColumn("co_avg_energy",         F.avg("estimated_energy_use").over(w_co))
    .withColumn("co_avg_carbon",         F.avg("carbon_usage").over(w_co))
    .withColumn("co_asset_count",        F.count("serial_number").over(w_co))
    .withColumn("co_critical_assets",    F.sum("critical_dimension_count").over(w_co))
    .withColumn("energy_pct_co",         F.percent_rank().over(w_co.orderBy("estimated_energy_use")))
    .withColumn("carbon_pct_co",         F.percent_rank().over(w_co.orderBy("carbon_usage")))
    .withColumn("energy_vs_co_avg",      F.col("estimated_energy_use") - F.col("co_avg_energy"))
    .withColumn("carbon_vs_co_avg",      F.col("carbon_usage") - F.col("co_avg_carbon"))
    .withColumn("is_high_energy",        (F.col("energy_pct_co") >= ENERGY_HIGH_PCT).cast("int"))
    .withColumn("is_high_carbon",        (F.col("carbon_pct_co") >= CARBON_HIGH_PCT).cast("int"))
)

# Global percentile ranks
df_feat = (
    df_feat
    .withColumn("global_energy_pct",
                F.percent_rank().over(Window.orderBy("estimated_energy_use")))
    .withColumn("global_carbon_pct",
                F.percent_rank().over(Window.orderBy("carbon_usage")))
    .withColumn("global_lifecycle_pct",
                F.percent_rank().over(Window.orderBy("lifecycle_risk_score")))
)

df_feat.cache()

# =============================================================================
# 3. Overlapping Recommendation Buckets
#    Each asset gets an ARRAY of (bucket_code, bucket_label, days_remaining)
#    so that a device expiring in 30 days appears in BOTH the 30d AND 60d bucket.
#    F.explode later creates one row per bucket membership.
# =============================================================================

# Helper: produce a struct(bucket_code, bucket_label, days_value)
def rec_struct(code, label, days_col):
    return F.struct(
        F.lit(code).alias("bucket_code"),
        F.lit(label).alias("bucket_label"),
        F.col(days_col).alias("days_remaining")
    )

df_buckets = (
    df_feat
    .withColumn(
        "rec_buckets",
        F.array_compact(   # removes NULLs (only available in Spark 3.4+ / DBR 13+)
            F.array(
                # ── End-of-Sale ──────────────────────────────────────────────
                F.when(F.col("is_eosa_expired") == 1,
                       rec_struct("EOSA_EXPIRED",
                                  "End-of-Sale ALREADY EXPIRED",
                                  "endofsaledaysremaining")),
                F.when(F.col("is_eosa_30d") == 1,
                       rec_struct("EOSA_30D",
                                  "End-of-Sale expiring within 30 days",
                                  "endofsaledaysremaining")),
                F.when(F.col("is_eosa_60d") == 1,
                       rec_struct("EOSA_60D",
                                  "End-of-Sale expiring within 60 days",
                                  "endofsaledaysremaining")),
                F.when(F.col("is_eosa_90d") == 1,
                       rec_struct("EOSA_90D",
                                  "End-of-Sale expiring within 90 days",
                                  "endofsaledaysremaining")),

                # ── End-of-Life ───────────────────────────────────────────────
                F.when(F.col("is_eol_expired") == 1,
                       rec_struct("EOL_EXPIRED",
                                  "End-of-Life ALREADY EXPIRED",
                                  "endoflifedaysremaining")),
                F.when(F.col("is_eol_30d") == 1,
                       rec_struct("EOL_30D",
                                  "End-of-Life expiring within 30 days",
                                  "endoflifedaysremaining")),
                F.when(F.col("is_eol_60d") == 1,
                       rec_struct("EOL_60D",
                                  "End-of-Life expiring within 60 days",
                                  "endoflifedaysremaining")),
                F.when(F.col("is_eol_90d") == 1,
                       rec_struct("EOL_90D",
                                  "End-of-Life expiring within 90 days",
                                  "endoflifedaysremaining")),

                # ── End-of-Support ────────────────────────────────────────────
                F.when(F.col("is_eos_expired") == 1,
                       rec_struct("EOS_EXPIRED",
                                  "End-of-Support ALREADY EXPIRED",
                                  "end_of_support_days_remaining")),
                F.when(F.col("is_eos_30d") == 1,
                       rec_struct("EOS_30D",
                                  "End-of-Support expiring within 30 days",
                                  "end_of_support_days_remaining")),
                F.when(F.col("is_eos_60d") == 1,
                       rec_struct("EOS_60D",
                                  "End-of-Support expiring within 60 days",
                                  "end_of_support_days_remaining")),
                F.when(F.col("is_eos_90d") == 1,
                       rec_struct("EOS_90D",
                                  "End-of-Support expiring within 90 days",
                                  "end_of_support_days_remaining")),

                # ── Contract ──────────────────────────────────────────────────
                F.when(F.col("is_contract_expired") == 1,
                       rec_struct("CONTRACT_EXPIRED",
                                  "Contract ALREADY EXPIRED",
                                  "contractdaysremaining")),
                F.when(F.col("is_contract_30d") == 1,
                       rec_struct("CONTRACT_30D",
                                  "Contract expiring within 30 days",
                                  "contractdaysremaining")),
                F.when(F.col("is_contract_60d") == 1,
                       rec_struct("CONTRACT_60D",
                                  "Contract expiring within 60 days",
                                  "contractdaysremaining")),
                F.when(F.col("is_contract_90d") == 1,
                       rec_struct("CONTRACT_90D",
                                  "Contract expiring within 90 days",
                                  "contractdaysremaining")),

                # ── Energy / Carbon ────────────────────────────────────────────
                F.when((F.col("is_high_energy") == 1) & (F.col("global_energy_pct") >= 0.90),
                       rec_struct("ENERGY_TOP10",
                                  "Top 10% energy consumer globally",
                                  "estimated_energy_use")),
                F.when(F.col("is_high_energy") == 1,
                       rec_struct("ENERGY_HIGH",
                                  "Top 25% energy consumer in company",
                                  "estimated_energy_use")),
                F.when((F.col("is_high_carbon") == 1) & (F.col("global_carbon_pct") >= 0.90),
                       rec_struct("CARBON_TOP10",
                                  "Top 10% carbon emitter globally",
                                  "carbon_usage")),
                F.when(F.col("is_high_carbon") == 1,
                       rec_struct("CARBON_HIGH",
                                  "Top 25% carbon emitter in company",
                                  "carbon_usage")),
            )
        )
    )
)

# NOTE: For Spark < 3.4 (DBR < 13), replace array_compact with filter:
# .withColumn("rec_buckets", F.filter(F.col("rec_buckets"), lambda x: x.isNotNull()))

# =============================================================================
# 4. Explode → one row per (asset × bucket)
#    This is the "long" table – every asset reappears for each bucket it belongs to
# =============================================================================
df_exploded = (
    df_buckets
    .withColumn("rec", F.explode("rec_buckets"))
    .select(
        "company_id", "serial_number",
        F.col("rec.bucket_code").alias("bucket_code"),
        F.col("rec.bucket_label").alias("bucket_label"),
        F.col("rec.days_remaining").alias("days_remaining"),
        "lifecycle_risk_score", "critical_dimension_count",
        "estimated_energy_use", "carbon_usage",
        # pass-through ML scores
        "ceo_score", "cfo_score", "cio_score", "security_score", "sustainability_score",
    )
)

# =============================================================================
# 5. Per-Bucket Summary → feed to LLM
#    For each bucket_code we compute: asset count, min/max/avg days remaining,
#    and collect a sample of (company_id, serial_number, days_remaining) tuples.
# =============================================================================

df_bucket_summary = (
    df_exploded
    .groupBy("bucket_code", "bucket_label")
    .agg(
        F.count("serial_number")        .alias("asset_count"),
        F.min("days_remaining")         .alias("min_days"),
        F.max("days_remaining")         .alias("max_days"),
        F.avg("days_remaining")         .alias("avg_days"),
        F.collect_list(
            F.struct("company_id", "serial_number", "days_remaining")
        ).alias("asset_sample_raw")
    )
    # Keep at most 20 sample assets for the LLM prompt (token budget)
    .withColumn("asset_sample",
                F.slice(F.col("asset_sample_raw"), 1, 20))
    .drop("asset_sample_raw")
    .orderBy(F.col("asset_count").desc())
)

# Collect to driver – this is safe because #buckets is small (~16)
bucket_rows = df_bucket_summary.collect()
print(f"[INFO] Distinct recommendation buckets: {len(bucket_rows)}")

# =============================================================================
# 6. LLM Call – generate a narrative recommendation per bucket
# =============================================================================

llm_client = OpenAI(
    api_key  = DATABRICKS_TOKEN,
    base_url = f"{DATABRICKS_HOST}/serving-endpoints",
)

SYSTEM_PROMPT = """You are an IT Asset Management advisor for enterprise infrastructure.
Your role is to write clear, actionable, executive-ready recommendations for batches of
network/IT assets that are at risk due to lifecycle or environmental thresholds.
Keep each recommendation concise (3–5 sentences), specific, and prioritised by urgency.
Always mention the exact asset count and timeframe when relevant.
Output ONLY the recommendation text – no JSON, no bullet headers, no markdown."""

def call_llm(bucket_code: str, bucket_label: str, asset_count: int,
             min_days: float, max_days: float, avg_days: float,
             sample_assets: list, retries: int = 3) -> str:
    """Call Databricks Foundation Model API and return recommendation text."""
    sample_lines = "\n".join(
        f"  - Company: {r['company_id']}, Serial: {r['serial_number']}, "
        f"Days Remaining: {int(r['days_remaining']) if r['days_remaining'] is not None else 'N/A'}"
        for r in (sample_assets[:10])   # max 10 in prompt
    )

    user_prompt = textwrap.dedent(f"""
        Recommendation bucket  : {bucket_label}
        Bucket code            : {bucket_code}
        Total affected assets  : {asset_count}
        Days remaining range   : {int(min_days) if min_days is not None else 'N/A'} – {int(max_days) if max_days is not None else 'N/A'}
        Average days remaining : {round(avg_days, 1) if avg_days is not None else 'N/A'}

        Sample affected assets (up to 10 shown):
        {sample_lines}

        Write a targeted recommendation for the IT/procurement team covering:
        1. The urgency and business risk of this specific expiry/threshold.
        2. Immediate actions to take (within the next 2 weeks).
        3. Strategic follow-up actions (within 30–90 days).
        Do NOT list the individual assets – the recipient already has the full list.
    """).strip()

    for attempt in range(retries):
        try:
            response = llm_client.chat.completions.create(
                model       = LLM_MODEL,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens  = LLM_MAX_TOKENS,
                temperature = LLM_TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"[WARN] LLM call failed (attempt {attempt+1}): {e} – retrying in 3s")
                time.sleep(3)
            else:
                print(f"[ERROR] LLM call failed after {retries} attempts: {e}")
                return f"[LLM unavailable] {bucket_label}: {asset_count} assets affected. Manual review required."


# Drive LLM calls from driver (sequential, but bucket count is small ~16)
llm_results = []
for row in bucket_rows:
    print(f"[LLM] Calling for bucket: {row['bucket_code']} ({row['asset_count']} assets)")
    rec_text = call_llm(
        bucket_code  = row["bucket_code"],
        bucket_label = row["bucket_label"],
        asset_count  = row["asset_count"],
        min_days     = row["min_days"],
        max_days     = row["max_days"],
        avg_days     = row["avg_days"],
        sample_assets= row["asset_sample"],
    )
    llm_results.append({
        "bucket_code"        : row["bucket_code"],
        "bucket_label"       : row["bucket_label"],
        "asset_count"        : row["asset_count"],
        "min_days"           : row["min_days"],
        "max_days"           : row["max_days"],
        "avg_days"           : row["avg_days"],
        "llm_recommendation" : rec_text,
    })
    print(f"  → Done: {rec_text[:120]}...")

# Build Spark DataFrame of LLM outputs
llm_schema = StructType([
    StructField("bucket_code",         StringType(), True),
    StructField("bucket_label",        StringType(), True),
    StructField("asset_count",         IntegerType(), True),
    StructField("min_days",            DoubleType(), True),
    StructField("max_days",            DoubleType(), True),
    StructField("avg_days",            DoubleType(), True),
    StructField("llm_recommendation",  StringType(), True),
])
df_llm_recs = spark.createDataFrame(llm_results, schema=llm_schema)

# =============================================================================
# 7. Link LLM Recommendations back to original asset table
#    Join path:  df_exploded (asset × bucket)  ←→  df_llm_recs (bucket × rec)
#    Result    :  one row per (asset × bucket) carrying both asset data + LLM text
#    → Same asset WILL appear multiple times if it belongs to multiple buckets ✓
# =============================================================================

df_asset_recommendations = (
    df_exploded
    .join(df_llm_recs.select("bucket_code", "bucket_label", "llm_recommendation"),
          on="bucket_code", how="left")
    # Re-attach ALL original source columns via join on unique key
    .join(
        df_raw.select(
            "company_id", "serial_number",
            "end_of_support_days_remaining", "endofsaledaysremaining",
            "endoflifedaysremaining", "contractdaysremaining",
            "estimated_energy_use", "carbon_usage",
            "ceo_score", "cfo_score", "cio_score",
            "security_score", "sustainability_score",
        ),
        on=["company_id", "serial_number"],
        how="left"
    )
    .select(
        # Identity
        "company_id",
        "serial_number",
        # Which bucket triggered this row
        "bucket_code",
        "bucket_label",
        "days_remaining",       # the specific days value for THIS bucket's column
        # LLM narrative for this bucket
        "llm_recommendation",
        # Full lifecycle context (original values)
        "end_of_support_days_remaining",
        "endofsaledaysremaining",
        "endoflifedaysremaining",
        "contractdaysremaining",
        "estimated_energy_use",
        "carbon_usage",
        # ML scores (pass-through)
        "ceo_score", "cfo_score", "cio_score",
        "security_score", "sustainability_score",
        # Risk metadata
        "lifecycle_risk_score",
        "critical_dimension_count",
    )
    .orderBy("bucket_code", "days_remaining")
)

# =============================================================================
# 8. Bottom-N assets per ML score  (aggregation-only use of ML scores)
# =============================================================================

def bottom_n_assets(df_source, score_col: str, n: int = BOTTOM_N):
    """Return bottom-N assets globally ranked by score_col ascending."""
    w_rank = Window.orderBy(F.col(score_col).asc())
    return (
        df_source
        .withColumn("score_rank", F.dense_rank().over(w_rank))
        .filter(F.col("score_rank") <= n)
        .select(
            "company_id", "serial_number", score_col, "score_rank",
            "lifecycle_risk_score", "critical_dimension_count",
            "end_of_support_days_remaining", "endofsaledaysremaining",
            "endoflifedaysremaining", "contractdaysremaining",
        )
        .orderBy(F.col(score_col).asc())
    )


df_bottom_ceo          = bottom_n_assets(df_feat, "ceo_score")
df_bottom_cfo          = bottom_n_assets(df_feat, "cfo_score")
df_bottom_cio          = bottom_n_assets(df_feat, "cio_score")
df_bottom_security     = bottom_n_assets(df_feat, "security_score")
df_bottom_sustainability = bottom_n_assets(df_feat, "sustainability_score")

# =============================================================================
# 9. Company-Level Rollup Summary
# =============================================================================

df_company_summary = (
    df_feat
    .groupBy("company_id")
    .agg(
        F.count("serial_number")              .alias("total_assets"),
        F.sum("is_eol_expired")               .alias("eol_expired_count"),
        F.sum("is_eol_90d")                   .alias("eol_critical_90d"),
        F.sum("is_eosa_expired")              .alias("eosa_expired_count"),
        F.sum("is_eosa_90d")                  .alias("eosa_critical_90d"),
        F.sum("is_eos_expired")               .alias("eos_expired_count"),
        F.sum("is_eos_90d")                   .alias("eos_critical_90d"),
        F.sum("is_contract_expired")          .alias("contract_expired_count"),
        F.sum("is_contract_90d")              .alias("contract_critical_90d"),
        F.sum("is_high_energy")               .alias("high_energy_assets"),
        F.sum("is_high_carbon")               .alias("high_carbon_assets"),
        F.avg("lifecycle_risk_score")         .alias("avg_lifecycle_risk"),
        F.avg("estimated_energy_use")         .alias("avg_energy_use"),
        F.avg("carbon_usage")                 .alias("avg_carbon_usage"),
        F.avg("ceo_score")                    .alias("avg_ceo_score"),
        F.avg("cfo_score")                    .alias("avg_cfo_score"),
        F.avg("cio_score")                    .alias("avg_cio_score"),
        F.avg("security_score")               .alias("avg_security_score"),
        F.avg("sustainability_score")         .alias("avg_sustainability_score"),
        F.sum("critical_dimension_count")     .alias("total_critical_issues"),
    )
    .withColumn(
        "company_risk_tier",
        F.when(
            (F.col("eol_expired_count") + F.col("eosa_expired_count") + F.col("eos_expired_count")) >= 3,
            "CRITICAL"
        ).when(
            (F.col("eol_critical_90d") + F.col("eosa_critical_90d") + F.col("eos_critical_90d")) >= 5,
            "HIGH"
        ).when(
            (F.col("eol_critical_90d") + F.col("eosa_critical_90d")) >= 2,
            "MEDIUM"
        ).otherwise("LOW")
    )
    .orderBy(F.col("total_critical_issues").desc())
)

# =============================================================================
# 10. Write all outputs as Delta tables
# =============================================================================

def write_delta(df, table: str, mode: str = "overwrite"):
    (
        df.write
          .format("delta")
          .mode(mode)
          .option("overwriteSchema", "true")
          .saveAsTable(table)
    )
    print(f"[INFO] Written: {table}")


# Main feature table (one row per asset)
write_delta(
    df_feat.select(
        "company_id", "serial_number",
        "end_of_support_days_remaining", "endofsaledaysremaining",
        "endoflifedaysremaining", "contractdaysremaining",
        "estimated_energy_use", "carbon_usage",
        "eos_urgency", "eosa_urgency", "eol_urgency", "contract_urgency",
        "is_eol_expired", "is_eol_30d", "is_eol_60d", "is_eol_90d",
        "is_eosa_expired", "is_eosa_30d", "is_eosa_60d", "is_eosa_90d",
        "is_eos_expired", "is_eos_30d", "is_eos_60d", "is_eos_90d",
        "is_contract_expired", "is_contract_30d", "is_contract_60d", "is_contract_90d",
        "lifecycle_risk_score", "critical_dimension_count",
        "log_energy_use", "log_carbon_usage", "energy_carbon_ratio",
        "co_avg_energy", "co_avg_carbon", "co_asset_count", "co_critical_assets",
        "energy_vs_co_avg", "carbon_vs_co_avg",
        "is_high_energy", "is_high_carbon",
        "energy_pct_co", "carbon_pct_co",
        "global_energy_pct", "global_carbon_pct", "global_lifecycle_pct",
        # ML scores – pass-through only
        "ceo_score", "cfo_score", "cio_score", "security_score", "sustainability_score",
    ),
    f"{OUTPUT_DATABASE}.asset_health_features_v2"
)

# LLM recommendation narratives per bucket
write_delta(df_llm_recs,
            f"{OUTPUT_DATABASE}.asset_health_llm_bucket_recommendations")

# ★ KEY TABLE: every asset × every bucket it belongs to + LLM recommendation
write_delta(df_asset_recommendations,
            f"{OUTPUT_DATABASE}.asset_health_recommendations_linked")

# Company summary
write_delta(df_company_summary,
            f"{OUTPUT_DATABASE}.asset_health_company_summary_v2")

# Bottom-N by ML score
write_delta(df_bottom_ceo,           f"{OUTPUT_DATABASE}.bottom_n_ceo_score")
write_delta(df_bottom_cfo,           f"{OUTPUT_DATABASE}.bottom_n_cfo_score")
write_delta(df_bottom_cio,           f"{OUTPUT_DATABASE}.bottom_n_cio_score")
write_delta(df_bottom_security,      f"{OUTPUT_DATABASE}.bottom_n_security_score")
write_delta(df_bottom_sustainability,f"{OUTPUT_DATABASE}.bottom_n_sustainability_score")

# =============================================================================
# 11. Convenience Queries – display in notebook
# =============================================================================

print("\n" + "="*70)
print("SAMPLE: All assets expiring End-of-Sale within 60 days")
print("="*70)
spark.sql(f"""
    SELECT company_id, serial_number, bucket_code, days_remaining,
           endofsaledaysremaining, endoflifedaysremaining,
           llm_recommendation
    FROM   {OUTPUT_DATABASE}.asset_health_recommendations_linked
    WHERE  bucket_code IN ('EOSA_30D','EOSA_60D','EOSA_EXPIRED')
    ORDER  BY days_remaining ASC
""").show(40, truncate=80)


print("\n" + "="*70)
print("SAMPLE: Assets in MULTIPLE buckets (overlapping risk)")
print("="*70)
spark.sql(f"""
    SELECT company_id, serial_number,
           COUNT(DISTINCT bucket_code) AS bucket_count,
           COLLECT_LIST(bucket_code)   AS all_buckets
    FROM   {OUTPUT_DATABASE}.asset_health_recommendations_linked
    GROUP  BY company_id, serial_number
    HAVING COUNT(DISTINCT bucket_code) >= 2
    ORDER  BY bucket_count DESC
    LIMIT  20
""").show(truncate=80)


print("\n" + "="*70)
print("SAMPLE: LLM Recommendations per Bucket")
print("="*70)
spark.sql(f"""
    SELECT bucket_code, bucket_label, asset_count,
           ROUND(avg_days,1) AS avg_days,
           llm_recommendation
    FROM   {OUTPUT_DATABASE}.asset_health_llm_bucket_recommendations
    ORDER  BY asset_count DESC
""").show(truncate=100)


print("\n" + "="*70)
print("SAMPLE: Bottom-10 Assets by CEO Score")
print("="*70)
spark.sql(f"""
    SELECT company_id, serial_number, ceo_score, score_rank,
           lifecycle_risk_score, endoflifedaysremaining
    FROM   {OUTPUT_DATABASE}.bottom_n_ceo_score
    ORDER  BY ceo_score ASC
""").show(truncate=60)

# =============================================================================
# 12. Cleanup
# =============================================================================
df_raw.unpersist()
df_feat.unpersist()
print("\n[INFO] Pipeline v2 complete.")
