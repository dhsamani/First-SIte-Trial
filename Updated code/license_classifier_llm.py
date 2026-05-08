
# ============================================================
# LLM-Powered License Tier Classifier — PySpark / Databricks
# ============================================================
# Problem: 10,000+ raw license names → auto-classify into
#          (license_family, tier_level, tier_name, confidence)
#
# Strategy:
#   1. Deduplicate license names first (avoid LLM calls on dupes)
#   2. Batch deduplicated names → LLM inference (OpenAI / Azure OpenAI)
#   3. Parse & validate LLM JSON response
#   4. JOIN classifications back to full 10K+ dataset
#   5. Write results to license_tier_config Delta table
#   6. Flag low-confidence rows for human review
#
# Tier convention for Cisco Smart Licenses:
#   Network Essentials / DNA Essentials  → tier 1
#   Network Advantage  / DNA Advantage   → tier 2
#   DNA Premier / Premier                → tier 3
#   Unknown / Other                      → tier 0 (flagged)
# ============================================================

import os
import re
import json
import time
import math
import uuid
import hashlib
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, BooleanType,
)
from delta.tables import DeltaTable

# ── Optional: use openai SDK (pip install openai in Databricks) ──
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARN] openai package not installed. Run: %pip install openai")


# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
# ── Databricks Secrets (recommended over plaintext) ──
# Set via: databricks secrets put-secret --scope llm --key openai-key
LLM_PROVIDER      = "azure_openai"         # "openai" | "azure_openai"
OPENAI_API_KEY    = dbutils.secrets.get("llm", "openai-key")           # noqa
AZURE_ENDPOINT    = dbutils.secrets.get("llm", "azure-endpoint")       # noqa
AZURE_DEPLOYMENT  = "gpt-4o"               # your deployment name
AZURE_API_VERSION = "2024-08-01-preview"

# Fallback for plain OpenAI (non-Azure)
# OPENAI_API_KEY  = dbutils.secrets.get("llm", "openai-key")
# OPENAI_MODEL    = "gpt-4o"

INVENTORY_TABLE   = "catalog.schema.smart_license_inventory"
TIER_CONFIG_TABLE = "catalog.schema.license_tier_config"
REVIEW_TABLE      = "catalog.schema.license_tier_review"   # low-confidence rows

LLM_BATCH_SIZE    = 50      # license names per LLM call (keeps prompts focused)
MAX_RETRIES       = 3       # retry on LLM API errors
CONFIDENCE_FLOOR  = 0.70    # below this → flag for human review

spark = SparkSession.builder.getOrCreate()


# ─────────────────────────────────────────────────────────────
# 2. LLM CLIENT FACTORY
# ─────────────────────────────────────────────────────────────
def get_llm_client():
    """Returns Azure OpenAI or OpenAI client based on config."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed.")
    if LLM_PROVIDER == "azure_openai":
        return AzureOpenAI(
            api_key=OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────
# 3. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a Cisco Smart License expert. Your task is to classify Cisco license
names into structured tier metadata.

TIER DEFINITIONS (use these exact values):
  Tier 1 — Essentials  : Network Essentials, DNA Essentials, Base, Routing Essentials
  Tier 2 — Advantage   : Network Advantage, DNA Advantage, Advanced, Routing Advantage
  Tier 3 — Premier     : DNA Premier, Premier, Routing Premier
  Tier 0 — Unknown     : Cannot determine from the name alone

FAMILY EXAMPLES:
  ISE        → Identity Services Engine licenses
  DNA        → Digital Network Architecture (switching/wireless)
  ROUTING    → SD-WAN / routing platform licenses
  CATALYST   → Catalyst hardware-tied licenses (C9300, C9600, etc.)
  FIREWALL   → Firepower, ASA licenses
  WIRELESS   → AP, Wireless LAN Controller licenses
  OTHER      → Anything that doesn't fit above families

RULES:
  - Output ONLY valid JSON. No explanations outside JSON.
  - Confidence: 1.0 = certain, 0.0 = pure guess.
  - If a license name contains both a platform (C9300L) and tier keyword
    (DNA Essentials), use the tier keyword, not the platform, for tier assignment.
  - "Routing Network Essentials" = family ROUTING, tier 1.
  - Return exactly one JSON object per input license name.
""".strip()


def build_user_prompt(license_names: list[str]) -> str:
    names_json = json.dumps(license_names, indent=2)
    return f"""
Classify each of the following Cisco license names.

Input license names:
{names_json}

Return a JSON array — one object per license name, IN THE SAME ORDER:
[
  {{
    "license_name":    "<exact input name>",
    "license_family":  "<ISE|DNA|ROUTING|CATALYST|FIREWALL|WIRELESS|OTHER>",
    "tier_level":      <0|1|2|3>,
    "tier_name":       "<Unknown|Essentials|Advantage|Premier>",
    "confidence":      <0.0 to 1.0>,
    "reasoning":       "<one sentence explaining the classification>"
  }},
  ...
]
""".strip()


# ─────────────────────────────────────────────────────────────
# 4. LLM INFERENCE — single batch call with retry
# ─────────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    license_name:    str
    license_family:  str
    tier_level:      int
    tier_name:       str
    confidence:      float
    reasoning:       str
    needs_review:    bool
    raw_response:    Optional[str] = None


def call_llm_batch(
    client,
    license_names: list[str],
) -> list[ClassificationResult]:
    """
    Send one batch of license names to LLM.
    Retries up to MAX_RETRIES on failure.
    Returns ClassificationResult per name.
    """
    deployment = AZURE_DEPLOYMENT if LLM_PROVIDER == "azure_openai" else "gpt-4o"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(license_names)},
                ],
                temperature=0.0,         # deterministic — no creativity needed
                response_format={"type": "json_object"},
                max_tokens=4096,
                timeout=60,
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # LLM may return {"results": [...]} or just [...]
            if isinstance(parsed, dict):
                items = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
            else:
                items = parsed

            results = []
            for item in items:
                conf = float(item.get("confidence", 0.5))
                results.append(ClassificationResult(
                    license_name=item.get("license_name", ""),
                    license_family=item.get("license_family", "OTHER"),
                    tier_level=int(item.get("tier_level", 0)),
                    tier_name=item.get("tier_name", "Unknown"),
                    confidence=conf,
                    reasoning=item.get("reasoning", ""),
                    needs_review=(conf < CONFIDENCE_FLOOR),
                    raw_response=raw,
                ))
            return results

        except json.JSONDecodeError as e:
            print(f"[WARN] Attempt {attempt}: JSON parse error — {e}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)

        except Exception as e:
            print(f"[WARN] Attempt {attempt}: LLM call failed — {e}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)

    # Fallback: return Unknown for all names in this batch
    return [
        ClassificationResult(
            license_name=n,
            license_family="OTHER",
            tier_level=0,
            tier_name="Unknown",
            confidence=0.0,
            reasoning="LLM call failed after retries",
            needs_review=True,
        )
        for n in license_names
    ]


# ─────────────────────────────────────────────────────────────
# 5. DEDUPLICATE → BATCH → CLASSIFY
# ─────────────────────────────────────────────────────────────
def classify_all_licenses(
    spark: SparkSession,
    inventory_table: str = INVENTORY_TABLE,
    existing_config_table: Optional[str] = TIER_CONFIG_TABLE,
) -> list[ClassificationResult]:
    """
    1. Pull distinct license names from inventory
    2. Subtract names already in tier_config (avoid re-classifying known ones)
    3. Split into LLM_BATCH_SIZE chunks → call LLM
    4. Return all results
    """
    # ── Step A: Get distinct names from inventory ──
    all_names_df = spark.table(inventory_table).select("license_name").distinct()

    # ── Step B: Exclude already-classified names ──
    try:
        known_df = spark.table(existing_config_table).select("license_name")
        new_names_df = all_names_df.join(known_df, on="license_name", how="left_anti")
    except Exception:
        print(f"[INFO] {existing_config_table} not found — classifying all names.")
        new_names_df = all_names_df

    # ── Step C: Collect to driver (deduplicated — always small) ──
    unique_names = [row["license_name"] for row in new_names_df.collect()]
    print(f"[INFO] {len(unique_names)} unique license name(s) to classify.")

    if not unique_names:
        print("[INFO] All license names already classified.")
        return []

    # ── Step D: Batch into chunks → LLM ──
    client  = get_llm_client()
    results = []
    batches = math.ceil(len(unique_names) / LLM_BATCH_SIZE)

    for i in range(batches):
        batch = unique_names[i * LLM_BATCH_SIZE : (i + 1) * LLM_BATCH_SIZE]
        print(f"[LLM] Batch {i+1}/{batches} — {len(batch)} names...")
        batch_results = call_llm_batch(client, batch)
        results.extend(batch_results)
        time.sleep(0.5)   # gentle rate limiting

    return results


# ─────────────────────────────────────────────────────────────
# 6. WRITE RESULTS
#    a) Confident results → license_tier_config (MERGE)
#    b) Low-confidence   → license_tier_review  (for human QA)
# ─────────────────────────────────────────────────────────────
TIER_CONFIG_SCHEMA = StructType([
    StructField("license_name",    StringType(),  False),
    StructField("license_family",  StringType(),  True),
    StructField("tier_level",      IntegerType(), True),
    StructField("tier_name",       StringType(),  True),
    StructField("confidence",      FloatType(),   True),
    StructField("reasoning",       StringType(),  True),
    StructField("is_active",       BooleanType(), True),
    StructField("classified_by",   StringType(),  True),
    StructField("classified_at",   StringType(),  True),   # ISO timestamp string
])

REVIEW_SCHEMA = StructType([
    StructField("license_name",    StringType(),  False),
    StructField("suggested_family",StringType(),  True),
    StructField("suggested_tier",  IntegerType(), True),
    StructField("suggested_name",  StringType(),  True),
    StructField("confidence",      FloatType(),   True),
    StructField("reasoning",       StringType(),  True),
    StructField("review_status",   StringType(),  True),   # 'PENDING' | 'APPROVED' | 'REJECTED'
    StructField("created_at",      StringType(),  True),
])


def write_tier_config(
    spark: SparkSession,
    results: list[ClassificationResult],
    batch_id: str,
) -> tuple[int, int]:
    """
    Splits results into confident vs. review-needed.
    Returns (confident_count, review_count).
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    confident  = [r for r in results if not r.needs_review]
    needs_review = [r for r in results if r.needs_review]

    # ── Write confident to license_tier_config via MERGE ──
    if confident:
        conf_rows = [
            (
                r.license_name, r.license_family, r.tier_level,
                r.tier_name, r.confidence, r.reasoning,
                True, f"llm-batch:{batch_id}", now,
            )
            for r in confident
        ]
        conf_df = spark.createDataFrame(conf_rows, schema=TIER_CONFIG_SCHEMA)

        try:
            delta_tbl = DeltaTable.forName(spark, TIER_CONFIG_TABLE)
            (
                delta_tbl.alias("tgt")
                .merge(conf_df.alias("src"), "tgt.license_name = src.license_name")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        except Exception:
            # Table doesn't exist yet → create it
            conf_df.write.format("delta").mode("overwrite") \
                   .option("mergeSchema", "true") \
                   .saveAsTable(TIER_CONFIG_TABLE)

        print(f"[INFO] {len(confident)} classification(s) written to {TIER_CONFIG_TABLE}.")

    # ── Write low-confidence to review table ──
    if needs_review:
        review_rows = [
            (
                r.license_name, r.license_family, r.tier_level,
                r.tier_name, r.confidence, r.reasoning,
                "PENDING", now,
            )
            for r in needs_review
        ]
        review_df = spark.createDataFrame(review_rows, schema=REVIEW_SCHEMA)
        review_df.write.format("delta").mode("append") \
                 .option("mergeSchema", "true") \
                 .saveAsTable(REVIEW_TABLE)
        print(f"[INFO] {len(needs_review)} low-confidence name(s) flagged → {REVIEW_TABLE}.")

    return len(confident), len(needs_review)


# ─────────────────────────────────────────────────────────────
# 7. HUMAN REVIEW HELPER — approve / reject from review table
#    Run this in a separate notebook cell after QA
# ─────────────────────────────────────────────────────────────
def promote_reviewed_licenses(
    spark: SparkSession,
    approved_overrides: Optional[dict] = None,
) -> None:
    """
    Moves APPROVED rows from review table → tier_config table.

    approved_overrides: optional dict to manually fix classifications
        {
          "C9600 Network Advantage": {
              "license_family": "CATALYST",
              "tier_level": 2,
              "tier_name": "Advantage"
          }
        }
    """
    review_df = spark.table(REVIEW_TABLE).filter(
        F.col("review_status") == "APPROVED"
    )

    if approved_overrides:
        # Apply manual corrections for specific license names
        override_rows = [
            (lic, meta["license_family"], meta["tier_level"],
             meta["tier_name"], 1.0, "manual_override", True, "human", None)
            for lic, meta in approved_overrides.items()
        ]
        override_df = spark.createDataFrame(override_rows, schema=TIER_CONFIG_SCHEMA)
        review_df = review_df.unionByName(override_df, allowMissingColumns=True)

    # MERGE into tier_config
    delta_tbl = DeltaTable.forName(spark, TIER_CONFIG_TABLE)
    (
        delta_tbl.alias("tgt")
        .merge(review_df.alias("src"), "tgt.license_name = src.license_name")
        .whenMatchedUpdate(set={
            "license_family": "src.suggested_family",
            "tier_level":     "src.suggested_tier",
            "tier_name":      "src.suggested_name",
            "confidence":     F.lit(1.0),
            "classified_by":  F.lit("human_reviewed"),
        })
        .whenNotMatchedInsert(values={
            "license_name":   "src.license_name",
            "license_family": "src.suggested_family",
            "tier_level":     "src.suggested_tier",
            "tier_name":      "src.suggested_name",
            "confidence":     F.lit(1.0),
            "is_active":      F.lit(True),
            "classified_by":  F.lit("human_reviewed"),
        })
        .execute()
    )
    print("[INFO] Approved reviews promoted to tier_config.")


# ─────────────────────────────────────────────────────────────
# 8. JOIN TIER CONFIG BACK TO INVENTORY
#    After classification, enrich the inventory table with
#    tier metadata so the transfer engine can use it directly
# ─────────────────────────────────────────────────────────────
def enrich_inventory_with_tiers(spark: SparkSession) -> None:
    """
    LEFT JOINs tier_config onto smart_license_inventory and
    writes license_family + tier_level back via Delta MERGE.
    Rows with no match get tier_level=0 and family='UNKNOWN'.
    """
    inventory_df = spark.table(INVENTORY_TABLE)
    tier_df = spark.table(TIER_CONFIG_TABLE).select(
        "license_name", "license_family", "tier_level", "tier_name"
    )

    enriched = (
        inventory_df
        .join(tier_df, on="license_name", how="left")
        .withColumn("license_family",
                    F.coalesce(F.col("license_family"), F.lit("UNKNOWN")))
        .withColumn("tier_level",
                    F.coalesce(F.col("tier_level"), F.lit(0)))
    )

    delta_tbl = DeltaTable.forName(spark, INVENTORY_TABLE)
    (
        delta_tbl.alias("tgt")
        .merge(
            enriched.alias("src"),
            "tgt.account_id = src.account_id AND tgt.license_name = src.license_name",
        )
        .whenMatchedUpdate(set={
            "license_family": "src.license_family",
            "tier_level":     "src.tier_level",
            "updated_at":     F.current_timestamp(),
        })
        .execute()
    )
    print("[INFO] Inventory enriched with tier metadata.")


# ─────────────────────────────────────────────────────────────
# 9. MAIN — Full classification pipeline
# ─────────────────────────────────────────────────────────────
def run_license_classification(spark: SparkSession) -> None:
    """
    Full pipeline:
      1. Deduplicate license names from inventory
      2. Skip already-classified names
      3. Batch → LLM classification
      4. Write confident results to tier_config
      5. Flag low-confidence to review table
      6. Enrich inventory with tier metadata
    """
    batch_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*60}")
    print(f"  License Classification Run  |  batch={batch_id}")
    print(f"{'='*60}")

    print("[STEP 1] Classifying license names via LLM...")
    results = classify_all_licenses(spark)

    if not results:
        print("[INFO] Nothing to classify.")
        return

    print(f"[STEP 2] Writing {len(results)} classification(s)...")
    confident_n, review_n = write_tier_config(spark, results, batch_id)

    print(f"[STEP 3] Enriching inventory with tier metadata...")
    enrich_inventory_with_tiers(spark)

    print(f"\n[DONE] batch={batch_id}")
    print(f"       ✅ Confident : {confident_n}")
    print(f"       ⚠️  Review    : {review_n}  → check {REVIEW_TABLE}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_license_classification(spark)

# ─────────────────────────────────────────────────────────────
# EXAMPLE: Manual override after reviewing flagged names
# ─────────────────────────────────────────────────────────────
# promote_reviewed_licenses(
#     spark,
#     approved_overrides={
#         "C9600 Network Advantage": {
#             "license_family": "CATALYST",
#             "tier_level": 2,
#             "tier_name": "Advantage",
#         },
#         "Routing Network Essentials: Tier 2": {
#             "license_family": "ROUTING",
#             "tier_level": 1,
#             "tier_name": "Essentials",
#         },
#     }
# )
