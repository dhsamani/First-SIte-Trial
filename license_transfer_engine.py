
# ============================================================
# Smart License Transfer Engine — PySpark / Databricks
# ============================================================
# Tables:
#   smart_license_inventory   → source + writeback (transfer_qty, substituted_qty)
#   license_transfer_log      → audit trail per batch run
#
# Rules:
#   1. Only same license_family is eligible for transfer
#   2. Higher tier CAN substitute lower tier (not reverse)
#   3. Only transfer required (deficit) quantity — never over-provision
#   4. Prefer exact-tier source first; within tier, largest surplus first
#   5. Process lowest-tier deficits first (preserve high-tier licenses)
# ============================================================

import uuid
import copy
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BooleanType, TimestampType
)
from delta.tables import DeltaTable


# ─────────────────────────────────────────────────────────────
# 0. Spark Session (already available in Databricks — kept for
#    local testing compatibility)
# ─────────────────────────────────────────────────────────────
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")   # tune for cluster size


# ─────────────────────────────────────────────────────────────
# 1. CONFIG — Table paths (update to your Unity Catalog paths)
# ─────────────────────────────────────────────────────────────
INVENTORY_TABLE  = "catalog.schema.smart_license_inventory"
AUDIT_LOG_TABLE  = "catalog.schema.license_transfer_log"


# ─────────────────────────────────────────────────────────────
# 2. LOAD INVENTORY
#    Reads the source table and derives availability columns
# ─────────────────────────────────────────────────────────────
def load_inventory(spark: SparkSession) -> list[dict]:
    """
    Load smart_license_inventory from Delta table.
    Returns a plain list of dicts for in-driver computation
    (license counts are always small — fits in driver memory).

    available_qty = purchased_qty + transfer_qty - used_qty - substituted_qty
    surplus       = max(0,  available_qty)
    deficit       = max(0, -available_qty)
    """
    df = spark.table(INVENTORY_TABLE).select(
        "account_id",
        "license_name",
        "license_family",
        "tier_level",
        "purchased_qty",
        "used_qty",
        "substituted_qty",
        "transfer_qty",
    )
    rows = [row.asDict() for row in df.collect()]

    for r in rows:
        r["available_qty"] = (
            r["purchased_qty"]
            + r["transfer_qty"]
            - r["used_qty"]
            - r["substituted_qty"]
        )
    return rows


# ─────────────────────────────────────────────────────────────
# 3. BUILD TIER METADATA from inventory rows
#    No separate config table needed — family + tier_level
#    already exist in smart_license_inventory
# ─────────────────────────────────────────────────────────────
def build_tier_metadata(rows: list[dict]):
    """
    Returns:
      family_tiers : {family -> [(tier_level, license_name), ...]} sorted ASC
      license_meta : {license_name -> {family, tier_level}}
    """
    family_tiers: dict[str, list] = defaultdict(list)
    license_meta: dict[str, dict] = {}
    seen = set()

    for r in rows:
        lic = r["license_name"]
        if lic not in seen:
            seen.add(lic)
            family_tiers[r["license_family"]].append(
                (r["tier_level"], lic)
            )
            license_meta[lic] = {
                "family":     r["license_family"],
                "tier_level": r["tier_level"],
            }

    for fam in family_tiers:
        family_tiers[fam].sort(key=lambda x: x[0])   # sort by tier ASC

    return dict(family_tiers), license_meta


# ─────────────────────────────────────────────────────────────
# 4. HELPER — live surplus / deficit from mutable state
# ─────────────────────────────────────────────────────────────
def get_surplus(state: dict, account_id: str, license_name: str) -> int:
    row = state.get((account_id, license_name))
    if not row:
        return 0
    avail = (
        row["purchased_qty"]
        + row["transfer_qty"]
        - row["used_qty"]
        - row["substituted_qty"]
    )
    return max(0, avail)


def get_deficit(state: dict, account_id: str, license_name: str) -> int:
    row = state.get((account_id, license_name))
    if not row:
        return 0
    avail = (
        row["purchased_qty"]
        + row["transfer_qty"]
        - row["used_qty"]
        - row["substituted_qty"]
    )
    return max(0, -avail)


# ─────────────────────────────────────────────────────────────
# 5. ELIGIBLE SOURCE FINDER
#    Returns eligible (account_id, license_name, live_surplus)
#    tuples sorted by: exact tier first → largest surplus first
# ─────────────────────────────────────────────────────────────
def get_eligible_sources(
    state: dict,
    family_tiers: dict,
    license_meta: dict,
    deficit_license: str,
    all_account_ids: list[str],
    exclude_account: str,
) -> list[tuple[str, str, int]]:
    """
    Higher tier CAN substitute lower tier (tier_level >= deficit_tier).
    Exact tier match is preferred over cross-tier to conserve higher licenses.
    Within same preference group → largest surplus first.
    """
    meta = license_meta.get(deficit_license)
    if not meta:
        return []

    family       = meta["family"]
    deficit_tier = meta["tier_level"]
    eligible     = []

    for acc_id in all_account_ids:
        if acc_id == exclude_account:
            continue
        for tier_level, lic_name in family_tiers.get(family, []):
            if tier_level >= deficit_tier:              # same or higher only
                surplus = get_surplus(state, acc_id, lic_name)
                if surplus > 0:
                    eligible.append((acc_id, lic_name, surplus))

    # Sort: exact tier (False=0) before cross-tier (True=1), then surplus DESC
    eligible.sort(key=lambda x: (
        license_meta[x[1]]["tier_level"] != deficit_tier,
        -x[2],
    ))
    return eligible


# ─────────────────────────────────────────────────────────────
# 6. CORE TRANSFER ENGINE (runs on driver)
# ─────────────────────────────────────────────────────────────
@dataclass
class TransferRecord:
    from_account:       str
    to_account:         str
    source_license:     str    # license physically moved from source account
    satisfies_license:  str    # deficit license it covers (may differ cross-tier)
    quantity:           int
    is_cross_tier:      bool


def compute_transfers(rows: list[dict]) -> tuple[list[TransferRecord], dict]:
    """
    Pure Python transfer engine.

    Args:
        rows: list of inventory dicts from load_inventory()

    Returns:
        transfer_log : list[TransferRecord]
        final_state  : {(account_id, license_name) -> updated row dict}
    """
    family_tiers, license_meta = build_tier_metadata(rows)

    # Mutable state keyed by (account_id, license_name)
    state: dict = {(r["account_id"], r["license_name"]): dict(r) for r in rows}

    all_account_ids = list({r["account_id"] for r in rows})
    transfer_log: list[TransferRecord] = []

    # ── Collect all deficits; process lowest tier first ──────
    deficit_work = []
    for (acc_id, lic_name), row in state.items():
        d = get_deficit(state, acc_id, lic_name)
        if d > 0:
            tier = license_meta.get(lic_name, {}).get("tier_level", 99)
            deficit_work.append((tier, lic_name, acc_id, d))

    deficit_work.sort(key=lambda x: x[0])   # lowest tier first

    # ── Greedy matching loop ──────────────────────────────────
    for _, deficit_lic, target_id, _ in deficit_work:

        remaining = get_deficit(state, target_id, deficit_lic)
        if remaining <= 0:
            continue                         # already resolved by earlier pass

        sources = get_eligible_sources(
            state, family_tiers, license_meta,
            deficit_lic, all_account_ids, target_id,
        )

        for src_id, src_lic, _ in sources:
            if remaining <= 0:
                break

            live_surplus = get_surplus(state, src_id, src_lic)
            if live_surplus <= 0:
                continue

            qty      = min(remaining, live_surplus)
            is_cross = (src_lic != deficit_lic)

            # ── Mutate state (source: transfer_qty decreases) ──
            state[(src_id, src_lic)]["transfer_qty"] -= qty

            # ── Mutate state (target: transfer_qty increases) ──
            if (target_id, deficit_lic) not in state:
                # Row absent → create a placeholder (INSERT later)
                state[(target_id, deficit_lic)] = {
                    "account_id":      target_id,
                    "license_name":    deficit_lic,
                    "license_family":  license_meta[deficit_lic]["family"],
                    "tier_level":      license_meta[deficit_lic]["tier_level"],
                    "purchased_qty":   0,
                    "used_qty":        0,
                    "substituted_qty": 0,
                    "transfer_qty":    0,
                    "available_qty":   0,
                }
            state[(target_id, deficit_lic)]["transfer_qty"]    += qty

            # Cross-tier substitution → increment substituted_qty on target
            if is_cross:
                state[(target_id, deficit_lic)]["substituted_qty"] += qty

            transfer_log.append(TransferRecord(
                from_account=src_id,
                to_account=target_id,
                source_license=src_lic,
                satisfies_license=deficit_lic,
                quantity=qty,
                is_cross_tier=is_cross,
            ))
            remaining -= qty

        if remaining > 0:
            print(
                f"[WARN] '{target_id}' still needs {remaining}x "
                f"[{deficit_lic}] — insufficient surplus in pool."
            )

    return transfer_log, state


# ─────────────────────────────────────────────────────────────
# 7. WRITEBACK — Delta MERGE into smart_license_inventory
# ─────────────────────────────────────────────────────────────
def writeback_inventory(
    spark: SparkSession,
    final_state: dict,
    original_rows: list[dict],
) -> None:
    """
    MERGE updated transfer_qty + substituted_qty back into the
    Delta source table. Only rows that actually changed are touched.
    """
    original_map = {
        (r["account_id"], r["license_name"]): r for r in original_rows
    }

    changed_rows = []
    for (acc_id, lic_name), row in final_state.items():
        orig = original_map.get((acc_id, lic_name), {})
        if (row["transfer_qty"]    != orig.get("transfer_qty", 0) or
                row["substituted_qty"] != orig.get("substituted_qty", 0)):
            changed_rows.append({
                "account_id":      acc_id,
                "license_name":    lic_name,
                "transfer_qty":    row["transfer_qty"],
                "substituted_qty": row["substituted_qty"],
                "purchased_qty":   row["purchased_qty"],
                "used_qty":        row["used_qty"],
                "license_family":  row["license_family"],
                "tier_level":      row["tier_level"],
            })

    if not changed_rows:
        print("[INFO] No inventory changes to write back.")
        return

    schema = StructType([
        StructField("account_id",      StringType(),  False),
        StructField("license_name",    StringType(),  False),
        StructField("license_family",  StringType(),  True),
        StructField("tier_level",      IntegerType(), True),
        StructField("purchased_qty",   IntegerType(), True),
        StructField("used_qty",        IntegerType(), True),
        StructField("transfer_qty",    IntegerType(), True),
        StructField("substituted_qty", IntegerType(), True),
    ])

    updates_df = spark.createDataFrame(changed_rows, schema=schema)

    delta_tbl = DeltaTable.forName(spark, INVENTORY_TABLE)
    (
        delta_tbl.alias("tgt")
        .merge(
            updates_df.alias("src"),
            "tgt.account_id = src.account_id AND tgt.license_name = src.license_name",
        )
        .whenMatchedUpdate(set={
            "transfer_qty":    "src.transfer_qty",
            "substituted_qty": "src.substituted_qty",
            "updated_at":      F.current_timestamp(),
        })
        .whenNotMatchedInsert(values={           # handles new rows from cross-tier
            "account_id":      "src.account_id",
            "license_name":    "src.license_name",
            "license_family":  "src.license_family",
            "tier_level":      "src.tier_level",
            "purchased_qty":   "src.purchased_qty",
            "used_qty":        "src.used_qty",
            "transfer_qty":    "src.transfer_qty",
            "substituted_qty": "src.substituted_qty",
            "updated_at":      F.current_timestamp(),
        })
        .execute()
    )
    print(f"[INFO] Inventory writeback: {len(changed_rows)} row(s) merged.")


# ─────────────────────────────────────────────────────────────
# 8. AUDIT LOG — Append transfer records to Delta log table
# ─────────────────────────────────────────────────────────────
def write_audit_log(
    spark: SparkSession,
    transfer_log: list[TransferRecord],
    batch_id: str,
) -> None:
    """Appends one row per transfer action to license_transfer_log."""
    if not transfer_log:
        print("[INFO] No transfers to log.")
        return

    log_schema = StructType([
        StructField("batch_id",                StringType(),   False),
        StructField("from_account_id",         StringType(),   False),
        StructField("to_account_id",           StringType(),   False),
        StructField("source_license_name",     StringType(),   False),
        StructField("satisfies_license_name",  StringType(),   False),
        StructField("quantity",                IntegerType(),  False),
        StructField("is_cross_tier",           BooleanType(),  False),
        StructField("transferred_at",          TimestampType(), False),
    ])

    now = datetime.utcnow()
    log_rows = [
        {
            "batch_id":               batch_id,
            "from_account_id":        t.from_account,
            "to_account_id":          t.to_account,
            "source_license_name":    t.source_license,
            "satisfies_license_name": t.satisfies_license,
            "quantity":               t.quantity,
            "is_cross_tier":          t.is_cross_tier,
            "transferred_at":         now,
        }
        for t in transfer_log
    ]

    log_df = spark.createDataFrame(log_rows, schema=log_schema)
    log_df.write.format("delta").mode("append").saveAsTable(AUDIT_LOG_TABLE)
    print(f"[INFO] Audit log: {len(log_rows)} transfer record(s) written (batch={batch_id}).")


# ─────────────────────────────────────────────────────────────
# 9. MAIN ORCHESTRATOR — single entry point
# ─────────────────────────────────────────────────────────────
def run_license_transfer(spark: SparkSession) -> None:
    """
    Full pipeline:
      1. Load inventory from Delta
      2. Compute optimal transfers (driver-side)
      3. MERGE transfer_qty + substituted_qty back to inventory table
      4. Append audit records to transfer log table
    """
    batch_id = str(uuid.uuid4())
    print(f"
{'='*60}")
    print(f"  License Transfer Run  |  batch_id={batch_id}")
    print(f"{'='*60}")

    # Step 1 — Load
    print("[STEP 1] Loading inventory...")
    inventory_rows = load_inventory(spark)
    print(f"         Loaded {len(inventory_rows)} inventory rows.")

    # Step 2 — Compute
    print("[STEP 2] Computing transfers...")
    transfer_log, final_state = compute_transfers(inventory_rows)
    print(f"         {len(transfer_log)} transfer(s) computed.")
    for t in transfer_log:
        cross = f" [cross-tier → satisfies {t.satisfies_license}]" if t.is_cross_tier else ""
        print(f"         • {t.quantity}x [{t.source_license}]{cross}")
        print(f"           {t.from_account} → {t.to_account}")

    # Step 3 — Writeback inventory (Delta MERGE)
    print("[STEP 3] Writing back to inventory table...")
    writeback_inventory(spark, final_state, inventory_rows)

    # Step 4 — Audit log (Delta append)
    print("[STEP 4] Writing audit log...")
    write_audit_log(spark, transfer_log, batch_id)

    print(f"
[DONE] License transfer batch {batch_id} complete.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_license_transfer(spark)
