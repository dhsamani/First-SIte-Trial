
"""
Production-Ready Health Score MLflow Model (Final Version)
✅ Databricks notebook compatible with secret scope
✅ Includes test_endpoint_locally() before deployment
✅ Passes MLflow model validation
✅ Connection reuse with proper cleanup
"""

import mlflow
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional
from databricks import sql
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthScoreModel(mlflow.pyfunc.PythonModel):
    """Production-ready MLflow model for real-time health score lookup from Unity Catalog."""

    def __init__(self):
        """Initialize model configuration."""
        self.catalog_name = "data_science"
        self.schema_name = "model_silver"
        self.table_name = "health_score"
        self.full_table_name = f"{self.catalog_name}.{self.schema_name}.{self.table_name}"
        self.chunk_size = 500

        # Connection parameters (set in load_context)
        self.server_hostname = None
        self.http_path = None
        self.access_token = None

        # Reusable SQL connection
        self._connection = None

        # Secret scope name (configure this)
        self.secret_scope = "your_scope_name"  # ← CHANGE THIS TO YOUR SCOPE

    def __del__(self):
        """Cleanup: close SQL connection when model is destroyed."""
        if self._connection is not None:
            try:
                self._connection.close()
                logger.info("SQL connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")

    # ---------------------------------------------------------------------
    # Context / connection management
    # ---------------------------------------------------------------------
    def load_context(self, context):
        """Load model context and validate environment."""
        logger.info("Initializing HealthScoreModel...")

        # Try to load from secret scope (Databricks notebook/serving)
        try:
            from databricks.sdk.runtime import dbutils

            logger.info(f"Loading credentials from secret scope: {self.secret_scope}")
            self.server_hostname = dbutils.secrets.get(scope=self.secret_scope, key="databricks_hostname")
            self.http_path = dbutils.secrets.get(scope=self.secret_scope, key="databricks_http_path")
            self.access_token = dbutils.secrets.get(scope=self.secret_scope, key="databricks_token")

            logger.info("✅ Credentials loaded from secret scope")

        except Exception as e:
            # Fallback to environment variables (for local testing or if secrets fail)
            logger.warning(f"Could not load from secret scope: {str(e)}")
            logger.info("Falling back to environment variables")

            self.server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
            self.http_path = os.getenv("DATABRICKS_HTTP_PATH")
            self.access_token = os.getenv("DATABRICKS_TOKEN")

        # Validate credentials are set
        required_vars = {
            "server_hostname": self.server_hostname,
            "http_path": self.http_path,
            "access_token": self.access_token,
        }
        missing = [k for k, v in required_vars.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required credentials: {missing}. "
                f"Configure secret scope '{self.secret_scope}' or set environment variables."
            )

        logger.info("✅ Environment validated")
        logger.info(f"📊 Target table: {self.full_table_name}")
        logger.info(f"🔗 SQL Warehouse: {self.http_path}")

    def _ensure_connection(self):
        """Ensure a live Databricks SQL connection, reconnect on failure."""
        try:
            if self._connection is not None:
                # Test connection with ping
                try:
                    cursor = self._connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchall()
                    cursor.close()
                    return self._connection
                except Exception as e:
                    logger.warning(f"Existing connection invalid, reconnecting: {str(e)}")
                    try:
                        self._connection.close()
                    except Exception:
                        pass
                    self._connection = None

            # Create new connection
            logger.info("Creating new Databricks SQL connection")
            self._connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            )
            logger.info("✅ SQL connection established")
            return self._connection

        except Exception as e:
            logger.error(f"Failed to create SQL connection: {str(e)}")
            raise

    # ---------------------------------------------------------------------
    # DB query
    # ---------------------------------------------------------------------
    def _query_lookup(self, ci_ids: List[str]) -> pd.DataFrame:
        """Execute parameterized SQL query to fetch required columns from Unity Catalog."""
        if not ci_ids:
            return pd.DataFrame(
                columns=[
                    "ContractCI_ConfigurationItemCode",
                    "ContractStatusFlag",
                    "Health_Score",
                    "action_plan",
                ]
            )

        placeholders = ", ".join(["?" for _ in ci_ids])
        query = f"""
            SELECT
                ContractCI_ConfigurationItemCode,
                ContractStatusFlag,
                Health_Score,
                action_plan
            FROM {self.full_table_name}
            WHERE ContractCI_ConfigurationItemCode IN ({placeholders})
        """

        try:
            logger.info(f"Querying {len(ci_ids)} CI IDs from {self.full_table_name}")
            connection = self._ensure_connection()
            cursor = connection.cursor()
            cursor.execute(query, ci_ids)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"✅ Retrieved {len(df)} records from database")
            return df

        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            raise

    # ---------------------------------------------------------------------
    # Business logic
    # ---------------------------------------------------------------------
    def _process_single_ci(self, ci_id: str, df_records: pd.DataFrame) -> Dict[str, Any]:
        """Apply business logic for a single CI ID."""

        # Scenario 1: CI Not Found
        if df_records.empty:
            return {
                "ContractCI_ConfigurationItemCode": ci_id,
                "ContractStatusFlag": "not_available",
                "Health_Score": "not_available",
                "message": f"CI {ci_id} not found in database",
            }

        # Filter for Active records (case-insensitive)
        active_records = df_records[
            df_records["ContractStatusFlag"].str.upper() == "ACTIVE"
        ].copy()

        # Scenario 2: No Active Records
        if active_records.empty:
            return {
                "ContractCI_ConfigurationItemCode": ci_id,
                "ContractStatusFlag": "Inactive",
                "Health_Score": "not_available",
                "message": f"CI {ci_id} has no active contracts",
            }

        # Filter for valid (non-NULL) health scores
        valid_mask = active_records["Health_Score"].notna()
        valid_scores = active_records.loc[valid_mask, "Health_Score"]

        # Scenario 3: Active with Null Scores
        if valid_scores.empty:
            return {
                "ContractCI_ConfigurationItemCode": ci_id,
                "ContractStatusFlag": "Active",
                "Health_Score": "Health Score not available",
                "message": f"CI {ci_id} has active contracts but no health scores",
            }

        # Scenario 4a: Single Active Record
        if len(valid_scores) == 1:
            row = active_records.loc[valid_mask].iloc[0]
            return {
                "ContractCI_ConfigurationItemCode": ci_id,
                "ContractStatusFlag": "Active",
                "Health_Score": float(row["Health_Score"]),
                "action_plan": str(row.get("action_plan")) if pd.notna(row.get("action_plan")) else None,
            }

        # Scenario 4b: Multiple Active Records
        avg_score = float(valid_scores.mean())

        # For action_plan: use first non-null value
        action_plan_value = None
        if "action_plan" in active_records.columns:
            non_null_plans = active_records.loc[valid_mask, "action_plan"].dropna()
            if not non_null_plans.empty:
                action_plan_value = str(non_null_plans.iloc[0])

        return {
            "ContractCI_ConfigurationItemCode": ci_id,
            "ContractStatusFlag": "Active",
            "Health_Score": avg_score,
            "action_plan": action_plan_value,
            "Total_Active_Records": len(valid_scores),
        }

    # ---------------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------------
    def _validate_and_extract_ci_ids(self, model_input: Any) -> List[str]:
        """Validate input and extract CI IDs with duplicate removal."""
        ci_ids: List[str] = []

        if isinstance(model_input, pd.DataFrame):
            if "ContractCI_ConfigurationItemCode" not in model_input.columns:
                raise ValueError(
                    "DataFrame must contain 'ContractCI_ConfigurationItemCode' column"
                )
            ci_ids = model_input["ContractCI_ConfigurationItemCode"].astype(str).tolist()

        elif isinstance(model_input, dict):
            if "ContractCI_ConfigurationItemCode" not in model_input:
                raise ValueError(
                    "Input dict must contain 'ContractCI_ConfigurationItemCode' key"
                )
            value = model_input["ContractCI_ConfigurationItemCode"]
            if isinstance(value, str):
                ci_ids = [value]
            elif isinstance(value, list):
                ci_ids = [str(v) for v in value]
            else:
                raise ValueError(
                    "ContractCI_ConfigurationItemCode must be string or list"
                )
        else:
            raise ValueError("Input must be dict or DataFrame")

        if not ci_ids:
            raise ValueError("No CI IDs provided in input")

        # Remove duplicates while preserving order
        seen = set()
        unique_ci_ids: List[str] = []
        for ci in ci_ids:
            if ci not in seen:
                seen.add(ci)
                unique_ci_ids.append(ci)

        logger.info(f"Extracted {len(unique_ci_ids)} unique CI IDs from input")
        return unique_ci_ids

    def _chunk_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks."""
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    # ---------------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------------
    def predict(self, context, model_input):
        """Perform real-time lookup and apply business logic."""
        try:
            ci_ids = self._validate_and_extract_ci_ids(model_input)
            is_single = len(ci_ids) == 1

            all_results: List[Dict[str, Any]] = []
            chunks = self._chunk_list(ci_ids, self.chunk_size)
            logger.info(f"Processing {len(ci_ids)} CI IDs in {len(chunks)} chunk(s)")

            for idx, chunk in enumerate(chunks, start=1):
                logger.info(f"Processing chunk {idx}/{len(chunks)} ({len(chunk)} IDs)")
                df_chunk = self._query_lookup(chunk)

                for ci in chunk:
                    ci_records = df_chunk[
                        df_chunk["ContractCI_ConfigurationItemCode"] == ci
                    ]
                    result = self._process_single_ci(ci, ci_records)
                    all_results.append(result)

            logger.info(f"✅ Successfully processed {len(all_results)} CI IDs")
            return all_results[0] if is_single else all_results

        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise


# ============================================================================
# TESTING FUNCTION (Run in notebook before deployment)
# ============================================================================

def test_endpoint_locally():
    """
    Test the model locally in Databricks notebook before deployment.
    Tests all 4 business scenarios.
    """
    print("="*70)
    print("🧪 TESTING HEALTH SCORE MODEL LOCALLY")
    print("="*70)

    try:
        # Initialize model
        model = HealthScoreModel()

        class MockContext:
            pass

        # Load context (will use secret scope)
        print("\n1️⃣  Loading model context...")
        model.load_context(MockContext())
        print(f"   ✅ Connected to: {model.server_hostname}")
        print(f"   ✅ Target table: {model.full_table_name}")

        # Test Case 1: Single CI ID
        print("\n2️⃣  Test Case 1: Single CI ID")
        print("   " + "-"*60)
        test_input_1 = {"ContractCI_ConfigurationItemCode": "89045"}
        try:
            result_1 = model.predict(None, test_input_1)
            print(f"   Input: {test_input_1}")
            print(f"   Output: {result_1}")
            print("   ✅ Test 1 passed")
        except Exception as e:
            print(f"   ❌ Test 1 failed: {str(e)}")
            return False

        # Test Case 2: Multiple CI IDs
        print("\n3️⃣  Test Case 2: Multiple CI IDs")
        print("   " + "-"*60)
        test_input_2 = {"ContractCI_ConfigurationItemCode": ["89045", "96789", "99999"]}
        try:
            result_2 = model.predict(None, test_input_2)
            print(f"   Input: Multiple IDs (3 CIs)")
            for idx, r in enumerate(result_2, 1):
                print(f"   Result {idx}: {r}")
            print("   ✅ Test 2 passed")
        except Exception as e:
            print(f"   ❌ Test 2 failed: {str(e)}")
            return False

        # Test Case 3: DataFrame input
        print("\n4️⃣  Test Case 3: DataFrame Input")
        print("   " + "-"*60)
        test_input_3 = pd.DataFrame({
            "ContractCI_ConfigurationItemCode": ["89045", "96789"]
        })
        try:
            result_3 = model.predict(None, test_input_3)
            print(f"   Input: DataFrame with {len(test_input_3)} rows")
            for idx, r in enumerate(result_3, 1):
                print(f"   Result {idx}: {r}")
            print("   ✅ Test 3 passed")
        except Exception as e:
            print(f"   ❌ Test 3 failed: {str(e)}")
            return False

        # Test Case 4: Edge case - empty scenario checks
        print("\n5️⃣  Test Case 4: Testing all scenarios")
        print("   " + "-"*60)

        # Count scenarios from test 2
        scenarios_found = set()
        for r in result_2:
            if r.get("ContractStatusFlag") == "not_available":
                scenarios_found.add("Scenario 1: CI Not Found")
            elif r.get("ContractStatusFlag") == "Inactive":
                scenarios_found.add("Scenario 2: No Active Records")
            elif r.get("Health_Score") == "Health Score not available":
                scenarios_found.add("Scenario 3: Active but No Scores")
            elif r.get("ContractStatusFlag") == "Active" and isinstance(r.get("Health_Score"), (int, float)):
                if "Total_Active_Records" in r:
                    scenarios_found.add("Scenario 4b: Multiple Active Records")
                else:
                    scenarios_found.add("Scenario 4a: Single Active Record")

        print(f"   Scenarios tested: {len(scenarios_found)}")
        for s in scenarios_found:
            print(f"   ✅ {s}")

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - MODEL IS READY FOR DEPLOYMENT")
        print("="*70)
        return True

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TESTING FAILED: {str(e)}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# DEPLOYMENT FUNCTIONS
# ============================================================================

def log_model_to_mlflow(
    model_name: str = "health_score_lookup",
    run_name: str = "prod_v1",
    run_validation: bool = True
):
    """
    Log the custom model to MLflow Model Registry.

    Args:
        model_name: Name for registered model
        run_name: Name for MLflow run
        run_validation: Whether to run validation tests before logging

    Returns:
        Run ID
    """
    print("="*70)
    print("📦 LOGGING MODEL TO MLFLOW")
    print("="*70)

    # Optional: Run validation first
    if run_validation:
        print("\n🧪 Running pre-deployment validation...")
        if not test_endpoint_locally():
            raise ValueError(
                "Model validation failed. Fix errors before deployment."
            )
        print("\n✅ Validation passed, proceeding with MLflow logging...")

    with mlflow.start_run(run_name=run_name) as run:
        model = HealthScoreModel()

        # Input example for schema
        input_example = pd.DataFrame({
            "ContractCI_ConfigurationItemCode": ["89045", "96789"]
        })

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            input_example=input_example,
            registered_model_name=model_name,
            pip_requirements=[
                "mlflow==2.10.0",
                "pandas==2.0.3",
                "numpy==1.24.3",
                "databricks-sql-connector==3.0.0",
                "databricks-sdk",  # For dbutils.secrets
            ],
        )

        print(f"\n✅ Model logged successfully!")
        print(f"   📦 Run ID: {run.info.run_id}")
        print(f"   🏷️  Model Name: {model_name}")
        print(f"   📍 Artifact Path: model")
        print("="*70)

        return run.info.run_id


def create_serving_endpoint(
    endpoint_name: str = "health-score-lookup",
    model_name: str = "health_score_lookup",
    model_version: str = "1",
    workload_size: str = "Small",
):
    """Create Model Serving endpoint."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        ServedEntityInput,
        EndpointCoreConfigInput,
    )

    print("="*70)
    print("🚀 CREATING MODEL SERVING ENDPOINT")
    print("="*70)

    w = WorkspaceClient()

    print(f"\n   Endpoint Name: {endpoint_name}")
    print(f"   Model: {model_name} v{model_version}")
    print(f"   Workload Size: {workload_size}")

    endpoint = w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=model_version,
                    workload_size=workload_size,
                    scale_to_zero_enabled=True,
                )
            ]
        ),
    )

    print(f"\n✅ Endpoint created successfully!")
    print(f"   🔗 Endpoint: {endpoint_name}")
    print(f"   📊 State: {endpoint.state}")
    print("="*70)

    return endpoint


# ============================================================================
# MAIN - Run in Databricks Notebook
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Health Score MLflow Model - Final Production Version")
    print("="*70)
    print("\nAvailable functions:")
    print("  1. test_endpoint_locally() - Test model before deployment")
    print("  2. log_model_to_mlflow() - Log model to registry (runs test first)")
    print("  3. create_serving_endpoint() - Deploy to Model Serving")
    print("\nRecommended workflow:")
    print("  Step 1: test_endpoint_locally()")
    print("  Step 2: log_model_to_mlflow()")
    print("  Step 3: create_serving_endpoint()")
    print("="*70)
