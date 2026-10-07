"""Configuration module for the Kestrel Home Returns Risk project.

Defines all file paths, schemas, operational constants, and economic parameters.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Base Directory Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# -----------------------------------------------------------------------------
# Raw File Paths (Read-Only)
# -----------------------------------------------------------------------------
TRAIN_RAW_PATH = RAW_DATA_DIR / "train.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test_unlabelled.csv"
CUSTOMERS_RAW_PATH = RAW_DATA_DIR / "customers.csv"
PRODUCTS_RAW_PATH = RAW_DATA_DIR / "products.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

# -----------------------------------------------------------------------------
# Processed Output Paths
# -----------------------------------------------------------------------------
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_features.parquet"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_features.parquet"
FEATURE_PIPELINE_PATH = PROCESSED_DATA_DIR / "feature_pipeline.joblib"

# -----------------------------------------------------------------------------
# Identifier & Target Definitions
# -----------------------------------------------------------------------------
ID_COLUMN = "order_id"
CUSTOMER_ID_COLUMN = "customer_id"
PRODUCT_ID_COLUMN = "sku"
TARGET_COLUMN = "returned"

# Target leakage columns: strictly prohibited from feature engineering
LEAKAGE_COLUMNS = [
    "pickup_scheduled_at",      # Reverse logistics booking timestamp (post-outcome)
    "last_service_event_type",   # Service lifecycle state (post-outcome: REVERSE_PICKUP, etc.)
]

# Metadata columns not representing predictive features
METADATA_COLUMNS = [
    "source",                   # 'crm' vs 'partner_feed' ingestion origin
]

# Timestamp columns
ORDER_DATE_COL = "order_placed_at"
SIGNUP_DATE_COL = "signup_date"
LAUNCH_DATE_COL = "launch_date"

# -----------------------------------------------------------------------------
# Domain & Data Integrity Constants
# -----------------------------------------------------------------------------
# October 2025 CRM migration bug (paise recorded instead of rupees)
OCT_2025_START = "2025-10-01"
OCT_2025_END = "2025-10-31 23:59:59"
PAISE_TO_RUPEE_FACTOR = 100.0

# Products requiring professional installation (Ops Policy v4.1)
INSTALLATION_FAMILIES = [
    "Water Purifier",
    "Robot Vacuum",
    "Ceiling Fan",
]

# Categorical domain vocabularies
KNOWN_SALES_CHANNELS = ["app", "marketplace", "web", "partner_outlet"]
KNOWN_PAYMENT_MODES = ["cod", "emi", "prepaid_card", "prepaid_upi"]
KNOWN_PRODUCT_FAMILIES = [
    "Air Fryer",
    "Ceiling Fan",
    "Induction Cooktop",
    "Mixer Grinder",
    "Robot Vacuum",
    "Room Heater",
    "Water Purifier",
]
KNOWN_PRODUCT_TIERS = ["Lite", "Pro", "Max"]
KNOWN_CUSTOMER_STATES = ["DL", "KA", "MH", "MP", "RJ", "TN", "TS", "UP"]

# -----------------------------------------------------------------------------
# Unit Economics & Operational Decision Parameters (Ops Policy v4.1)
# -----------------------------------------------------------------------------
RETURN_COST_INR = 1150.0            # Average cost of return logistics, QC, restocking
CONFIRMATION_CALL_COST_INR = 45.0   # Outbound contact centre verification call cost
INTERVENTION_PREVENTION_RATE = 0.35 # Historical proportion of returns prevented by call
OPTIMAL_CALL_THRESHOLD = 0.1118     # Break-even threshold: 45 / (0.35 * 1150)
MAX_DISPATCH_HOLD_HOURS = 24.0      # Maximum permissible hold before cancellation surge
HOLD_CANCELLATION_PENALTY_RATE = 0.12 # 12% customer cancellation if held > 24 hours

