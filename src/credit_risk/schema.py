"""The data contract, in one place.

Every other module imports its column names and valid codes from here. The
alternative -- string literals sprinkled across ingestion, training and
serving -- is how a rename in one place silently stops matching the other two.

The facts below were measured on the raw file, not copied from the UCI page:
30,000 rows x 25 columns, zero nulls, 22.12% positive class. Where the data
disagrees with the published dictionary, the disagreement is recorded here
rather than quietly patched.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------- source

DATASET_URL: Final = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
DATASET_MEMBER: Final = "default of credit card clients.xls"

# The header sits on the SECOND row of the sheet; the first row is a banner.
EXCEL_HEADER_ROW: Final = 1

RAW_N_ROWS: Final = 30_000
RAW_N_COLS: Final = 25

# --------------------------------------------------------------- columns

ID_COL: Final = "ID"
TARGET_RAW: Final = "default payment next month"
TARGET: Final = "default_next_month"

# UCI ships the first repayment-status column as PAY_0 while the rest are
# PAY_2..PAY_6. It is an off-by-one in their export, not a different variable.
PAY_0_RAW: Final = "PAY_0"
PAY_1: Final = "PAY_1"

LIMIT_BAL: Final = "LIMIT_BAL"
SEX: Final = "SEX"
EDUCATION: Final = "EDUCATION"
MARRIAGE: Final = "MARRIAGE"
AGE: Final = "AGE"
AGE_GROUP: Final = "AGE_GROUP"

PAY_COLS: Final[tuple[str, ...]] = ("PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
BILL_COLS: Final[tuple[str, ...]] = tuple(f"BILL_AMT{i}" for i in range(1, 7))
PAY_AMT_COLS: Final[tuple[str, ...]] = tuple(f"PAY_AMT{i}" for i in range(1, 7))

DEMOGRAPHIC_COLS: Final[tuple[str, ...]] = (SEX, EDUCATION, MARRIAGE, AGE)

RAW_COLUMNS: Final[tuple[str, ...]] = (
    ID_COL,
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    PAY_0_RAW,
    *PAY_COLS[1:],
    *BILL_COLS,
    *PAY_AMT_COLS,
    TARGET_RAW,
)

CLEAN_COLUMNS: Final[tuple[str, ...]] = (
    ID_COL,
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    *PAY_COLS,
    *BILL_COLS,
    *PAY_AMT_COLS,
    TARGET,
)

RENAME_MAP: Final[dict[str, str]] = {PAY_0_RAW: PAY_1, TARGET_RAW: TARGET}

# ------------------------------------------------------- categorical codes

# Published dictionary. The raw file contains codes outside these sets --
# see UNDOCUMENTED_* below. Cleaning folds them into the documented "other"
# bucket, and a data-quality test asserts nothing outside these sets survives.
SEX_CODES: Final[dict[int, str]] = {1: "male", 2: "female"}
EDUCATION_CODES: Final[dict[int, str]] = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "other",
}
MARRIAGE_CODES: Final[dict[int, str]] = {1: "married", 2: "single", 3: "other"}

# Measured on the raw file: EDUCATION has 0 (14 rows), 5 (280), 6 (51);
# MARRIAGE has 0 (54). None appear in the dictionary.
UNDOCUMENTED_EDUCATION: Final[tuple[int, ...]] = (0, 5, 6)
UNDOCUMENTED_MARRIAGE: Final[tuple[int, ...]] = (0,)
EDUCATION_OTHER: Final = 4
MARRIAGE_OTHER: Final = 3

# PAY_* encodes months of delay. -2 = no consumption, -1 = paid in full,
# 0 = revolving credit, 1..8 = months overdue. 9 does not occur.
PAY_MIN: Final = -2
PAY_MAX: Final = 8

AGE_MIN: Final = 21
AGE_MAX: Final = 79
AGE_GROUP_CUTOFF: Final = 35  # <=35 -> "young", >35 -> "older"

# ------------------------------------------------------------- fairness

PRIMARY_PROTECTED: Final = SEX
PROTECTED_ATTRIBUTES: Final[tuple[str, ...]] = (SEX, AGE_GROUP, EDUCATION, MARRIAGE)

# Gates. A candidate model that breaches either is not registered.
MAX_DEMOGRAPHIC_PARITY_DIFF: Final = 0.05
MAX_EQUALIZED_ODDS_DIFF: Final = 0.08

# --------------------------------------------------------------- batches

# The file is static. Splitting it into six ID-ordered batches lets the
# pipeline ingest "an arrival" rather than copy a file. This is simulation
# and is declared as such in DATASHEET.md.
N_BATCHES: Final = 6
BATCH_SIZE: Final = RAW_N_ROWS // N_BATCHES  # 5,000
TRAIN_BATCHES: Final[tuple[int, ...]] = (1, 2, 3, 4)
TEST_BATCH: Final = 5
SERVING_BATCH: Final = 6  # pool the traffic generator draws from

RANDOM_SEED: Final = 42

# ---------------------------------------------------------- expectations

# Base rates measured on the raw file. Used as monitoring baselines and as
# fixtures in data-quality tests, so a silent change of source is caught.
BASE_POSITIVE_RATE: Final = 0.2212
BASE_RATE_BY_SEX: Final[dict[int, float]] = {1: 0.2417, 2: 0.2078}

MAX_BAD_ROW_FRACTION: Final = 0.05  # above this, ingestion fails the DAG
