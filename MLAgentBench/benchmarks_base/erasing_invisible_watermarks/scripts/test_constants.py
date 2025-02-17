"""Constants for test evaluation"""

# Paths
TEST_DATA_STEGASTAMP = "scripts/test_data/test_images_stegastamp.pkl"
TEST_DATA_TREERING = "scripts/test_data/test_images_treering.pkl"

# Dataset URL
DATASET_URL = "https://drive.google.com/file/d/1Q0Ahhg_wLk3OK15fs_cQZ7_GOkye5acS/view?usp=sharing"

# Dataset split ratio
TEST_RATIO = 0.2
RANDOM_SEED = 42

# Evaluation thresholds
MIN_WATERMARK_REMOVAL_SCORE = 0.5
MAX_QUALITY_DEGRADATION = 0.3
MIN_OVERALL_SCORE = 0.6

# Method-specific thresholds
STEGASTAMP_MIN_REMOVAL = 0.45
TREERING_MIN_REMOVAL = 0.55 