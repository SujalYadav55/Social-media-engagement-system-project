from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "instagram_posts.csv"
MODEL_PATH = BASE_DIR / "models" / "engagement_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

RANDOM_STATE = 42  # makes results repeatable every time you run the project

POST_TYPES = ["image", "carousel", "video", "reel"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

# Columns the model uses as INPUTS
NUMERIC_FEATURES = ["followers", "caption_length", "hashtags",
                    "posting_hour", "has_cta"]
CATEGORICAL_FEATURES = ["post_type", "day_of_week"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The model learns to PREDICT two percentages (likes and comments per 100 followers).
# Likes, comments and the engagement rate are then worked out from them.
TARGETS = ["likes_rate", "comments_rate"]

# Range of follower counts seen in the sample dataset (used for a warning)
TRAINED_FOLLOWER_RANGE = (200, 500_000)
TRAINED_MAX_CAPTION = 700

# Text shown on the About page
COLUMN_INFO = [
    ("followers", "Number of followers the account had when the post went live"),
    ("post_type", "image, carousel, video or reel"),
    ("caption_length", "Characters in the caption, not counting hashtags"),
    ("hashtags", "Number of hashtags used (0 to 30)"),
    ("posting_hour", "Hour of the day the post was published (0 to 23)"),
    ("day_of_week", "Day of the week the post was published"),
    ("has_cta", "1 if the caption asked viewers to do something (comment, share, tap the link), else 0"),
    ("likes", "Likes the post received (target to predict)"),
    ("comments", "Comments the post received (target to predict)"),
    ("engagement_rate", "(likes + comments) / followers x 100, in percent"),
]
