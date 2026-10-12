"""
generate_dataset.py
Creates a small SAMPLE dataset of Instagram posts and saves it to
data/instagram_posts.csv.

IMPORTANT: this data is synthetic (computer-generated). It follows common
sense patterns (reels do well, evenings are busy, small accounts have higher
engagement rates) so that the model has something realistic to learn, but it is
NOT real Instagram data. Say this clearly in your report. To use real data,
replace the CSV with your own file that has the same columns.

Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd

from config import DATA_PATH, DAYS, POST_TYPES, RANDOM_STATE

N_POSTS = 800

# How much each factor raises or lowers engagement (1.0 = no change)
TYPE_FACTOR = {"image": 1.00, "carousel": 1.35, "video": 1.10, "reel": 1.50}
DAY_FACTOR = {"Monday": 0.95, "Tuesday": 1.00, "Wednesday": 1.05,
              "Thursday": 1.00, "Friday": 0.95, "Saturday": 1.05,
              "Sunday": 1.10}


def make_dataset(n_posts=N_POSTS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    # --- 1. Create the input columns ---------------------------------------
    followers = np.exp(rng.normal(8.5, 1.1, n_posts))          # many small, few big accounts
    followers = np.clip(followers, 200, 500_000).round().astype(int)

    post_type = rng.choice(POST_TYPES, size=n_posts, p=[0.35, 0.20, 0.15, 0.30])
    caption_length = np.clip(rng.normal(180, 110, n_posts), 0, 700).round().astype(int)
    hashtags = np.clip(rng.poisson(8, n_posts), 0, 30)

    # People post mostly around lunchtime and in the evening
    hour_weights = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6,
                             7, 6, 5, 5, 6, 8, 9, 10, 9, 7, 4, 2], dtype=float)
    posting_hour = rng.choice(24, size=n_posts, p=hour_weights / hour_weights.sum())

    day_of_week = rng.choice(DAYS, size=n_posts)
    has_cta = rng.binomial(1, 0.4, n_posts)

    # --- 2. Decide the engagement rate (percent) ---------------------------
    # Smaller accounts usually get a higher rate than huge accounts
    base_rate = 4.0 * (followers / 1000) ** -0.15

    type_f = np.array([TYPE_FACTOR[t] for t in post_type])
    day_f = np.array([DAY_FACTOR[d] for d in day_of_week])
    hashtag_f = 0.85 + 0.30 * np.exp(-((hashtags - 10) / 7) ** 2)        # best near 10
    caption_f = 0.90 + 0.20 * np.exp(-((caption_length - 150) / 150) ** 2)  # medium length best
    hour_f = 0.80 + 0.40 * (np.exp(-((posting_hour - 19) / 3) ** 2)
                            + 0.6 * np.exp(-((posting_hour - 12) / 2.5) ** 2))
    cta_f = np.where(has_cta == 1, 1.12, 1.00)
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=n_posts)            # real life is messy

    rate = base_rate * type_f * day_f * hashtag_f * caption_f * hour_f * cta_f * noise

    # --- 3. Turn the rate into likes and comments --------------------------
    total = followers * rate / 100
    comment_share = np.clip(rng.normal(0.04, 0.015, n_posts) + 0.015 * has_cta, 0.01, 0.12)
    comments = np.round(total * comment_share).astype(int)
    likes = np.maximum(np.round(total).astype(int) - comments, 1)

    df = pd.DataFrame({
        "post_id": np.arange(1, n_posts + 1),
        "followers": followers,
        "post_type": post_type,
        "caption_length": caption_length,
        "hashtags": hashtags,
        "posting_hour": posting_hour,
        "day_of_week": day_of_week,
        "has_cta": has_cta,
        "likes": likes,
        "comments": comments,
    })
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / df["followers"] * 100).round(2)
    return df


def main():
    df = make_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} sample posts to {DATA_PATH}")
    print(f"Average engagement rate: {df['engagement_rate'].mean():.2f}%")


if __name__ == "__main__":
    main()
