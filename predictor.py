"""
predictor.py
Turns the two trained models into the three numbers the user wants:
likes, comments and engagement rate.
"""
import numpy as np
import pandas as pd

from config import FEATURES


def predict_engagement(models, rows):
    """rows: a DataFrame with the FEATURES columns (one row per post)."""
    inputs = rows[FEATURES]
    likes_rate = np.maximum(models["likes_rate"].predict(inputs), 0)        # likes per 100 followers
    comments_rate = np.maximum(models["comments_rate"].predict(inputs), 0)  # comments per 100 followers
    followers = rows["followers"].to_numpy(dtype=float)
    return pd.DataFrame({
        "likes": likes_rate * followers / 100,
        "comments": comments_rate * followers / 100,
        "engagement_rate": likes_rate + comments_rate,   # in percent
    }, index=rows.index)
