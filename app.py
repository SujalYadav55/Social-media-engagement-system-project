"""
app.py
The Flask web application.

Pages:
  /          the form where you describe a post
  /predict   shows the predicted likes, comments and engagement rate
  /about     explains the dataset, the models and how accurate they are

Run:  python app.py      then open http://127.0.0.1:5000
"""
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

import train_model
from config import (COLUMN_INFO, DAYS, FEATURES, METRICS_PATH, MODEL_PATH,
                    POST_TYPES, TRAINED_FOLLOWER_RANGE, TRAINED_MAX_CAPTION)
from predictor import predict_engagement

app = Flask(__name__)

DEFAULT_VALUES = {
    "followers": "5000",
    "post_type": "reel",
    "caption": "",
    "day_of_week": "Wednesday",
    "posting_hour": "18",
    "has_cta": False,
}

_models = None


# ---------------------------------------------------------------- model loading
def get_models():
    """Load the trained models once. Train them first if they are missing."""
    global _models
    if _models is None:
        if not MODEL_PATH.exists():
            print("No trained model found - training one now (about 10 seconds)...")
            train_model.main()
        try:
            _models = joblib.load(MODEL_PATH)
        except Exception:
            print("Saved model could not be loaded (different scikit-learn version?). Retraining...")
            train_model.main()
            _models = joblib.load(MODEL_PATH)
    return _models


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as file:
            return json.load(file)
    return None


# ---------------------------------------------------------------- helpers
def hour_label(hour):
    return f"{hour:02d}:00"


def rate_category(rate):
    """Plain-language label. These bands are a rough rule of thumb, not an official standard."""
    if rate < 1:
        return "Low"
    if rate < 3:
        return "Average"
    if rate < 6:
        return "High"
    return "Very high"


def parse_form(form):
    """Check what the user typed. Returns (values_to_show_again, features, errors, notes)."""
    values = {
        "followers": form.get("followers", "").strip(),
        "post_type": form.get("post_type", ""),
        "caption": form.get("caption", ""),
        "day_of_week": form.get("day_of_week", ""),
        "posting_hour": form.get("posting_hour", ""),
        "has_cta": form.get("has_cta") == "on",
    }
    errors, notes = [], []

    # Followers
    followers = None
    try:
        followers = int(values["followers"].replace(",", ""))
        if not 1 <= followers <= 50_000_000:
            errors.append("Followers must be a number between 1 and 50,000,000.")
        elif not TRAINED_FOLLOWER_RANGE[0] <= followers <= TRAINED_FOLLOWER_RANGE[1]:
            notes.append(
                f"The sample data only covers accounts with {TRAINED_FOLLOWER_RANGE[0]:,} to "
                f"{TRAINED_FOLLOWER_RANGE[1]:,} followers, so treat this prediction as a rough guess.")
    except ValueError:
        errors.append("Followers must be a whole number, for example 5000.")

    # Post type and day
    if values["post_type"] not in POST_TYPES:
        errors.append("Choose a post type.")
    if values["day_of_week"] not in DAYS:
        errors.append("Choose a day of the week.")

    # Hour
    posting_hour = None
    try:
        posting_hour = int(values["posting_hour"])
        if not 0 <= posting_hour <= 23:
            raise ValueError
    except ValueError:
        errors.append("Choose an hour between 00:00 and 23:00.")

    # Caption: split it into plain text and hashtags
    words = values["caption"].split()
    hashtags = [w for w in words if w.startswith("#")]
    text = " ".join(w for w in words if not w.startswith("#"))
    if len(values["caption"]) > 2200:
        errors.append("Captions can be at most 2,200 characters.")
    if len(hashtags) > 30:
        errors.append("Instagram allows at most 30 hashtags per post.")
    if len(text) > TRAINED_MAX_CAPTION:
        notes.append(f"Captions in the sample data were at most {TRAINED_MAX_CAPTION} characters, "
                     "so treat this prediction as a rough guess.")

    features = None
    if not errors:
        features = {
            "followers": followers,
            "caption_length": len(text),
            "hashtags": len(hashtags),
            "posting_hour": posting_hour,
            "has_cta": int(values["has_cta"]),
            "post_type": values["post_type"],
            "day_of_week": values["day_of_week"],
        }
    return values, features, errors, notes


def predict_one(features):
    row = pd.DataFrame([features])
    result = predict_engagement(get_models(), row).iloc[0]
    return float(result["likes"]), float(result["comments"]), float(result["engagement_rate"])


def compare_post_types(features):
    """Same post, but as each post type. Shows which format the model likes best."""
    rows = pd.DataFrame([{**features, "post_type": t} for t in POST_TYPES])
    rates = predict_engagement(get_models(), rows)["engagement_rate"].to_numpy()
    top = max(float(rates.max()), 0.01)
    return [{"post_type": t, "rate": round(float(r), 2), "width": round(float(r) / top * 100),
             "selected": t == features["post_type"]} for t, r in zip(POST_TYPES, rates)]


def find_best_hour(features):
    """Try all 24 hours and report the one with the highest predicted rate."""
    rows = pd.DataFrame([{**features, "posting_hour": h} for h in range(24)])
    rates = predict_engagement(get_models(), rows)["engagement_rate"].to_numpy()
    best = int(np.argmax(rates))
    return {"hour": hour_label(best), "rate": round(float(rates[best]), 2)}


def page_context():
    return {"post_types": POST_TYPES, "days": DAYS,
            "hours": [(h, hour_label(h)) for h in range(24)]}


# ---------------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template("index.html", values=DEFAULT_VALUES, errors=[], **page_context())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    values, features, errors, notes = parse_form(request.form)
    if errors:
        return render_template("index.html", values=values, errors=errors,
                               **page_context()), 400

    likes, comments, rate = predict_one(features)
    return render_template(
        "result.html",
        likes=round(likes), comments=round(comments), rate=round(rate, 2),
        category=rate_category(rate),
        features=features, hour=hour_label(features["posting_hour"]),
        type_comparison=compare_post_types(features),
        best_hour=find_best_hour(features),
        notes=notes,
    )


@app.route("/about")
def about():
    return render_template("about.html", metrics=load_metrics(), columns=COLUMN_INFO)


if __name__ == "__main__":
    get_models()  # train now if needed, so the first page load is quick
    app.run(debug=True)
