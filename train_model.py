import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import generate_dataset
from predictor import predict_engagement
from config import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, METRICS_PATH,
                    MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE, TARGETS)


def build_model(regressor):
    """Wrap a regressor with the data preparation steps it needs."""
    other_numbers = [c for c in NUMERIC_FEATURES if c != "followers"]
    prepare = ColumnTransformer([
        # Follower counts range from 200 to 500,000, so take the log first
        ("followers", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                ("scale", StandardScaler())]), ["followers"]),
        ("numbers", StandardScaler(), other_numbers),                                  # put numbers on one scale
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),  # text -> 0/1 columns
    ])
    pipeline = Pipeline([("prepare", prepare), ("model", regressor)])
    # Rates are skewed (many small values, a few big ones), so learn on log(rate)
    return TransformedTargetRegressor(regressor=pipeline,
                                      func=np.log1p, inverse_func=np.expm1)


def score(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def feature_importance(fitted_forest):
    """Add up how much the forest relied on each original input column."""
    pipeline = fitted_forest.regressor_
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    result = {}
    for feature in FEATURES:
        total = 0.0
        for name, value in zip(names, importances):
            column = name.split("__", 1)[1]
            if column == feature or column.startswith(feature + "_"):
                total += value
        result[feature] = round(float(total) * 100, 1)
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def main():
    if not DATA_PATH.exists():
        print("Dataset not found - creating the sample dataset first...")
        generate_dataset.main()

    df = pd.read_csv(DATA_PATH)
    # The two things the models learn: likes and comments per 100 followers
    df["likes_rate"] = df["likes"] / df["followers"] * 100
    df["comments_rate"] = df["comments"] / df["followers"] * 100

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Rows: {len(df)} total, {len(train_df)} for training, {len(test_df)} for testing\n")

    candidates = {
        "Linear Regression": LinearRegression,
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=RANDOM_STATE),
    }

    best_models = {}
    metrics = {"models": {}}
    forest_for_importance = None

    for target in TARGETS:
        metrics["models"][target] = {}
        best_name, best_r2, best_model = None, -999, None
        print(f"--- {target} (percentage points) ---")
        for name, make_regressor in candidates.items():
            model = build_model(make_regressor())
            model.fit(train_df[FEATURES], train_df[target])
            result = score(test_df[target], model.predict(test_df[FEATURES]))
            metrics["models"][target][name] = result
            print(f"{name:<18} MAE={result['mae']:>7}  RMSE={result['rmse']:>7}  R2={result['r2']:>6}")
            if result["r2"] > best_r2:
                best_name, best_r2, best_model = name, result["r2"], model
            if target == "likes_rate" and name == "Random Forest":
                forest_for_importance = model
        metrics["models"][target]["best"] = best_name
        best_models[target] = best_model
        print(f"Best for {target}: {best_name}\n")

    # Final check on the numbers the user will actually see
    predicted = predict_engagement(best_models, test_df)
    metrics["final"] = {
        "likes": score(test_df["likes"], predicted["likes"]),
        "comments": score(test_df["comments"], predicted["comments"]),
        "engagement_rate": score(test_df["engagement_rate"], predicted["engagement_rate"]),
    }
    print("--- FINAL PREDICTIONS ON THE TEST POSTS ---")
    for name, result in metrics["final"].items():
        print(f"{name:<16} MAE={result['mae']:>9}  RMSE={result['rmse']:>9}  R2={result['r2']:>6}")

    metrics["dataset"] = {"rows": len(df), "train_rows": len(train_df),
                          "test_rows": len(test_df)}
    metrics["feature_importance"] = feature_importance(forest_for_importance)
    metrics["sklearn_version"] = sklearn.__version__

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"\nSaved models to {MODEL_PATH}")


if __name__ == "__main__":
    main()
