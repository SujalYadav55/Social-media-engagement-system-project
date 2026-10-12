# Social Media Engagement Prediction System

**A Mini Project Report**

Student name: ____________________  Roll no.: ____________  
Department / Course: ____________________  
Guide: ____________________  Academic year: ____________

---

## Abstract

Social media creators want to know how a post will perform before they publish it. This project builds a web application that predicts the **likes**, **comments** and **engagement rate** of an Instagram post from details known in advance: the account's follower count, the post type, the caption length, the number of hashtags, the day and hour of posting, and whether the caption contains a call to action. The system is written in Python. Two machine learning algorithms, Linear Regression and Random Forest, are trained and compared using a sample dataset of 800 computer-generated posts, and the better one is used by a Flask web application with an HTML and CSS interface. On 160 unseen test posts the final predictions reached an R² of 0.88 for likes, 0.69 for comments and 0.45 for engagement rate. Because the dataset is synthetic, the results show that the pipeline works end to end and are not evidence about real Instagram behaviour.

**Keywords:** engagement prediction, regression, Random Forest, Flask, Instagram, scikit-learn

---

## 1. Introduction

Instagram posts differ widely in how much attention they receive. A post by an account with 500 followers cannot be compared with one by an account with 500,000 followers using raw like counts, so analysts use the **engagement rate**: the interactions a post receives as a share of the account's followers. Being able to estimate this rate before posting helps decide on the format, timing and caption style.

Machine learning can learn such relationships from past posts. This project applies supervised regression to a small dataset and wraps the result in a simple web interface so that anyone can try it without writing code.

## 2. Problem Statement

Given the features of a planned Instagram post, predict how many likes and comments it will receive and its engagement rate, and present the answer in an easy-to-use web application.

## 3. Objectives

1. Build a sample dataset of Instagram posts with the features and targets described below.
2. Train and compare at least two regression algorithms and keep the better one.
3. Predict likes, comments and engagement rate for a new post.
4. Provide a Flask web application with an HTML and CSS interface.
5. Document the system and provide setup instructions for Windows.

## 4. Scope and Limitations

**In scope:** a single platform (Instagram), post-level features that are known before posting, a sample dataset, and a local web application.

**Out of scope:** image or video content analysis, live data from Instagram, user accounts, and deployment to a public server.

**Limitations:**
- The dataset is synthetic. It was generated with rules based on common patterns (for example, reels and carousels receive more engagement than single images, and evenings are popular posting times). The model therefore learns the rules written into the generator and cannot reveal true behaviour of real audiences.
- Factors such as image quality, topic, trends, audience and the platform's ranking algorithm are not available to the model.
- Accounts outside 200 to 500,000 followers are outside the training data.

## 5. System Requirements

| Item | Requirement |
|------|-------------|
| Operating system | Windows 10 or 11 (also works on macOS and Linux) |
| Python | 3.10 or newer |
| Packages | Flask, pandas, NumPy, scikit-learn, joblib |
| Browser | Any modern browser |
| Hardware | Any computer that runs Python; training takes about 10 seconds |

## 6. System Architecture

```
          +-------------------+
          | generate_dataset  |  creates sample data
          +---------+---------+
                    |
                    v
          data/instagram_posts.csv
                    |
                    v
          +-------------------+
          |   train_model     |  split, train, compare, save
          +---------+---------+
                    |
                    v
          models/engagement_model.pkl  +  models/metrics.json
                    |
                    v
+---------+   form   +-------------+  features  +-----------+
| Browser | -------> |  Flask app  | ---------> | predictor |
|  HTML   | <------- |   (app.py)  | <--------- |   .py     |
|  + CSS  |  result  +-------------+ likes,     +-----------+
+---------+                          comments,
                                     rate
```

**Modules**

| File | Responsibility |
|------|----------------|
| `config.py` | File paths, column lists and constants shared by all modules |
| `generate_dataset.py` | Creates the sample dataset |
| `train_model.py` | Prepares data, trains and compares models, saves the best |
| `predictor.py` | Converts model output into likes, comments and engagement rate |
| `app.py` | Flask routes, form validation, extra analysis (post type comparison, best hour) |
| `templates/*.html` | Page layouts (Jinja2 templates) |
| `static/style.css`, `static/app.js` | Styling and the live caption counter |
| `tests/test_app.py` | Automated tests |

## 7. Dataset

### 7.1 Source

The file `data/instagram_posts.csv` is produced by `generate_dataset.py` with a fixed random seed (42), so every run gives the same 800 posts. It is a **synthetic sample dataset** meant for learning. A real dataset with the same columns can be used instead (see the README).

### 7.2 Columns

| Column | Type | Description |
|--------|------|-------------|
| `followers` | integer | Followers when the post was published |
| `post_type` | text | image, carousel, video or reel |
| `caption_length` | integer | Characters in the caption, excluding hashtags |
| `hashtags` | integer | Number of hashtags (0 to 30) |
| `posting_hour` | integer | Hour of publication (0 to 23) |
| `day_of_week` | text | Monday to Sunday |
| `has_cta` | 0 or 1 | Whether the caption asks viewers to act |
| `likes` | integer | Target |
| `comments` | integer | Target |
| `engagement_rate` | decimal | (likes + comments) / followers x 100 |

### 7.3 Sample rows

```
post_id,followers,post_type,caption_length,hashtags,posting_hour,day_of_week,has_cta,likes,comments,engagement_rate
1,6872,video,27,6,13,Thursday,0,172,7,2.6
2,1566,video,25,11,19,Monday,0,89,5,6.0
3,11220,carousel,161,10,5,Monday,0,361,8,3.29
4,13830,video,100,7,16,Tuesday,0,393,16,2.96
5,575,reel,222,9,8,Wednesday,1,30,3,5.74
```

### 7.4 Summary

| Measure | Value |
|---------|-------|
| Posts | 800 |
| Followers: median / mean / max | 4,868 / 8,520 / 121,206 |
| Engagement rate: mean / median | 4.86% / 4.52% |
| Average rate by post type | Reel 5.62%, Carousel 5.38%, Video 4.47%, Image 3.99% |
| Posts by type | Image 274, Reel 247, Carousel 175, Video 104 |

### 7.5 How the synthetic data was generated

1. Inputs are drawn at random: followers from a skewed distribution (many small accounts, few large ones), post types with fixed probabilities, hours weighted toward lunchtime and evening, and so on.
2. A base engagement rate is set from the follower count (smaller accounts get a higher base rate).
3. The base rate is multiplied by factors for post type, day, hashtag count (best near 10), caption length (best at a medium length), posting hour (peaks near 12:00 and 19:00) and call to action.
4. Random noise is applied so that the data is not perfectly predictable.
5. The resulting interactions are split into likes and comments.

## 8. Methodology

### 8.1 Data preparation
- Likes and comments are converted to **percentages of followers** (`likes_rate`, `comments_rate`). This makes posts from accounts of very different sizes comparable. The models predict these two percentages.
- Follower counts are log-transformed because they span a very wide range.
- Numeric inputs are standardised; `post_type` and `day_of_week` are one-hot encoded.
- The rate targets are learned on a log scale because they are skewed, and converted back afterwards.

### 8.2 Train and test split
The data is split 80% (640 posts) for training and 20% (160 posts) for testing, with a fixed random state so results can be repeated. The test posts are never used for training.

### 8.3 Algorithms
- **Linear Regression:** fits a straight-line relationship. It is simple and acts as the baseline.
- **Random Forest Regressor:** builds 200 decision trees on random parts of the data and averages their answers. It can capture non-linear effects such as "the best posting hour is in the evening, not midnight".

For each target the algorithm with the higher R² on the test set is kept.

### 8.4 Evaluation measures
- **MAE (mean absolute error):** the average size of the miss, in the unit of the target.
- **RMSE (root mean squared error):** like MAE but penalises large misses more.
- **R² (coefficient of determination):** 1 means perfect, 0 means no better than always guessing the average.

### 8.5 From predictions to results

```
likes            = followers x likes_rate / 100
comments         = followers x comments_rate / 100
engagement rate  = likes_rate + comments_rate          (in percent)
```

### 8.6 Extra analysis in the app
- **Post type comparison:** the same post is predicted four times, once per post type.
- **Best time to post:** the same post is predicted for each of the 24 hours and the highest is reported.
- **Rate label:** below 1% Low, 1 to 3% Average, 3 to 6% High, above 6% Very high. These bands are a rough rule of thumb chosen for this project, not an official standard.

## 9. Implementation

### 9.1 Technologies
Python 3, Flask (web framework), pandas and NumPy (data handling), scikit-learn (machine learning), joblib (saving models), HTML and CSS (interface), a small amount of JavaScript (live caption counter).

### 9.2 Web application flow
1. `GET /` shows the form with sensible default values.
2. `POST /predict` reads the form. `parse_form()` validates every field, extracts hashtags from the caption and measures the caption length without them. If anything is invalid, the form is shown again with a clear message (HTTP 400).
3. For valid input the app predicts, runs the post type comparison and best-hour search, and renders `result.html`.
4. `GET /about` shows the dataset description, accuracy tables and feature importance read from `models/metrics.json`.

### 9.3 Input validation
- Followers must be a whole number from 1 to 50,000,000. Values outside the training range 200 to 500,000 are accepted but show a warning.
- At most 30 hashtags and 2,200 characters, which are Instagram's own limits.
- Post type, day and hour must be one of the allowed values.

### 9.4 Model loading
The models are loaded once when first needed. If the model file is missing, or was saved by a different scikit-learn version and cannot be loaded, the app trains it again automatically.

## 10. Results

### 10.1 Algorithm comparison (test set, 160 posts)

| Target | Algorithm | MAE | RMSE | R² | Kept |
|--------|-----------|-----|------|----|------|
| Likes per 100 followers | Linear Regression | 0.974 | 1.370 | 0.406 | |
| Likes per 100 followers | Random Forest | 0.957 | 1.325 | 0.445 | Yes |
| Comments per 100 followers | Linear Regression | 0.078 | 0.103 | 0.336 | Yes |
| Comments per 100 followers | Random Forest | 0.079 | 0.104 | 0.327 | |

Random Forest was slightly better for likes. For comments the two were nearly tied and Linear Regression scored marginally higher. The small gap between the two algorithms is expected on a small, noisy dataset whose effects are mostly smooth.

### 10.2 Final predictions (what the user sees)

| Prediction | MAE | RMSE | R² |
|------------|-----|------|----|
| Likes | 78.4 likes | 171.6 | 0.882 |
| Comments | 5.7 comments | 11.7 | 0.689 |
| Engagement rate | 1.00 percentage points | 1.39 | 0.448 |

Likes and comments reach higher R² mainly because they scale with the follower count, which the model knows. Engagement rate removes that effect, so its R² of 0.45 is a fairer picture of how much the post details explain. The rest is the random variation built into the data.

### 10.3 Feature importance (Random Forest, likes model)

| Feature | Share of importance |
|---------|--------------------|
| followers | 37.3% |
| post_type | 21.2% |
| posting_hour | 17.3% |
| caption_length | 10.5% |
| day_of_week | 5.9% |
| hashtags | 5.2% |
| has_cta | 2.6% |

This ordering reflects the strength of the effects written into the synthetic generator. It should not be read as a finding about real Instagram.

### 10.4 Sample outputs

| Input | Likes | Comments | Engagement rate |
|-------|-------|----------|-----------------|
| 5,000 followers, reel, Wednesday 18:00, question caption with 2 hashtags, call to action | 271 | 16 | 5.74% (High) |
| 5,000 followers, image, Wednesday 03:00, same caption, no call to action | 150 | 4 | 3.09% (High) |

For the first example the best hour reported by the app was 20:00 (5.98%), and the post type comparison ranked reel (5.74%) and carousel (5.67%) above video (4.54%) and image (3.99%).

*Your numbers may differ slightly if you train with a different scikit-learn version.*

### 10.5 Screenshots

*Insert your own screenshots here after running the app:*

- Figure 1: Predict page
- Figure 2: Results page
- Figure 3: Validation error message
- Figure 4: About page with accuracy tables

## 11. Testing

Automated tests in `tests/test_app.py` (run with `python -m unittest discover tests -v`):

| Test | Checks |
|------|--------|
| Pages load | `/` and `/about` return status 200 |
| Valid prediction | A correct form returns a result page containing the predicted rate and likes |
| Bad followers | The text "abc" is rejected with status 400 and a clear message |
| Too many hashtags | 31 hashtags are rejected |
| Caption parsing | Hashtags are counted and left out of the caption length |
| Rate labels | 0.5, 2, 4 and 8 map to Low, Average, High, Very high |
| Follower effect | Different follower counts give different results |

All 7 tests passed.

## 12. Conclusion

The project delivers a complete working pipeline: data creation, model training and comparison, and a web application that returns predicted likes, comments and engagement rate with a post type comparison and a best-hour suggestion. Random Forest gave the best likes model on the test posts, and Linear Regression was equally good for comments. Since the data is synthetic, the accuracy figures describe how well the model recovers the patterns built into the sample data, not how well it would predict real posts.

## 13. Future Scope

1. Train on a real, properly licensed Instagram dataset, or on data collected with the platform's official tools.
2. Add features such as caption sentiment, topic, image features, and the account's recent average engagement.
3. Try gradient boosting models and tune hyperparameters with cross-validation.
4. Extend to other platforms (YouTube, X, LinkedIn).
5. Add user accounts and a history of predictions, and deploy the app to a public server.

## 14. References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61.
4. Flask documentation. https://flask.palletsprojects.com
5. scikit-learn documentation. https://scikit-learn.org
6. pandas documentation. https://pandas.pydata.org

---

## Appendix A: Setup and run on Windows

See `README.md`. In short:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://127.0.0.1:5000 in a browser.

## Appendix B: Project files

See the folder structure in `README.md`. Source code is in `app.py`, `config.py`, `generate_dataset.py`, `train_model.py`, `predictor.py`, the `templates` folder and the `static` folder.
