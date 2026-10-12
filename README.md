# Social Media Engagement Prediction System

A beginner-friendly college mini project. You describe an Instagram post (followers, post type, caption, day and time) and the web app predicts its **likes**, **comments** and **engagement rate**.

- Language and framework: Python 3 and Flask
- Interface: HTML and CSS (plain, no frameworks), with a few lines of JavaScript for a live caption counter
- Machine learning: scikit-learn (Linear Regression and Random Forest, compared and the better one kept)
- Data: a built-in **sample dataset of 800 computer-generated posts** (not real Instagram data)

Tested on Python 3.12 with Flask 3.1, pandas 3.0, NumPy 2.4 and scikit-learn 1.8, on Linux. The setup steps below are written for Windows but I could not run them on a Windows machine, so if a step behaves differently, see Troubleshooting.

---

## What the app does

| Page | What you can do |
|------|-----------------|
| **Predict** (`/`) | Fill in the form: followers, post type, caption (hashtags are detected automatically), day and hour, and whether the caption asks viewers to act |
| **Results** (`/predict`) | See predicted engagement rate, likes and comments, a comparison of the four post types, and the best hour to post |
| **About** (`/about`) | Read how the system works, the dataset columns, model accuracy and which inputs mattered most |

---

## Setup and run on Windows

### Step 0: Install Python (once)

1. Download Python 3.10 or newer from <https://www.python.org/downloads/>.
2. On the first installer screen, **tick "Add python.exe to PATH"**, then choose *Install Now*.
3. Open a new **Command Prompt** (press the Windows key, type `cmd`, press Enter) and check:

```
python --version
```

You should see something like `Python 3.12.x`. If Windows says Python was not found, see Troubleshooting.

### Option A: one double-click

1. Unzip the project.
2. Double-click **`setup_and_run.bat`**.
3. Wait for the packages to install. The first run also trains the model (about 10 seconds).
4. When you see `Running on http://127.0.0.1:5000`, open that address in your browser.

### Option B: step by step (recommended so you understand each step)

Open Command Prompt in the project folder. The quickest way: open the folder in File Explorer, click the address bar, type `cmd` and press Enter.

```
:: 1. Create a virtual environment (a private box for this project's packages)
python -m venv venv

:: 2. Turn it on. You should now see (venv) at the start of the line
venv\Scripts\activate

:: 3. Install the packages
pip install -r requirements.txt

:: 4. Create the sample dataset
python generate_dataset.py

:: 5. Train the models
python train_model.py

:: 6. Start the web app
python app.py
```

Now open **<http://127.0.0.1:5000>** in your browser. Press **Ctrl+C** in the Command Prompt to stop the server.

Next time you only need steps 2 and 6 (activate, then `python app.py`). Steps 4 and 5 are optional because `app.py` creates the dataset and trains the model by itself if they are missing.

### Run the tests

With the virtual environment active:

```
python -m unittest discover tests -v
```

You should see `OK` after 7 tests.

### On macOS or Linux

Replace step 2 with `source venv/bin/activate` and use `python3` where `python` does not work.

---

## Try it

Use these on the Predict page to see the model react:

| Scenario | Followers | Type | Time | Expected result |
|----------|-----------|------|------|-----------------|
| Strong post | 5,000 | Reel | Wednesday 18:00, caption with a question and 2 hashtags | About 5.7% |
| Weak post | 5,000 | Image | Wednesday 03:00, no call to action | About 3.1% |

Your exact numbers can differ slightly if you retrain with a different scikit-learn version.

---

## Project structure

```
engagement_predictor/
|-- app.py                 Flask web app (routes, form checking, predictions)
|-- config.py              Shared settings: file paths, column names, lists
|-- generate_dataset.py    Creates data/instagram_posts.csv (sample data)
|-- train_model.py         Trains, compares and saves the models
|-- predictor.py           Turns model output into likes, comments, rate
|-- requirements.txt       Packages to install
|-- setup_and_run.bat      One-click Windows launcher
|-- data/
|   `-- instagram_posts.csv
|-- models/                Created by train_model.py (model + accuracy scores)
|-- templates/             HTML pages (base, index, result, about)
|-- static/
|   |-- style.css          All styling
|   `-- app.js             Live caption character and hashtag counter
|-- tests/test_app.py      Automated tests
`-- docs/PROJECT_REPORT.md Full project report
```

---

## How it works in one minute

1. `generate_dataset.py` writes 800 sample posts to a CSV file.
2. `train_model.py` reads the CSV and splits it: 80% to learn from, 20% to test on.
3. It trains two algorithms to predict **likes per 100 followers** and **comments per 100 followers**, compares them and keeps the better one for each.
4. `app.py` loads the saved models. When you submit the form it builds one row of data, asks the models for the two percentages and calculates:
   - likes = followers x likes percentage / 100
   - comments = followers x comments percentage / 100
   - **engagement rate = likes percentage + comments percentage**

Predicting percentages instead of raw counts keeps the model fair to small and large accounts.

Read `docs/PROJECT_REPORT.md` for the full explanation, results and limitations.

---

## Use your own dataset

1. Make a CSV with these columns: `followers, post_type, caption_length, hashtags, posting_hour, day_of_week, has_cta, likes, comments, engagement_rate` (see the About page for what each means). `post_type` must be one of image, carousel, video, reel.
2. Save it as `data/instagram_posts.csv`.
3. Run `python train_model.py`, then restart the app.

---

## Troubleshooting

**`'python' is not recognized`**
Python is not on PATH. Re-run the installer, choose *Modify*, and tick *Add Python to environment variables*. Or try `py` instead of `python` (for example `py -m venv venv`). Open a new Command Prompt afterwards.

**`venv\Scripts\activate` is blocked in PowerShell ("running scripts is disabled")**
Use Command Prompt (`cmd`) instead of PowerShell, or run this once in PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**`ModuleNotFoundError: No module named 'flask'` (or pandas, sklearn)**
The virtual environment is not active or the packages are not installed. Run `venv\Scripts\activate`, then `pip install -r requirements.txt`.

**`Address already in use` or port 5000 is busy**
Another program is using port 5000. Change the last line of `app.py` to `app.run(debug=True, port=5001)` and open `http://127.0.0.1:5001`.

**Warnings or an error about the saved model after upgrading packages**
Delete `models\engagement_model.pkl` and run `python train_model.py` again. The app also retrains by itself if it cannot load the file.

**`pip install` fails or is very slow**
Check your internet connection, then try `python -m pip install --upgrade pip` and run the install command again.

**The page looks unstyled**
Hard-refresh the browser (Ctrl+F5) and make sure you started the app from inside the project folder.

---

## Limitations

- The dataset is synthetic. The model learned patterns that were written into the generator, so it does not tell you anything new about real Instagram.
- The form leaves out image quality, topic, audience, trends and the platform algorithm.
- Accounts outside 200 to 500,000 followers are outside the data. The app warns you.
- The engagement-rate bands (low, average, high, very high) are a rough rule of thumb used by this project, not an official standard.
