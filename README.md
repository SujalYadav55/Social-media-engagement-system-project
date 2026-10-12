##Social Media Engagement Prediction System

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
