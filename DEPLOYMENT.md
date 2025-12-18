# ☁️ Deployment Guide

Since you are running on a **Windows ARM64** device where local dependencies are difficult to install, the easiest way to see your application "live" is to deploy it to the cloud.

The **Streamlit App** in this repository is designed to run independently (it doesn't need the separate API running), making it perfect for **Streamlit Community Cloud** (which is free).

## 🚀 Option 1: Streamlit Community Cloud (Recommended)

1.  **Push your latest code to GitHub** (We have already done this).
2.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign up/login with GitHub.
3.  Click **"New app"**.
4.  Select your repository: `millenniumsingha/StyleNet`
5.  Select Branch: `feature/upgrade-to-production`
6.  Main file path: `app/streamlit_app.py`
7.  Click **"Deploy!"**

Streamlit Cloud runs on Linux (x64), so it will successfully install `tensorflow` and `pandas`, and your app will go live in minutes.

## 🔗 Adding a Demo to Your README

Once your app is live, you will get a URL (e.g., `https://your-app-name.streamlit.app`).

### 1. Add a Badge
Copy this code into your `README.md` to add a clickable badge:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<YOUR-APP-URL>)
```

### 2. Add a GIF Demo
1.  Open your live app.
2.  Use a screen recorder (like LICEcap or Windows Screen Recorder) to record yourself uploading an image and getting a prediction.
3.  Save the recording as `demo.gif`.
4.  Add it to the `images/` folder in your repo.
5.  Update `README.md`:
    ```markdown
    ## 🎥 Live Demo
    ![App Demo](images/demo.gif)
    ```

## 🐳 Option 2: Docker (If you fix Docker Desktop)

If you get Docker Desktop running on your machine in the future:

```bash
docker-compose up --build
```

This will spin up both the API and the Streamlit app locally.
