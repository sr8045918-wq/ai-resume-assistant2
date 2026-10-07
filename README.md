# ai-resume-assistant2
# 📄 ATS Resume Checker

A Streamlit app that scores your resume for Applicant Tracking Systems (ATS) and gives specific, actionable improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for tailored keyword matching
- **Overall ATS score (0-100)** with a 5-part weighted breakdown:
  | Category | Weight |
  |---|---|
  | Keywords & Relevance | 30% |
  | Content & Impact | 25% |
  | ATS-Friendly Formatting | 20% |
  | Structure & Sections | 15% |
  | Readability & Language | 10% |
- Strengths, weaknesses, and prioritized improvements (High / Medium / Low)
- Missing vs. found keywords
- Suggested bullet-point rewrites
- Download the report as JSON

## Project structure
```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/app/apikey
2. Install and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide the key in one of these ways:
   - Paste it in the app sidebar, **or**
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # GEMINI_MODEL = "gemini-2.5-flash"   # optional
     ```
   - Or set the environment variable `GEMINI_API_KEY`.

> ⚠️ Never commit your API key or `secrets.toml` to GitHub.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## How it works
1. Text is extracted from the uploaded file (`pypdf` / `python-docx`).
2. The text (plus optional job description) is sent to Gemini Flash with a structured prompt that returns JSON.
3. The app validates the JSON, clamps scores to 0-100, and computes the weighted overall score itself for consistency.

## Limitations
- Scanned or image-only PDFs can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The score is an AI-based estimate, not the output of a real ATS such as Workday or Greenhouse. Use it as guidance.
- Your resume is sent to Google's Gemini API for analysis; the app does not store it.
