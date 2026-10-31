# Career Pivot Coach

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

5. Enter your Groq API key in the sidebar. Get one from https://console.groq.com/.

## Features
- Resume intake: PDF, DOCX and TXT.
- Resume-to-real-job-description skill map with resume and job evidence.
- Four-week learning roadmap with measurable tasks.
- Portfolio project suggestions linked to skill gaps.
- Mock interview question generation and rubric scoring.
- Progress tracking and adaptive roadmap revision.
- Application tracker with role-category response-rate analysis.
- SQLite persistence in `career_coach.db`.

## Important notes
- Paste the actual employer job description to ground the skill-gap analysis. The app does not scrape or verify job postings.
- "Not mentioned" in a resume is not proof that a candidate lacks a skill.
- Application response insights are descriptive and should be treated cautiously when sample sizes are small.
- Keep API keys private. For a public deployment, use Streamlit secrets or environment variables and add authentication/database hosting.
