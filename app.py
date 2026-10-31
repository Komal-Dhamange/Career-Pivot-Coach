import os
import re
import json
import sqlite3
from datetime import datetime, date
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq
from pypdf import PdfReader
from docx import Document

# ============================================================
# CAREER PIVOT COACH — Hackathon MVP
# Streamlit + Groq + SQLite
# ============================================================

st.set_page_config(
    page_title="Career Pivot Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.getenv("CAREER_COACH_DB", "career_coach.db")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------- Styling -----------------------
st.markdown("""
<style>
.stApp { background: #f7f8fc; }
[data-testid="stSidebar"] { background: #eef1f8; }
.block-container { padding-top: 1.6rem; max-width: 1450px; }
.hero {
  padding: 1.6rem 1.8rem; border-radius: 18px; color: white;
  background: linear-gradient(115deg,#14213d,#46327e 65%,#7655a6);
  margin-bottom: 1.2rem;
}
.hero h1 { color: white; margin-bottom: .35rem; }
.hero p { color: #eef0ff; margin-bottom: 0; }
.metric-card {
  background: white; padding: 1rem 1.1rem; border-radius: 14px;
  border: 1px solid #e8eaf2; min-height: 105px;
}
.small-muted { color: #6b7280; font-size: .88rem; }
div[data-testid="stMetric"] {
  background: white; border: 1px solid #e8eaf2;
  padding: 12px 15px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------- Database ----------------------
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            resume_text TEXT DEFAULT '',
            resume_name TEXT DEFAULT '',
            target_role TEXT DEFAULT '',
            target_level TEXT DEFAULT 'Entry level',
            target_location TEXT DEFAULT '',
            job_description TEXT DEFAULT '',
            job_company TEXT DEFAULT '',
            job_url TEXT DEFAULT '',
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            report TEXT DEFAULT '',
            data_json TEXT DEFAULT '{}',
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS roadmap_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week INTEGER,
            skill TEXT,
            task TEXT,
            outcome TEXT,
            completed INTEGER DEFAULT 0,
            UNIQUE(week, skill, task)
        );
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            gap_skill TEXT,
            objective TEXT,
            features TEXT,
            deliverable TEXT
        );
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            question TEXT,
            answer TEXT,
            score REAL,
            feedback TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            job_title TEXT NOT NULL,
            role_category TEXT,
            source TEXT,
            job_url TEXT,
            applied_date TEXT,
            status TEXT,
            response_date TEXT,
            notes TEXT,
            created_at TEXT
        );
        """)
        con.execute(
            "INSERT OR IGNORE INTO profile(id, updated_at) VALUES(1, ?)",
            (datetime.now().isoformat(timespec="seconds"),)
        )
        con.execute(
            "INSERT OR IGNORE INTO analysis(id, updated_at) VALUES(1, ?)",
            (datetime.now().isoformat(timespec="seconds"),)
        )


def query(sql, params=()):
    with db() as con:
        return con.execute(sql, params).fetchall()


def execute(sql, params=()):
    with db() as con:
        cur = con.execute(sql, params)
        return cur.lastrowid


def get_profile():
    row = query("SELECT * FROM profile WHERE id=1")[0]
    return dict(row)


def save_profile(**kwargs):
    allowed = {
        "resume_text", "resume_name", "target_role", "target_level",
        "target_location", "job_description", "job_company", "job_url"
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
    assignments = ", ".join(f"{k}=?" for k in fields)
    execute(
        f"UPDATE profile SET {assignments} WHERE id=1",
        tuple(fields.values())
    )


def get_analysis():
    row = query("SELECT * FROM analysis WHERE id=1")[0]
    return dict(row)


def save_analysis(report, data):
    with db() as con:
        con.execute(
            "UPDATE analysis SET report=?, data_json=?, updated_at=? WHERE id=1",
            (report, json.dumps(data, ensure_ascii=False),
             datetime.now().isoformat(timespec="seconds"))
        )


init_db()


# ---------------------------- Resume parsing -----------------
def extract_resume(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if name.endswith(".docx"):
        doc = Document(BytesIO(raw))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(chunks).strip()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="replace").strip()
    raise ValueError("Unsupported file type. Upload PDF, DOCX or TXT.")


# ---------------------------- AI helpers ---------------------
def get_client():
    key = st.session_state.get("groq_api_key", "").strip()
    if not key:
        raise ValueError("Groq API key missing. Add it in the sidebar.")
    return Groq(api_key=key)


def ask_ai(prompt, system=None, temperature=0.25, max_tokens=3500):
    client = get_client()
    response = client.chat.completions.create(
        model=st.session_state.get("model_name", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system or
             "You are a careful, practical career coach and technical recruiter. "
             "Use only supplied evidence for candidate claims. Never invent skills, "
             "experience, employers, credentials, job descriptions, or outcomes. "
             "Distinguish 'not shown in resume' from 'candidate cannot do this'. "
             "Be concise, specific, supportive, and action-oriented."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def ask_ai_json(prompt, system=None, max_tokens=3500):
    raw = ask_ai(
        prompt + "\n\nReturn ONLY valid JSON. Do not use Markdown fences.",
        system=system,
        max_tokens=max_tokens,
    )
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("AI returned a response that was not valid JSON. Please retry.")


def safe_list(value):
    return value if isinstance(value, list) else []


def load_analysis_data():
    try:
        return json.loads(get_analysis().get("data_json") or "{}")
    except (ValueError, TypeError):
        return {}


# ---------------------------- Sidebar ------------------------
st.sidebar.title("🚀 Career Pivot Coach")
st.sidebar.caption("Your AI-powered career navigation workspace")
st.sidebar.text_input(
    "Groq API Key", type="password", key="groq_api_key",
    help="Used only for this app session. Do not share your key."
)
st.sidebar.text_input("Groq model", value=DEFAULT_MODEL, key="model_name")
st.sidebar.caption("Tip: get an API key from console.groq.com")
pages = [
    "Dashboard", "Resume & Skill Gap", "Learning Roadmap",
    "Portfolio Projects", "Mock Interview", "Progress Tracker",
    "Application Tracker"
]
page = st.sidebar.radio("NAVIGATION", pages)
st.sidebar.divider()
st.sidebar.caption("Hackathon MVP • AI-assisted, evidence-led career planning")
st.sidebar.caption("Your local SQLite data stays in this app environment.")


# ---------------------------- Shared data --------------------
profile = get_profile()
analysis_row = get_analysis()
analysis_data = load_analysis_data()
tasks_rows = query("SELECT * FROM roadmap_tasks ORDER BY week, id")
apps_rows = query("SELECT * FROM applications ORDER BY applied_date DESC, id DESC")
interview_rows = query("SELECT * FROM interview_sessions ORDER BY id DESC")


def task_dataframe():
    rows = query("SELECT * FROM roadmap_tasks ORDER BY week, id")
    return pd.DataFrame([dict(r) for r in rows])


def app_dataframe():
    rows = query("SELECT * FROM applications ORDER BY applied_date DESC, id DESC")
    return pd.DataFrame([dict(r) for r in rows])


def make_roadmap_from_analysis(data):
    with db() as con:
        con.execute("DELETE FROM roadmap_tasks")
        con.execute("DELETE FROM projects")
        for week_obj in safe_list(data.get("roadmap")):
            week_num = int(week_obj.get("week", 1))
            for item in safe_list(week_obj.get("tasks")):
                if isinstance(item, str):
                    item = {"skill": "General", "task": item, "outcome": "Complete the task"}
                con.execute("""
                    INSERT OR IGNORE INTO roadmap_tasks(week, skill, task, outcome)
                    VALUES(?,?,?,?)
                """, (
                    week_num, str(item.get("skill", "General")),
                    str(item.get("task", "")), str(item.get("outcome", ""))
                ))
        for project in safe_list(data.get("projects")):
            con.execute("""
                INSERT INTO projects(title, gap_skill, objective, features, deliverable)
                VALUES(?,?,?,?,?)
            """, (
                str(project.get("title", "Portfolio project")),
                str(project.get("gap_skill", "")),
                str(project.get("objective", "")),
                "\n".join(safe_list(project.get("features"))),
                str(project.get("deliverable", ""))
            ))


# ---------------------------- Dashboard ----------------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>🚀 Career Pivot Coach</h1>
      <p>Turn your career goal into a measurable, adaptive job-readiness plan.</p>
      <p>Resume evidence · Real job descriptions · Skill gaps · Learning · Practice · Outcomes</p>
    </div>
    """, unsafe_allow_html=True)

    df_tasks = task_dataframe()
    df_apps = app_dataframe()
    completed = int(df_tasks["completed"].sum()) if not df_tasks.empty else 0
    total_tasks = len(df_tasks)
    app_count = len(df_apps)
    interview_count = len(interview_rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Career analysis", "Complete" if analysis_row["report"] else "Not started")
    c2.metric("Learning tasks", f"{completed}/{total_tasks}")
    c3.metric("Applications", app_count)
    c4.metric("Interview evaluations", interview_count)

    st.subheader("Your career workspace")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Recommended workflow")
        st.markdown("""
        1. Upload your resume and paste a real job description.
        2. Review evidence-backed skill matches and missing evidence.
        3. Follow the weekly plan and build gap-matched portfolio projects.
        4. Practice interview answers and review rubric scores.
        5. Track applications and use response patterns to refine your strategy.
        """)
        if not profile.get("target_role"):
            st.info("Start in **Resume & Skill Gap** to set your target role.")
        else:
            st.success(f"Current target: **{profile['target_role']}**")
    with right:
        st.markdown("#### Learning progress")
        if total_tasks:
            st.progress(completed / total_tasks)
            st.write(f"{round(completed / total_tasks * 100)}% of roadmap tasks completed")
        else:
            st.info("Your learning progress will appear after generating a roadmap.")
        if not df_apps.empty:
            st.markdown("#### Application outcomes")
            st.write(df_apps["status"].value_counts().to_dict())
        else:
            st.info("Add applications to unlock response analytics.")


# ---------------------------- Resume and skill-gap -----------
elif page == "Resume & Skill Gap":
    st.title("🔍 Resume & Target-role Skill Gap Analysis")
    st.write("Compare resume evidence against requirements from a real job description.")
    with st.expander("Candidate profile", expanded=True):
        uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Target job role", value=profile.get("target_role", ""),
                                 placeholder="e.g. Junior Data Scientist")
            level = st.selectbox(
                "Target seniority", ["Internship", "Entry level", "Junior",
                                     "Mid-level", "Senior", "Career switcher"],
                index=["Internship", "Entry level", "Junior", "Mid-level",
                       "Senior", "Career switcher"].index(
                    profile.get("target_level", "Entry level")
                    if profile.get("target_level", "Entry level") in
                    ["Internship", "Entry level", "Junior", "Mid-level",
                     "Senior", "Career switcher"] else "Entry level"
                )
            )
        with col2:
            location = st.text_input("Preferred location / work mode",
                                     value=profile.get("target_location", ""),
                                     placeholder="e.g. Remote, Mumbai")
            company = st.text_input("Company from job description",
                                    value=profile.get("job_company", ""))
        job_url = st.text_input("Job posting URL (optional)",
                                value=profile.get("job_url", ""))
        if uploaded:
            try:
                resume_text = extract_resume(uploaded)
                st.success(f"Resume loaded: {uploaded.name} ({len(resume_text)} characters)")
            except Exception as exc:
                resume_text = ""
                st.error(str(exc))
        else:
            resume_text = profile.get("resume_text", "")
            if resume_text:
                st.caption(f"Using saved resume: {profile.get('resume_name', 'Previously uploaded')}")
        jd = st.text_area(
            "Paste the actual job description",
            value=profile.get("job_description", ""),
            height=220,
            placeholder="Paste responsibilities, required skills, qualifications and experience from the employer's job posting."
        )
        st.caption("For evidence-based results, paste the employer's real job description. "
                   "The app does not claim to crawl or verify external job sites.")
        analyze = st.button("✨ Analyze my career fit", type="primary", use_container_width=True)

    if analyze:
        if not resume_text.strip():
            st.error("Please upload a readable resume first.")
        elif not role.strip():
            st.error("Please enter your target role.")
        elif not jd.strip():
            st.error("Paste a real job description. Without it, the analysis cannot be described as job-description-based.")
        else:
            try:
                save_profile(
                    resume_text=resume_text[:18000],
                    resume_name=uploaded.name if uploaded else profile.get("resume_name", ""),
                    target_role=role.strip(), target_level=level,
                    target_location=location.strip(), job_description=jd[:12000],
                    job_company=company.strip(), job_url=job_url.strip()
                )
                prompt = f"""
Create a candidate-vs-job analysis using ONLY the supplied resume and job description.
Target role: {role}
Seniority: {level}
Location preference: {location}
Company: {company}
Job URL/source: {job_url}

RESUME:
{resume_text[:18000]}

REAL JOB DESCRIPTION:
{jd[:12000]}

Return JSON with this exact structure:
{{
 "candidate_summary": "short summary",
 "role_readiness": "Strong | Developing | Needs Preparation",
 "readiness_reason": "evidence and limitations",
 "skills": [
   {{"skill":"Python","importance":"High","resume_evidence":"specific resume evidence or Not mentioned",
     "status":"Demonstrated | Partial evidence | Not mentioned","job_evidence":"requirement from JD",
     "next_action":"practical action"}}
 ],
 "strengths": ["..."],
 "resume_improvements": ["..."],
 "roadmap": [
   {{"week":1,"theme":"...","tasks":[
      {{"skill":"...","task":"specific action","outcome":"measurable deliverable"}}]}}
 ],
 "projects": [
   {{"title":"...","gap_skill":"...","objective":"...","features":["...","..."],
     "deliverable":"..."}}
 ],
 "interview_focus": ["..."],
 "caveats": ["..."]
}}
Requirements:
- Extract skills from the supplied JD; do not add generic requirements as if employer requested them.
- For every skill, quote or accurately paraphrase job_evidence and cite resume_evidence.
- "Not mentioned" means no evidence in this resume, not proof the candidate lacks the skill.
- Prioritize gaps using the JD's own wording (required/preferred/responsibilities).
- Create a practical 4-week plan with 2-4 tasks per week and measurable outcomes.
- Suggest 2-3 portfolio projects that directly demonstrate the highest-priority gaps.
- Never invent candidate experience, metrics, degrees, certifications or projects.
"""
                with st.spinner("Comparing your resume with the job description..."):
                    data = ask_ai_json(prompt, max_tokens=5000)
                report = (
                    f"## Candidate summary\n{data.get('candidate_summary','')}\n\n"
                    f"## Role readiness: {data.get('role_readiness','Developing')}\n"
                    f"{data.get('readiness_reason','')}\n\n"
                    f"## Evidence-based skill map\n"
                )
                for s in safe_list(data.get("skills")):
                    report += (
                        f"\n### {s.get('skill','Skill')} — {s.get('status','Not mentioned')} "
                        f"({s.get('importance','Priority not stated')})\n"
                        f"- **Job requirement:** {s.get('job_evidence','')}\n"
                        f"- **Resume evidence:** {s.get('resume_evidence','')}\n"
                        f"- **Next action:** {s.get('next_action','')}\n"
                    )
                report += "\n## Strengths\n" + "\n".join(
                    f"- {x}" for x in safe_list(data.get("strengths")))
                report += "\n\n## Resume improvements\n" + "\n".join(
                    f"- {x}" for x in safe_list(data.get("resume_improvements")))
                report += "\n\n## Interview focus\n" + "\n".join(
                    f"- {x}" for x in safe_list(data.get("interview_focus")))
                report += "\n\n## Caveats\n" + "\n".join(
                    f"- {x}" for x in safe_list(data.get("caveats")))
                save_analysis(report, data)
                make_roadmap_from_analysis(data)
                st.success("Career analysis, roadmap and portfolio projects generated and saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")

    if analysis_row["report"]:
        st.divider()
        st.subheader("📊 Your personalized career report")
        st.markdown(analysis_row["report"])
        st.download_button(
            "⬇️ Download career report",
            data=analysis_row["report"],
            file_name="career_pivot_report.md",
            mime="text/markdown",
        )


# ---------------------------- Learning roadmap ---------------
elif page == "Learning Roadmap":
    st.title("🗓️ Week-by-week Learning Roadmap")
    st.write("A practical plan generated from your resume gaps and the supplied job description.")
    rows = query("SELECT * FROM roadmap_tasks ORDER BY week, id")
    if not rows:
        st.info("Generate your Skill Gap Analysis first to create a personalized roadmap.")
    else:
        df = pd.DataFrame([dict(r) for r in rows])
        total = len(df)
        done = int(df["completed"].sum())
        st.progress(done / total if total else 0)
        st.caption(f"{done} of {total} tasks complete ({round(done/total*100) if total else 0}%).")
        for week in sorted(df["week"].unique()):
            st.markdown(f"### Week {week}")
            week_df = df[df["week"] == week]
            for _, item in week_df.iterrows():
                label = f"**{item['skill']}** — {item['task']}\n\n<small>Outcome: {item['outcome']}</small>"
                checked = st.checkbox(
                    f"{item['skill']} — {item['task']}  |  Outcome: {item['outcome']}",
                    value=bool(item["completed"]), key=f"task_{item['id']}"
                )
                if checked != bool(item["completed"]):
                    execute("UPDATE roadmap_tasks SET completed=? WHERE id=?",
                            (int(checked), int(item["id"])))
        st.divider()
        st.subheader("Adapt my plan")
        st.write("The coach reviews completed and unfinished work and proposes a revised plan.")
        if st.button("🔄 Adjust roadmap using my progress", type="primary"):
            try:
                current = task_dataframe()
                summary = current[["week", "skill", "task", "outcome", "completed"]].to_dict("records")
                prompt = f"""
Target role: {profile.get('target_role')}
Original job description: {profile.get('job_description','')[:7000]}
Current roadmap tasks and completion:
{json.dumps(summary, ensure_ascii=False)}

Create an adjusted next 4-week plan. Keep unfinished high-priority skills,
reduce or split unrealistic tasks, and build on completed tasks. Do not assume
a completed task means mastery. Return JSON:
{{"reasoning":"brief explanation","roadmap":[
 {{"week":1,"theme":"...","tasks":[
  {{"skill":"...","task":"...","outcome":"measurable outcome"}}]}}
]}}
Return 4 weeks, 2-4 tasks per week.
"""
                adjusted = ask_ai_json(prompt, max_tokens=3500)
                with db() as con:
                    con.execute("DELETE FROM roadmap_tasks")
                    for w in safe_list(adjusted.get("roadmap")):
                        for t in safe_list(w.get("tasks")):
                            con.execute("""
                            INSERT OR IGNORE INTO roadmap_tasks(week,skill,task,outcome,completed)
                            VALUES(?,?,?,?,0)
                            """, (int(w.get("week", 1)), t.get("skill", "General"),
                                  t.get("task", ""), t.get("outcome", "")))
                st.success(adjusted.get("reasoning", "Roadmap updated based on your progress."))
                st.rerun()
            except Exception as exc:
                st.error(f"Could not adjust roadmap: {exc}")


# ---------------------------- Portfolio projects -------------
elif page == "Portfolio Projects":
    st.title("🧰 Gap-matched Portfolio Projects")
    st.write("Projects are mapped to the highest-priority skills not demonstrated in the resume.")
    projects = query("SELECT * FROM projects ORDER BY id")
    if not projects:
        st.info("Complete Skill Gap Analysis to generate projects matched to your skill gaps.")
    else:
        for i, p in enumerate(projects, 1):
            with st.expander(f"{i}. {p['title']}  •  Gap: {p['gap_skill']}", expanded=True):
                st.markdown(f"**Objective:** {p['objective']}")
                st.markdown("**Suggested features:**")
                for feature in (p["features"] or "").splitlines():
                    if feature.strip():
                        st.markdown(f"- {feature}")
                st.markdown(f"**Portfolio deliverable:** {p['deliverable']}")
                st.caption("Only claim this project on your resume after you have actually completed it.")


# ---------------------------- Mock interview -----------------
elif page == "Mock Interview":
    st.title("🎤 AI Mock Interview")
    st.write("Practice role-specific questions and receive transparent rubric-based feedback.")
    default_role = profile.get("target_role", "")
    role = st.text_input("Interview target role", value=default_role)
    question_type = st.selectbox(
        "Question type", ["Mixed", "Technical", "Project-based", "Behavioral", "Problem-solving"]
    )
    if st.button("Generate a new interview question", type="primary"):
        if not role.strip():
            st.warning("Enter a target role.")
        else:
            try:
                with st.spinner("Preparing a realistic interview question..."):
                    result = ask_ai_json(f"""
Create ONE realistic {question_type} interview question for {role}.
Use this target job description as context:
{profile.get('job_description','')[:5000]}
Candidate resume evidence:
{profile.get('resume_text','')[:5000]}
Return JSON: {{"question":"...","why_asked":"...","strong_answer_should_include":["...","..."],
"rubric":["Relevance","Technical accuracy","Structure and clarity","Evidence/examples"]}}.
Do not assume experience not present in the resume.
""")
                st.session_state["current_interview"] = result
            except Exception as exc:
                st.error(f"Could not generate question: {exc}")

    q = st.session_state.get("current_interview")
    if q:
        st.subheader("Your question")
        st.markdown(f"**{q.get('question','')}**")
        st.caption(q.get("why_asked", ""))
        answer = st.text_area("Your answer", height=190, key="interview_answer",
                              placeholder="Answer as if you were speaking to an interviewer...")
        if st.button("Evaluate my answer", type="primary"):
            if not answer.strip():
                st.warning("Write your answer first.")
            else:
                try:
                    with st.spinner("Scoring your answer against the rubric..."):
                        feedback = ask_ai_json(f"""
Evaluate the candidate's answer to this question for role {role}.
Question: {q.get('question')}
Answer: {answer}
Rubric dimensions: Relevance, Technical accuracy, Structure and clarity, Evidence/examples.
Return JSON:
{{"scores":{{"relevance":1,"technical_accuracy":1,"clarity":1,"evidence_examples":1}},
"overall_score":1.0,"strengths":["..."],"improvements":["..."],
"sample_better_answer":"...","fairness_note":"..."}}
Each dimension must be an integer 1-5 with a short rationale in a separate
"rationales" object. Score only what the answer demonstrates. Do not reward
fabricated experience; this is coaching feedback, not an official hiring score.
""", max_tokens=2500)
                    score = float(feedback.get("overall_score", 0))
                    execute("""
                    INSERT INTO interview_sessions(role,question,answer,score,feedback,created_at)
                    VALUES(?,?,?,?,?,?)
                    """, (role, q.get("question", ""), answer, score,
                          json.dumps(feedback, ensure_ascii=False),
                          datetime.now().isoformat(timespec="seconds")))
                    st.session_state["last_interview_feedback"] = feedback
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    feedback = st.session_state.get("last_interview_feedback")
    if feedback:
        st.divider()
        st.subheader("Your rubric score")
        st.metric("Overall coaching score", f"{feedback.get('overall_score', 0)}/5")
        scores = feedback.get("scores", {})
        rationales = feedback.get("rationales", {})
        score_df = pd.DataFrame([
            {"Dimension": k.replace("_", " ").title(), "Score": v,
             "Rationale": rationales.get(k, "")}
            for k, v in scores.items()
        ])
        if not score_df.empty:
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Strengths**")
            for x in safe_list(feedback.get("strengths")):
                st.markdown(f"- {x}")
        with c2:
            st.markdown("**Improvements**")
            for x in safe_list(feedback.get("improvements")):
                st.markdown(f"- {x}")
        st.markdown("**Example of a stronger answer**")
        st.write(feedback.get("sample_better_answer", ""))
        st.caption(feedback.get("fairness_note", ""))

    history = query("SELECT role, score, created_at FROM interview_sessions ORDER BY id DESC LIMIT 10")
    if history:
        st.divider()
        st.subheader("Recent practice history")
        st.dataframe(pd.DataFrame([dict(x) for x in history]),
                     use_container_width=True, hide_index=True)


# ---------------------------- Progress tracker ---------------
elif page == "Progress Tracker":
    st.title("📈 Progress Tracker")
    st.write("Track learning completion and interview practice. Use the adaptive roadmap action to respond to your progress.")
    df = task_dataframe()
    if not df.empty:
        total = len(df)
        done = int(df["completed"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Tasks completed", done)
        c2.metric("Tasks remaining", total - done)
        c3.metric("Completion rate", f"{round(done/total*100)}%")
        chart_df = df.groupby("week", as_index=False).agg(
            Total=("id", "count"), Completed=("completed", "sum")
        )
        chart_df["Remaining"] = chart_df["Total"] - chart_df["Completed"]
        st.plotly_chart(px.bar(chart_df, x="week", y=["Completed", "Remaining"],
                               barmode="stack", title="Weekly learning progress",
                               labels={"week": "Week", "value": "Tasks"}),
                        use_container_width=True)
        st.dataframe(df[["week", "skill", "task", "completed"]].rename(
            columns={"week": "Week", "skill": "Skill", "task": "Task",
                     "completed": "Completed"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("Your progress dashboard appears after a roadmap is generated.")
    interview_data = query("SELECT score, created_at FROM interview_sessions ORDER BY id")
    if interview_data:
        idf = pd.DataFrame([dict(x) for x in interview_data])
        st.subheader("Interview score trend")
        st.line_chart(idf.set_index("created_at")["score"])
    st.info("Progress adjustment is available in the Learning Roadmap page. "
            "It uses your task completion data to revise priorities; it does not claim skill mastery.")


# ---------------------------- Application tracker ------------
elif page == "Application Tracker":
    st.title("📌 Job Application Tracker")
    st.write("Track outcomes and learn which role categories are getting responses.")
    with st.form("add_application", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input("Company name *")
            title = st.text_input("Job title *")
            category = st.text_input("Role category", placeholder="e.g. Data Analyst, ML Engineer")
            source = st.selectbox("Application source",
                                  ["Company website", "LinkedIn", "Referral", "Job board", "Recruiter", "Other"])
        with c2:
            url = st.text_input("Job posting URL (optional)")
            applied = st.date_input("Applied date", value=date.today())
            status = st.selectbox("Current status",
                                  ["Planned", "Applied", "Assessment", "Interview", "Offer", "Rejected", "No response"])
            has_response_date = st.checkbox("I have a response date")
            response_date = st.date_input("Response date", value=date.today(),
                                          disabled=not has_response_date)
        notes = st.text_area("Notes (optional)", height=80)
        submitted = st.form_submit_button("➕ Add application", type="primary")
        if submitted:
            if not company.strip() or not title.strip():
                st.error("Company and job title are required.")
            else:
                execute("""
                INSERT INTO applications(company,job_title,role_category,source,job_url,
                    applied_date,status,response_date,notes,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """, (company.strip(), title.strip(), category.strip() or "Unspecified",
                      source, url.strip(), str(applied), status,
                      str(response_date) if has_response_date else "", notes.strip(),
                      datetime.now().isoformat(timespec="seconds")))
                st.success("Application saved.")
                st.rerun()

    df = app_dataframe()
    if not df.empty:
        st.divider()
        st.subheader("Application history")
        display_cols = ["id", "company", "job_title", "role_category", "applied_date",
                        "status", "response_date", "source", "job_url", "notes"]
        edited = st.data_editor(
            df[display_cols], use_container_width=True, hide_index=True,
            disabled=["id", "company", "job_title", "role_category", "applied_date",
                      "source", "job_url", "notes"],
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["Planned", "Applied", "Assessment", "Interview",
                                       "Offer", "Rejected", "No response"]
                )
            },
            key="application_editor"
        )
        if st.button("Save status updates"):
            for _, row in edited.iterrows():
                execute("UPDATE applications SET status=? WHERE id=?",
                        (row["status"], int(row["id"])))
            st.success("Statuses updated.")
            st.rerun()

        st.download_button("⬇️ Export applications CSV",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="career_applications.csv", mime="text/csv")

        st.divider()
        st.subheader("📊 What kinds of roles get responses?")
        df["responded"] = df["status"].isin(["Assessment", "Interview", "Offer", "Rejected"])
        df["positive_progress"] = df["status"].isin(["Assessment", "Interview", "Offer"])
        stats = df.groupby("role_category", dropna=False).agg(
            Applications=("id", "count"),
            Responses=("responded", "sum"),
            Assessments_or_better=("positive_progress", "sum"),
            Offers=("status", lambda x: (x == "Offer").sum())
        ).reset_index()
        stats["Response rate"] = (stats["Responses"] / stats["Applications"] * 100).round(1)
        stats["Progress rate"] = (stats["Assessments_or_better"] / stats["Applications"] * 100).round(1)
        st.dataframe(stats, use_container_width=True, hide_index=True)
        if len(stats):
            st.plotly_chart(px.bar(stats, x="role_category", y="Response rate",
                                   title="Observed response rate by role category",
                                   labels={"role_category": "Role category", "Response rate": "Response rate (%)"}),
                            use_container_width=True)
        st.caption("Method: a response is recorded when status is Assessment, Interview, Offer or Rejected. "
                   "'No response' is not counted as a response. Small samples are directional only.")
        if len(df) >= 3:
            try:
                advice = ask_ai(f"""
Analyze these application outcomes and suggest cautious next experiments.
Data by role category:
{stats.to_json(orient='records')}
Individual application statuses:
{df[['job_title','role_category','source','status']].to_json(orient='records')}
Identify categories with comparatively stronger observed response rates,
but explicitly mention sample sizes and avoid causal claims. Give 3 practical
experiments for the next 10 applications. Do not claim statistical certainty.
""", max_tokens=1200)
                with st.expander("AI application strategy insights", expanded=True):
                    st.markdown(advice)
            except Exception as exc:
                st.caption(f"AI insights unavailable: {exc}")
    else:
        st.info("No applications yet. Add your first application to start learning from outcomes.")


# ---------------------------- Footer -------------------------
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit, Groq and SQLite.")
st.sidebar.caption("AI suggestions are decision support, not a hiring guarantee.")
