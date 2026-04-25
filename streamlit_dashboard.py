import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="NJFP Moodle Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================

# =========================
# LOAD DATA WITH CACHE/FALLBACK
# =========================

DATA_PATH = "data/final_results.csv"
BACKUP_PATH = "data/final_results_backup.csv"

@st.cache_data(ttl=600)
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)

        # save backup after successful load
        df.to_csv(BACKUP_PATH, index=False)

        return df, "latest"

    except Exception as e:
        try:
            df = pd.read_csv(BACKUP_PATH)
            return df, "backup"

        except Exception:
            return pd.DataFrame(), "failed"


raw_df, data_status = load_data()

if data_status == "backup":
    st.warning("⚠️ Latest data failed to load. Showing backup cached data.")

elif data_status == "failed":
    st.error("❌ No data available. Latest and backup files failed.")
    st.stop()



REQUIRED_COLUMNS = [
    "course_id", "course_name", "quiz_id", "quiz_name", "user_id",
    "fullname", "email", "username", "suspended", "best_grade",
    "grade_max", "grade_error", "grade_percent", "attempted", "status"
]




def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in working.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    working["best_grade"] = pd.to_numeric(working["best_grade"], errors="coerce")
    working["grade_max"] = pd.to_numeric(working["grade_max"], errors="coerce")
    working["grade_percent"] = pd.to_numeric(working["grade_percent"], errors="coerce")

    working["attempted"] = (
        working["attempted"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    working["status"] = working["status"].astype(str).str.strip().str.title()
    working["fullname"] = working["fullname"].astype(str).str.strip().str.title()
    working["course_name"] = working["course_name"].astype(str).str.strip().str.title()
    working["quiz_name"] = working["quiz_name"].astype(str).str.strip().str.title()
    working["email"] = working["email"].astype(str).str.strip().str.lower()

    return working


def build_learner_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("user_id")
        .agg(
            fullname=("fullname", "first"),
            email=("email", "first"),
            course_name=("course_name", "first"),
            quiz_name=("quiz_name", "first"),
            attempted=("attempted", "max"),
            best_score=("grade_percent", "max"),
            attempt_count=("attempted", "sum"),
        )
        .reset_index()
    )

    summary["status"] = np.select(
        [
            ~summary["attempted"],
            summary["best_score"] >= 50,
            summary["best_score"] < 50,
        ],
        ["Not Attempted", "Passed", "Failed"],
        default="Unknown",
    )

    return summary


def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)



df = prepare_data(raw_df)

st.title("NJFP Learning Performance & Assessment Dashboard")

with st.sidebar:
    st.header("Filters")

    course_options = ["All"] + sorted(df["course_name"].dropna().unique())
    selected_course = st.selectbox("Course", course_options)

    if selected_course != "All":
        quiz_options = ["All"] + sorted(
            df.loc[df["course_name"] == selected_course, "quiz_name"]
            .dropna()
            .unique()
        )
    else:
        quiz_options = ["All"] + sorted(df["quiz_name"].dropna().unique())

    selected_quiz = st.selectbox("Quiz", quiz_options)
    selected_status = st.selectbox(
        "Status",
        ["All", "Passed", "Failed", "Not Attempted"]
    )

filtered_df = df.copy()

if selected_course != "All":
    filtered_df = filtered_df[filtered_df["course_name"] == selected_course]

if selected_quiz != "All":
    filtered_df = filtered_df[filtered_df["quiz_name"] == selected_quiz]

if selected_status != "All":
    filtered_df = filtered_df[filtered_df["status"] == selected_status]

learner_summary = build_learner_summary(filtered_df)
attempted_summary = learner_summary[learner_summary["attempted"]].copy()

total_courses = filtered_df["course_id"].nunique()
total_quizzes = filtered_df["quiz_id"].nunique()
total_learners = filtered_df["user_id"].nunique()
learners_attempted = int(learner_summary["attempted"].sum())
learners_not_attempted = int((~learner_summary["attempted"]).sum())

avg_score = (
    round(attempted_summary["best_score"].mean(), 2)
    if not attempted_summary.empty
    else 0.0
)



pass_rate = round(
    (learner_summary["status"].eq("Passed").sum() / max(learners_attempted, 1)) * 100,
    2
)
avg_score_display = f"{avg_score:.2f}%" if avg_score is not None else "N/A"

completion_rate = round(
    (learners_attempted / max(total_learners, 1)) * 100,
    2
)

metric_row([
    ("Total Courses", total_courses),
    ("Attempted Quizzes", total_quizzes),
    ("Total Learners", total_learners),
    ("Attempted", learners_attempted),
])

metric_row([
    ("Not Attempted", learners_not_attempted),
    ("Average Score (%)", avg_score_display),
    ("Pass Rate (%)", pass_rate),
    ("Completion Rate (%)", completion_rate),
])

left, right = st.columns(2)

with left:
    status_dist = learner_summary["status"].value_counts().reset_index()
    status_dist.columns = ["status", "count"]

    fig_status = px.pie(
        status_dist,
        names="status",
        values="count",
        title="Learner Attempt & Outcome Status",
    )
    st.plotly_chart(fig_status, use_container_width=True)

with right:
    if not attempted_summary.empty:
        fig_grade = px.histogram(
            attempted_summary,
            x="best_score",
            nbins=10,
            title="Learner Score Distribution (%)",
        )
        st.plotly_chart(fig_grade, use_container_width=True)
    else:
        st.info("No attempted records available.")

st.subheader("Score Band Distribution")

if not attempted_summary.empty:
    grade_dist = pd.cut(
        attempted_summary["best_score"],
        bins=[0, 20, 40, 60, 80, 100],
        include_lowest=True,
    ).astype(str).value_counts().sort_index().reset_index()

    grade_dist.columns = ["Range", "Count"]

    fig_band = px.bar(
        grade_dist,
        x="Range",
        y="Count",
        text="Count",
        title="Learner Score Bands",
    )
    st.plotly_chart(fig_band, use_container_width=True)


st.subheader("Top Performing Learners")

top_performers = (
    learner_summary[learner_summary["attempted"]]
    .sort_values(["best_score", "attempt_count"], ascending=[False, False])
    [["course_name", "quiz_name", "fullname", "email", "best_score", "attempt_count"]]
    .head(10)
    .rename(columns={
        "course_name": "Course",
        "quiz_name": "Quiz",
        "fullname": "Learner Name",
        "email": "Email",
        "best_score": "Score (%)",
        "attempt_count": "Attempts",
    })
)

st.dataframe(top_performers, use_container_width=True, hide_index=True)



st.subheader("Learner Final View")

learner_final_view = (
    learner_summary[
        ["course_name", "quiz_name", "fullname", "email", "best_score", "status"]
    ]
    .sort_values(["course_name", "quiz_name", "fullname"])
    .rename(columns={
        "course_name": "Course",
        "quiz_name": "Quiz",
        "fullname": "Learner Name",
        "email": "Email",
        "best_score": "Score (%)",
        "status": "Status",
    })
)

st.dataframe(learner_final_view, use_container_width=True, hide_index=True)

csv_download = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered raw data as CSV",
    data=csv_download,
    file_name="filtered_moodle_dashboard_data.csv",
    mime="text/csv",
)


st.markdown("## 📊 KPI Definitions & Methodology")

with st.expander("View how KPIs are calculated"):
    st.markdown("""
### 🔢 KPI Definitions

**1. Total Courses**
- Represents the number of unique courses included in the analysis  
- **Formula:** `COUNT(DISTINCT course_id)`  
- This reflects the scope of learning programs being tracked  

---

**2. Attempted Quizzes (Active Quizzes)**
- Number of quizzes that have at least one learner record in the dataset  
- This reflects quizzes that are actively being engaged with  
- **Formula:** `COUNT(DISTINCT quiz_id)`  

---

**3. Total Learners**
- Total number of unique learners enrolled in the selected course(s)  
- Each learner is counted **once**, regardless of how many quizzes they attempted  
- **Important Logic:**  
  - If a learner attempts multiple quizzes, only **their highest-scoring quiz** is retained  
  - This ensures each learner contributes only one record to performance metrics  
- **Formula:** `COUNT(DISTINCT user_id)`  

---

**4. Attempted**
- Number of learners who attempted at least one quiz  
- A learner is considered to have attempted if they have a valid score (including 0)  
- **Formula:** `COUNT(user_id WHERE attempted = True)`  

---

**5. Not Attempted**
- Number of learners who have not attempted any quiz  
- These learners are included in total population but excluded from performance metrics  
- **Formula:** `COUNT(user_id WHERE attempted = False)`  

---

**6. Average Score (%)**
- The average performance score of learners who attempted quizzes  
- Only learners with valid scores are included  
- Learners who did not attempt are excluded  
- **Formula:**  
  `AVG(best_score WHERE attempted = True)`  

---

**7. Pass Rate (%)**
- Percentage of learners who passed among those who attempted  
- A learner is considered to have passed if their score is **≥ 50%**  
- **Formula:**  
  `(Number of Passed Learners ÷ Number of Attempted Learners) × 100`  

---

**8. Completion Rate (%)**
- Measures learner engagement by showing the proportion of learners who attempted at least one quiz  
- **Formula:**  
  `(Attempted Learners ÷ Total Learners) × 100`  

---

### ⚙️ Data Processing Logic (Very Important)

- Each learner may have access to multiple quizzes within a course  
- However, for consistency and fairness:
  
  👉 **Only one quiz record is kept per learner**  
  👉 Specifically, the quiz with the **highest score**  

- This approach ensures:
  - No duplication of learners in metrics  
  - No inflation of performance statistics  
  - A fair representation of each learner’s best outcome  

---

### 📊 Score Normalization

- All scores are converted into percentages using:  
  `(best_grade ÷ grade_max) × 100`

- This allows comparison across quizzes with different scoring scales  

---

### ⚠️ Important Notes

- A score of **0% is considered an attempt**, not “Not Attempted”  
- Learners without any score are classified as **Not Attempted**  
- Performance metrics (Average Score, Pass Rate) exclude non-attempted learners  
- Dashboard values may differ from raw LMS totals due to:
  - Deduplication (one quiz per learner rule)  
  - Filtering applied in the dashboard  

---

### 📌 Data Source & Reliability

- Data is fetched directly from the LMS via API  
- The dataset is updated automatically through a scheduled pipeline  
- A backup dataset is used if the latest data retrieval fails  
- This ensures continuous availability and reliability of insights  

---

### 🧠 Interpretation Guide (For Stakeholders)

- **High Completion Rate + High Pass Rate** → Strong engagement and understanding  
- **High Completion Rate + Low Pass Rate** → Learners are engaged but struggling  
- **Low Completion Rate** → Engagement issue (learners not attempting quizzes)  
- **High Average Score** → Overall strong performance among active learners  

---
""")