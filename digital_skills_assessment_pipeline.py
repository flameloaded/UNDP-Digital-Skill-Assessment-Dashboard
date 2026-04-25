import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("MOODLE_BASE_URL")
TOKEN = os.getenv("MOODLE_TOKEN")

TIMEOUT = 60
REQUEST_DELAY = 0.08

OUTPUT_FILE = "data/final_results.csv"
BACKUP_FILE = "data/final_results_backup.csv"
TEMP_FILE = "data/temp_final_results.csv"

INCLUDE_COURSE_IDS = [28]  # change this to the course you want


def call_moodle(wsfunction, **kwargs):
    params = {
        "wstoken": TOKEN,
        "moodlewsrestformat": "json",
        "wsfunction": wsfunction,
    }
    params.update(kwargs)

    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and data.get("exception"):
        raise Exception(
            f"{wsfunction} failed: {data.get('message')} | {data.get('debuginfo', '')}"
        )

    time.sleep(REQUEST_DELAY)
    return data


def safe_call(wsfunction, **kwargs):
    try:
        return call_moodle(wsfunction, **kwargs), None
    except Exception as e:
        return None, str(e)


def fetch_courses():
    courses = call_moodle("core_course_get_courses")
    courses_df = pd.DataFrame(courses)

    courses_df = courses_df.rename(columns={
        "id": "course_id",
        "fullname": "course_name",
        "shortname": "course_shortname"
    })

    courses_df = courses_df[courses_df["course_id"].isin(INCLUDE_COURSE_IDS)].copy()

    return courses_df


def fetch_users(courses_df):
    course_users = []

    for _, row in courses_df.iterrows():
        course_id = int(row["course_id"])
        course_name = row["course_name"]

        users_data, err = safe_call(
            "core_enrol_get_enrolled_users",
            courseid=course_id
        )

        if err:
            print(f"Could not fetch users for course {course_id}: {err}", flush=True)
            continue

        if not isinstance(users_data, list):
            continue

        for user in users_data:
            course_users.append({
                "course_id": course_id,
                "course_name": course_name,
                "user_id": user.get("id"),
                "fullname": user.get("fullname"),
                "email": user.get("email"),
                "username": user.get("username"),
                "suspended": user.get("suspended"),
            })

    return pd.DataFrame(course_users).drop_duplicates(
        subset=["course_id", "user_id"]
    ).copy()


def fetch_quizzes(courses_df):
    course_ids = courses_df["course_id"].dropna().astype(int).tolist()

    params = {
        "wstoken": TOKEN,
        "moodlewsrestformat": "json",
        "wsfunction": "mod_quiz_get_quizzes_by_courses"
    }

    for i, cid in enumerate(course_ids):
        params[f"courseids[{i}]"] = int(cid)

    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()

    quiz_payload = response.json()

    if isinstance(quiz_payload, dict) and quiz_payload.get("exception"):
        raise Exception(
            f"mod_quiz_get_quizzes_by_courses failed: "
            f"{quiz_payload.get('message')} | {quiz_payload.get('debuginfo', '')}"
        )

    quizzes = quiz_payload.get("quizzes", [])

    if not quizzes:
        return pd.DataFrame(columns=[
            "course_id", "quiz_id", "quiz_name", "grade_max"
        ])

    quizzes_df = pd.DataFrame([{
        "course_id": q.get("course"),
        "quiz_id": q.get("id"),
        "quiz_name": q.get("name"),
        "grade_max": q.get("grademax"),
    } for q in quizzes])
    print("Total unique quizzes:", quizzes_df["quiz_id"].nunique())
    print("Quiz IDs:", quizzes_df["quiz_id"].unique())
    return quizzes_df


def fetch_quiz_grades(quizzes_df, users_df):
    users_by_course = {
        course_id: group.reset_index(drop=True)
        for course_id, group in users_df.groupby("course_id")
    }

    records = []

    for _, quiz_row in quizzes_df.iterrows():
        course_id = int(quiz_row["course_id"])
        quiz_id = int(quiz_row["quiz_id"])
        quiz_name = quiz_row["quiz_name"]
        grade_max = quiz_row.get("grade_max")

        enrolled_users = users_by_course.get(course_id)

        if enrolled_users is None or enrolled_users.empty:
            print(f"No users for course_id={course_id}", flush=True)
            continue

        print(f"Processing quiz: {quiz_name} | users: {len(enrolled_users)}", flush=True)

        for i, (_, user_row) in enumerate(enrolled_users.iterrows(), start=1):
            user_id = int(user_row["user_id"])

            grade_data, grade_err = safe_call(
                "mod_quiz_get_user_best_grade",
                quizid=quiz_id,
                userid=user_id
            )

            best_grade = grade_data.get("grade") if isinstance(grade_data, dict) else None

            records.append({
                "course_id": course_id,
                "course_name": user_row.get("course_name"),
                "quiz_id": quiz_id,
                "quiz_name": quiz_name,
                "user_id": user_id,
                "fullname": user_row.get("fullname"),
                "email": user_row.get("email"),
                "username": user_row.get("username"),
                "suspended": user_row.get("suspended"),
                "best_grade": best_grade,
                "grade_max": 10,
                "grade_error": grade_err,
            })

            if i % 50 == 0:
                print(f"  done {i}/{len(enrolled_users)} users", flush=True)

    return pd.DataFrame(records)


def clean_and_select_best_quiz(quiz_df):
    if quiz_df.empty:
        return quiz_df

    quiz_df = quiz_df.copy()

    text_cols = quiz_df.select_dtypes(include=["object", "string"]).columns
    quiz_df[text_cols] = quiz_df[text_cols].apply(lambda col: col.str.strip())

    quiz_df["fullname"] = quiz_df["fullname"].str.title()
    quiz_df["email"] = quiz_df["email"].str.lower()
    quiz_df["username"] = quiz_df["username"].str.lower()
    quiz_df["course_name"] = quiz_df["course_name"].str.title()
    quiz_df["quiz_name"] = quiz_df["quiz_name"].str.title()

    quiz_df["best_grade"] = pd.to_numeric(quiz_df["best_grade"], errors="coerce")
    quiz_df["grade_max"] = pd.to_numeric(quiz_df["grade_max"], errors="coerce")

    quiz_df["grade_percent"] = np.where(
        quiz_df["best_grade"].notna() & quiz_df["grade_max"].gt(0),
        (quiz_df["best_grade"] / quiz_df["grade_max"]) * 100,
        np.nan
    )

    quiz_df["attempted"] = quiz_df["best_grade"].notna()

    quiz_df["status"] = np.select(
        [
            quiz_df["grade_percent"].isna(),
            quiz_df["grade_percent"] >= 50,
            quiz_df["grade_percent"] < 50,
        ],
        ["Not Attempted", "Passed", "Failed"],
        default="Unknown"
    )

    # Important rule:
    # If user answered more than one quiz in same course,
    # keep only the highest score.
    quiz_df = (
        quiz_df
        .sort_values("grade_percent", ascending=False)
        .drop_duplicates(subset=["course_id", "user_id"], keep="first")
        .sort_values(["course_name", "fullname"])
        .reset_index(drop=True)
    )

    return quiz_df


def build_dataset():
    print("Fetching courses...", flush=True)
    courses_df = fetch_courses()

    print("Fetching users...", flush=True)
    users_df = fetch_users(courses_df)

    print("Fetching quizzes...", flush=True)
    quizzes_df = fetch_quizzes(courses_df)

    print("Fetching quiz grades...", flush=True)
    quiz_df = fetch_quiz_grades(quizzes_df, users_df)

    print("Cleaning and selecting best quiz per user...", flush=True)
    final_df = clean_and_select_best_quiz(quiz_df)

    return final_df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    try:
        df = build_dataset()

        if df.empty:
            raise ValueError("New dataset is empty. Aborting update.")

        if os.path.exists(OUTPUT_FILE):
            old_df = pd.read_csv(OUTPUT_FILE)
            old_df.to_csv(BACKUP_FILE, index=False)

        df.to_csv(TEMP_FILE, index=False, encoding="utf-8", na_rep="")
        os.replace(TEMP_FILE, OUTPUT_FILE)

        print("✅ Data updated successfully", flush=True)
        print(f"Rows saved: {len(df)}", flush=True)

    except Exception as e:
        print("❌ Pipeline failed:", e, flush=True)
        print("⚠️ Old dataset was not replaced.", flush=True)