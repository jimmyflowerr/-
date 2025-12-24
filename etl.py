import re
import pandas as pd
import numpy as np

# =========================
# Config
# =========================
IN_CSV  = r"C:\Users\nba20\Downloads\radar-gesture-recognition-chore-update-20250815\PythonProject1\Sleep_health_and_lifestyle_dataset.csv"
OUT_CSV = "clean_sleep.csv"

RENAME_MAP = {
    "Quality of Sleep": "quality_of_sleep",
    "Sleep Duration": "sleep_duration",
    "Stress Level": "stress_level",
    "Physical Activity Level": "physical_activity_level",
    "Heart Rate": "heart_rate",
    "Daily Steps": "daily_steps",
    "Blood Pressure": "blood_pressure",
    "BMI Category": "bmi_category",
    "Sleep Disorder": "sleep_disorder",
    "Person ID": "person_id",
    "Occupation": "occupation",
    "Gender": "gender",
    "Age": "age",
    "BMI": "bmi",
}

def snake_case(s: str) -> str:
    s = s.strip()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s.lower()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    df.columns = [snake_case(c) for c in df.columns]
    return df

def parse_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """將 blood_pressure 解析成 sbp / dbp（例如 '120/80'）"""
    df = df.copy()
    if "blood_pressure" not in df.columns:
        return df

    bp = df["blood_pressure"].astype("string").str.strip()
    m = bp.str.extract(r"(?P<sbp>\d{2,3})\s*/\s*(?P<dbp>\d{2,3})")

    df["sbp"] = pd.to_numeric(m["sbp"], errors="coerce")
    df["dbp"] = pd.to_numeric(m["dbp"], errors="coerce")

    # 合理範圍（可調）
    df.loc[(df["sbp"] < 70) | (df["sbp"] > 250), "sbp"] = np.nan
    df.loc[(df["dbp"] < 40) | (df["dbp"] > 150), "dbp"] = np.nan
    return df

def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("string").str.strip().str.upper()
        df.loc[~df["gender"].isin(["M", "F", "MALE", "FEMALE"]), "gender"] = pd.NA
        df["gender"] = df["gender"].replace({"MALE": "M", "FEMALE": "F"})

    # bmi_category
    if "bmi_category" in df.columns:
        df["bmi_category"] = df["bmi_category"].astype("string").str.strip().str.title()
        df["bmi_category"] = df["bmi_category"].replace({
            "Normal Weight": "Normal",
            "Over Weight": "Overweight",
        })

    # sleep_disorder（把缺失視為 None）
    if "sleep_disorder" in df.columns:
        df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
        df["sleep_disorder"] = df["sleep_disorder"].astype("string").str.strip().str.title()
        df["sleep_disorder"] = df["sleep_disorder"].replace({
            "No": "None",
            "No Disorder": "None",
            "None": "None",
            "Sleep Apnoea": "Sleep Apnea",
            "Nan": "None",
        })
        df["has_sleep_disorder"] = (df["sleep_disorder"] != "None").astype(int)

    return df  # ✅ 一定要回傳

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 數值欄位轉型
    numeric_cols = [
        "age", "sleep_duration", "quality_of_sleep", "stress_level",
        "physical_activity_level", "heart_rate", "daily_steps", "bmi"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 睡眠時數分組
    if "sleep_duration" in df.columns:
        bins = [-np.inf, 5, 7, 9, np.inf]
        labels = ["<5h", "5-7h", "7-9h", "9h+"]
        df["sleep_duration_group"] = pd.cut(df["sleep_duration"], bins=bins, labels=labels)

    # 睡眠品質分組
    if "quality_of_sleep" in df.columns:
        bins = [-np.inf, 4, 7, np.inf]
        labels = ["Low", "Medium", "High"]
        df["sleep_quality_group"] = pd.cut(df["quality_of_sleep"], bins=bins, labels=labels)

    # 壓力分組
    if "stress_level" in df.columns:
        bins = [-np.inf, 3, 6, np.inf]
        labels = ["Low", "Medium", "High"]
        df["stress_group"] = pd.cut(df["stress_level"], bins=bins, labels=labels)

    return df  # ✅ 這行要在函式裡（有縮排）

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    if "person_id" in df.columns:
        df["person_id"] = df["person_id"].astype("string").str.strip()
        df = df.drop_duplicates(subset=["person_id"])

    # 不合理值修正
    if "age" in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    if "sleep_duration" in df.columns:
        df.loc[(df["sleep_duration"] < 0) | (df["sleep_duration"] > 16), "sleep_duration"] = np.nan
    if "heart_rate" in df.columns:
        df.loc[(df["heart_rate"] < 30) | (df["heart_rate"] > 220), "heart_rate"] = np.nan
    if "daily_steps" in df.columns:
        df.loc[(df["daily_steps"] < 0) | (df["daily_steps"] > 100000), "daily_steps"] = np.nan

    return df

def summarize(df: pd.DataFrame) -> None:
    print("=== ETL Summary ===")
    print("Rows:", len(df))
    print("Cols:", len(df.columns))
    if "has_sleep_disorder" in df.columns:
        rate = df["has_sleep_disorder"].mean()
        print(f"Sleep disorder rate: {rate:.4f} ({rate*100:.2f}%)")
    for c in ["sleep_duration", "quality_of_sleep", "stress_level", "daily_steps", "sbp", "dbp"]:
        if c in df.columns:
            print(f"{c} (min/median/max):", df[c].min(), df[c].median(), df[c].max())

def main():
    df = pd.read_csv(IN_CSV)
    df = standardize_columns(df)
    df = parse_blood_pressure(df)
    df = normalize_categories(df)
    df = add_features(df)
    df = clean_data(df)

    summarize(df)

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
