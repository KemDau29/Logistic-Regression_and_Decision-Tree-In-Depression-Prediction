import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
def load_data(path="data/student_depression_dataset.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    encoders = {}

    # 🔹 Mapping logic cho các cột cụ thể
    mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Sleep duration": {
            "Less than 5 hours": 0,
            "5-6 hours": 1,
            "7-8 hours": 2,
            "More than 8 hours": 3,
            "Others": 4
        },
        "Degree": {
            "'Class 12'": 0, "BA": 0, "B.Com": 0, "BBA": 0, "BHM": 0, "B.Ed": 0,
            "B.Tech": 1, "BE": 1, "BCA": 1, "BSc": 1, "B.Pharm": 1, "B.Arch": 1, "LLB": 1, "ME": 1,
            "M.Tech": 2, "MBA": 2, "MCA": 2, "MSc": 2, "M.Ed": 2, "M.Com": 2, "MHM": 2,
            "M.Pharm": 2, "MA": 2, "LLM": 2, "MD": 2, "MBBS": 2, "PhD": 2,
            "Others": 1
        }
    }

    # 🔹 Áp dụng mapping logic
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            encoders[col] = mapping

    # 🔹 Với các cột còn lại kiểu object, dùng LabelEncoder
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoders[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    # 🔹 Loại bỏ các cột không cần thiết
    drop_cols = ["Depression", "id", "Profession", "City", "Work Pressure", "Job Satisfaction"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Depression"]

    # 🔹 Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoders
