import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model():
    print("🚀 Training new RandomForest model...")
    
    # Load data (ensure the correct relative path)
    df = pd.read_csv("./data/Student_performance_data.csv")  # ✅ no spaces

    features = [
        'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
        'Sports', 'Music', 'Volunteering', 'GPA'
    ]
    X = df[features]
    y = df['GradeClass']

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "./artifacts/rf_model.pkl")

    print("✅ Model trained and saved to ./artifacts/rf_model.pkl")