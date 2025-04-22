# 🎓 Student Performance Predictor - Dash App


## Group F:
Anele Nkayi - 577168
Rourke Veller - 601052
Kealeboga Molefe - 577482
Willem Booysen - 600613


## 📌 Project Overview

This project presents a machine learning-powered **Student Performance Predictor** designed for **BrightPath Academy**, a progressive high school focused on both academic excellence and holistic development. The goal is to assist educators in identifying at-risk students and personalizing academic interventions based on multiple student attributes, including demographics, study habits, parental support, and extracurricular involvement.

This solution is deployed as a user-friendly Dash web app that predicts a student's **Grade Class** (A–F) based on their profile.

---

## 🎯 Problem Statement

BrightPath Academy is facing:
- Delayed identification of students who may be underperforming.
- Lack of personalized academic support tools.
- Difficulty measuring the impact of extracurricular activities on performance.
- An overload of data with no centralized predictive mechanism.

This app addresses these by leveraging historical student data to **predict academic grade classes using machine learning**.

---

## 🧪 Hypothesis

We hypothesize that student academic performance (Grade Class) is influenced by:
- Demographics (age, gender, ethnicity, parental education)
- Study habits (weekly study time, absences)
- Parental involvement
- Participation in extracurricular activities (sports, music, volunteering)
- GPA

---

## 🗃️ Dataset

We used the `Student_performance_data.csv` dataset, which includes:
- **Inputs:** Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GPA
- **Target:** `GradeClass` — a classification of grades:
  - 0: A (GPA ≥ 3.5)
  - 1: B (3.0 ≤ GPA < 3.5)
  - 2: C (2.5 ≤ GPA < 3.0)
  - 3: D (2.0 ≤ GPA < 2.5)
  - 4: F (GPA < 2.0)

---

## ⚙️ Features

- Interactive Dash form to input new student data.
- Real-time prediction of Grade Class using a trained **Random Forest Classifier**.
- Clean, responsive layout with labels, dropdowns, and tooltips.

---

## 🤖 Model Details

- Model Type: Random Forest Classifier
- Trained with: scikit-learn
- Evaluation metrics used: Accuracy, Precision, Recall, F1-score
- Model stored as: `./artifacts/rf_model.pkl`

---

## 🚀 How to Run the App

### 🖥️ Requirements

Install dependencies:

```bash
pip install dash pandas scikit-learn joblib