import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


import os
if not os.path.exists("./artifacts/rf_model.pkl"):
    from train_model import train_and_save_model
    train_and_save_model()


rf_model = joblib.load("artifacts/rf_model.pkl")

# === Dash App ===
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🎓 Student Performance Predictor", style={'textAlign': 'center', 'color': '#333'}),

    html.Div([
        html.Label("Age (between 15 to 18 years):", style={'marginRight': '10px'}),
        dcc.Input(id="age", type="number", min=15, max=18, placeholder="e.g. 17", value=15, style={'width': '100%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Gender:", style={'marginRight': '10px'}),
        # dcc.Input(id="input-study-time", type="number", min=0, placeholder="e.g. 10", style={'width': '100%'}),
        dcc.Dropdown(
            options=[
                {'label': 'Male', 'value': 0},
                {'label': 'Female', 'value': 1}
            ],
            value=0,
            id='gender', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Ethnicity:", style={'marginRight': '10px'}),
        # dcc.Input(id="input-study-time", type="number", min=0, placeholder="e.g. 10", style={'width': '100%'})
        dcc.Dropdown(
            options=[
                {'label': 'Caucasian', 'value': 0},
                {'label': 'African American', 'value': 1},
                {'label': 'Asian', 'value': 2},
                {'label': 'Other', 'value': 3}
            ],
            value=0,
            id='ethnicity', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Parental Education:", style={'marginRight': '10px'}),
        # dcc.Input(id="input-study-time", type="number", min=0, placeholder="e.g. 10", style={'width': '100%'})
        dcc.Dropdown(
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'High School', 'value': 1},
                {'label': 'Some College', 'value': 2},
                {'label': 'Bachelor\'s', 'value': 3},
                {'label': 'Higher Study', 'value': 4}
            ],
            value=0,
            id='parent_education', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Study Time (hrs/week):", style={'marginRight': '10px'}),
        dcc.Input(id="study_time", type="number", min=0, max=20, placeholder="e.g. 10", value=0, style={'width': '100%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Absences:", style={'marginRight': '10px'}),
        dcc.Input(id="absences", type="number", min=0, max=30, placeholder="e.g. 10", value=0, style={'width': '100%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Tutoring:", style={'marginRight': '10px'}),
        # dcc.Input(id="tutoring", type="number", min=0, max=30,  placeholder="e.g. 10", style={'width': '100%'})
        dcc.Dropdown(
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1}
            ],
            value=0,
            id='tutoring', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),

    html.Div([
        html.Label("Parental Support (1–5):", style={'marginRight': '10px'}),
        # dcc.Input(id="input-parental-support", type="number", min=1, max=5, placeholder="e.g. 3", style={'width': '100%'})
        dcc.Dropdown(
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'Low', 'value': 1},
                {'label': 'Moderate', 'value': 2},
                {'label': 'High', 'value': 3},
                {'label': 'Very High', 'value': 4}
            ],
            value=0,
            id='parent_support', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Extracurricular:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1},
            ],
            value=0,
            id='extra_curr', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Sports:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1},
            ],
            value=0,
            id='sports', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Music:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1},
            ],
            value=0,
            id='music', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("Volunteering:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1},
            ],
            value=0,
            id='volunteering', 
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Label("GPA:", style={'marginRight': '10px'}),
        dcc.Input(id="gpa", type="number", min=2, max=4, step=0.1, placeholder="e.g. 2.5", value=2.0, style={'width': '100%'})
    ], style={'marginBottom': '15px'}),

    html.Div([
        html.Button("🔍 Predict", id="predict-button", n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px'})
    ], style={'textAlign': 'center', 'marginTop': '10px'}),

    html.Div(id="output-prediction", style={'marginTop': '20px', 'fontSize': '20px', 'textAlign': 'center', 'color': '#0074D9'})
], style={'maxWidth': '400px', 'margin': 'auto', 'padding': '30px', 'fontFamily': 'Arial, sans-serif'})

# === Callback for prediction ===
@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("ethnicity", "value"),
    State("parent_education", "value"),
    State("study_time", "value"),
    State("absences", "value"),
    State("tutoring", "value"),
    State("parent_support", "value"),
    State("extra_curr", "value"),
    State("sports", "value"),
    State("music", "value"),
    State("volunteering", "value"),
    State("gpa", "value"),
)
def predict_grade(n_clicks, age, gender, ethnicity, parent_education, study_time, absences, tutoring, parent_support, extra_curr, sports, music, volunteering, gpa):
    if n_clicks == 0:
        return ""

    new_student = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'ParentalEducation': parent_education,
        'StudyTimeWeekly': study_time,
        'Absences': absences,
        'Tutoring': tutoring,
        'ParentalSupport': parent_support,
        'Extracurricular': extra_curr,
        'Sports': sports,
        'Music': music,
        'Volunteering': volunteering,
        'GPA': gpa
    }])

    prediction = int(rf_model.predict(new_student)[0])
    results = 'A'
    if prediction == 1: results = 'B'
    elif prediction == 2: results = 'C'
    elif prediction == 3: results = 'D'
    elif prediction == 4: results = 'F'

    return f"🎯 Predicted Grade Class: {results}"

# === Run App ===
if __name__ == '__main__':
    app.run_server(debug=True)


