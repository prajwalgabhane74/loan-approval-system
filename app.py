import sqlite3
from datetime import datetime

def create_database():
    conn = sqlite3.connect("loan_predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            credit_history REAL,
            property_area TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Load trained model
model = joblib.load("loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"]),
        "Property_Area": request.form["Property_Area"]
    }

    input_data = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    # Find probability of predicted class
    prediction_probability = probabilities[
        list(classes).index(prediction)
    ]

    confidence = round(prediction_probability * 100, 2)

    if prediction == "Y":
        result = "Loan Approved"
    else:
        result = "Loan Not Approved"

    # Save prediction in database
    conn = sqlite3.connect("loan_predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            applicant_income,
            coapplicant_income,
            loan_amount,
            credit_history,
            property_area,
            prediction,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Credit_History"],
        data["Property_Area"],
        result,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        result=result,
        confidence=confidence
    )

@app.route("/history")
def history():

    conn = sqlite3.connect("loan_predictions.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )
create_database()

if __name__ == "__main__":
    app.run(debug=False)