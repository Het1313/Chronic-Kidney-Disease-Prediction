# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# app = Flask(__name__)

# # Step 1: Load the dataset and preprocess
# url = "D://classProjects//csv_result-chronic_kidney_disease.csv"
# df = pd.read_csv(url, on_bad_lines='skip')

# df.replace('?', np.nan, inplace=True)
# df.columns = [
#     "id", "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
#     "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "class"
# ]
# df = df.dropna()
# df["class"] = df["class"].map({"ckd": 1, "notckd": 0})
# df["htn"] = df["htn"].map({"yes": 1, "no": 0})
# df["dm"] = df["dm"].map({"yes": 1, "no": 0})
# df["appet"] = df["appet"].map({"good": 1, "poor": 0})
# df = df.apply(pd.to_numeric, errors='coerce')

# X = df[["age", "bp", "bgr", "bu", "sc", "hemo", "htn", "dm", "appet"]]
# y = df["class"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# accuracy = accuracy_score(y_test, model.predict(X_test))
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# with open("chronic_kidney_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         required_fields = ['age', 'bp', 'bgr', 'bu', 'sc', 'hemo', 'htn', 'dm', 'appet']

#         # Check if all required fields are present
#         if not all(field in data for field in required_fields):
#             return jsonify({"error": "Missing required data fields"}), 400

#         features = np.array([
#             data['age'], data['bp'], data['bgr'], data['bu'],
#             data['sc'], data['hemo'], data['htn'], data['dm'], data['appet']
#         ]).reshape(1, -1)

#         with open("chronic_kidney_model.pkl", "rb") as f:
#             loaded_model = pickle.load(f)

#         prediction = loaded_model.predict(features)[0]
#         confidence = max(loaded_model.predict_proba(features)[0])

#         # Modify risk level based on the diagnosis
#         if prediction == 0:  # Healthy
#             risk_level = "No Risk"
#         else:  # Chronic Kidney Disease
#             risk_level = "High Risk"

#         response = {
#             "diagnosis": "Chronic Kidney Disease" if prediction == 1 else "Healthy",
#             "confidence": round(confidence * 100, 2),  # Convert to percentage
#             "risk_level": risk_level
#         }
#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Step 1: Load the dataset and preprocess
url = "D://classProjects//csv_result-chronic_kidney_disease.csv"
df = pd.read_csv(url, on_bad_lines='skip')

df.replace('?', np.nan, inplace=True)
df.columns = [
    "id", "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "class"
]
df = df.dropna()
df["class"] = df["class"].map({"ckd": 1, "notckd": 0})
df["htn"] = df["htn"].map({"yes": 1, "no": 0})
df["dm"] = df["dm"].map({"yes": 1, "no": 0})
df["appet"] = df["appet"].map({"good": 1, "poor": 0})
df = df.apply(pd.to_numeric, errors='coerce')

X = df[["age", "bp", "bgr", "bu", "sc", "hemo", "htn", "dm", "appet"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open("chronic_kidney_model.pkl", "wb") as f:
    pickle.dump(model, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ['age', 'bp', 'bgr', 'bu', 'sc', 'hemo', 'htn', 'dm', 'appet']

        # Check if all required fields are present
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required data fields"}), 400

        features = np.array([
            data['age'], data['bp'], data['bgr'], data['bu'],
            data['sc'], data['hemo'], data['htn'], data['dm'], data['appet']
        ]).reshape(1, -1)

        with open("chronic_kidney_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        prediction = loaded_model.predict(features)[0]
        confidence = max(loaded_model.predict_proba(features)[0])

        # Modify risk level based on the diagnosis
        if prediction == 0:  # Healthy
            risk_level = "No Risk"
        else:  # Chronic Kidney Disease
            risk_level = "High Risk"

        # Store the input and result in a report (CSV format)
        report_data = {
            'age': data['age'],
            'bp': data['bp'],
            'bgr': data['bgr'],
            'bu': data['bu'],
            'sc': data['sc'],
            'hemo': data['hemo'],
            'htn': data['htn'],
            'dm': data['dm'],
            'appet': data['appet'],
            'diagnosis': "Chronic Kidney Disease" if prediction == 1 else "Healthy",
            'confidence': round(confidence * 100, 2),
            'risk_level': risk_level
        }

        report_df = pd.DataFrame([report_data])
        report_filename = "prediction_report.csv"
        report_df.to_csv(report_filename, index=False)

        response = {
            "diagnosis": "Chronic Kidney Disease" if prediction == 1 else "Healthy",
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_report', methods=['GET'])
def download_report():
    try:
        report_filename = "prediction_report.csv"
        if os.path.exists(report_filename):
            return send_file(report_filename, as_attachment=True)
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
