from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)
CORS(app)
# Read data from Excel sheet into a DataFrame
data = pd.read_excel('training_data1.xlsx')

# Extract input features (X) and target labels (y)
X = data.drop(columns=['disease'])
y = data['disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Read feature names from Excel sheet
def read_feature_names_from_excel():
    return data.columns.tolist()[:-1]  # Exclude the last column (label column)

# Endpoint to retrieve feature names
@app.route('/featureNames', methods=['GET'])
def get_feature_names():
    feature_names = read_feature_names_from_excel()
    return jsonify(feature_names)


# Function to predict disease probabilities based on symptoms provided by the user
def predict_disease_probabilities(symptoms):
    # Predict disease probabilities for user-provided symptoms
    disease_probabilities = model.predict_proba([symptoms])

    # Return predicted disease probabilities
    return {disease: probability * 100 for disease, probability in zip(model.classes_, disease_probabilities[0])}

def add_data_to_excel(file_path, symptoms, disease):
    # Read existing data from Excel file
    try:
        existing_data = pd.read_excel(file_path)
        print(existing_data.head())
    except FileNotFoundError:
        # If file doesn't exist, create a new DataFrame with default column names
        existing_data = pd.DataFrame(columns=['symptom1', 'symptom2', 'symptom3', 'symptom4',
                                              'symptom5', 'symptom6', 'symptom7', 'symptom8',
                                              'symptom9', 'symptom10', 'disease'])

    # Create DataFrame for single row of data
    new_row = pd.DataFrame([symptoms + [disease]], columns=existing_data.columns)

    # Append new row to existing data
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)

    # Write updated DataFrame to Excel file
    updated_data.to_excel(file_path, index=False)
    print(f"Data has been successfully added to '{file_path}'")

# @app.route('/predict', methods=['GET'])
# def predict():
#     # Get symptoms array from POST request
#     symptoms = request.json['symptoms']
#
#     # Predict disease probabilities based on symptoms
#     result = predict_disease_probabilities(symptoms)
#
#     # Return predicted probabilities as JSON response
#     return jsonify(result)
#
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get symptoms array from request body
        symptoms_str = request.json.get('symptoms')

        # Check if symptoms_str is None or empty
        if not symptoms_str:
            return jsonify({'error': 'Symptoms parameter is missing or empty'})

        # Convert string of symptoms to list of integers
        # symptoms = list(map(int, symptoms_str.split(',')))

        # Predict disease probabilities based on symptoms
        result = predict_disease_probabilities(symptoms_str)

        # Return predicted probabilities as JSON response
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        # Get symptoms array and text input from request body
        symptoms = request.json.get('selectedSymptoms')
        text_input = request.json.get('textInput')

        # Check if symptoms or text_input is None or empty
        if not symptoms or not text_input:
            print('ji er')
            return jsonify({'error': 'Symptoms or text input is missing or empty'})

        # Add data to Excel file
        add_data_to_excel('training_data1.xlsx', symptoms, text_input)

        # Return success message as JSON response
        return jsonify({'message': 'Data added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
