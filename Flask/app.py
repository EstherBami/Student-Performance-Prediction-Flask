from flask import Flask, jsonify, request, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and necessary preprocessing components
model = load('model.joblib')
scaler = load('scaler.joblib')
le_gender = load('le_gender.joblib')
le_race_ethnicity = load('le_race_ethnicity.joblib')
le_parental_level_of_education = load('le_parental_level_of_education.joblib')
le_lunch = load('le_lunch.joblib')
le_test_preparation_course = load('le_test_preparation_course.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_features = [
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',
            'test_preparation_course',
            'reading_score',
            'writing_score'
        ]
        
        # Validate input data
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Prepare input data as DataFrame
        input_data = {
            'gender': [data['gender']],
            'race_ethnicity': [data['race_ethnicity']],
            'parental_level_of_education': [data['parental_level_of_education']],
            'lunch': [data['lunch']],
            'test_preparation_course': [data['test_preparation_course']],
            'reading_score': [data['reading_score']],
            'writing_score': [data['writing_score']]
        }
        input_df = pd.DataFrame(input_data)

        # Apply label encoders
        input_df['gender'] = le_gender.transform(input_df['gender'])
        input_df['race_ethnicity'] = le_race_ethnicity.transform(input_df['race_ethnicity'])
        input_df['parental_level_of_education'] = le_parental_level_of_education.transform(input_df['parental_level_of_education'])
        input_df['lunch'] = le_lunch.transform(input_df['lunch'])
        input_df['test_preparation_course'] = le_test_preparation_course.transform(input_df['test_preparation_course'])

        # Ensure that all features are included
        input_df = input_df[[
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',
            'test_preparation_course',
            'reading_score',
            'writing_score'
        ]]

        # Scale features
        scaled_features = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

         # Cast the prediction to an integer
        prediction = int(prediction)

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
