## Overview
This project aims to predict student performance based on various features using a Gradient Boosting Regressor model. The application is built with Flask and includes data preprocessing, model training, and a web interface for making predictions.
## Files and Directories
1. app.py: The main Flask application that loads the model and the preprocessing components. It provides endpoints to render the HTML page and to handle predictions.
2. experiment.ipynb: Jupyter Notebook for experimenting with the data and the model.
3. data_ingestion.py: Script for reading and ingesting data.
4. data_preprocessing.py: Script for preprocessing the data.
5. data_transformation.py: Script for transforming the data, including label encoding, scaling and splitting into training and testing sets
6. model_trainer.py: Script for training the Gradient Boosting Regressor model and saving the trained model and preprocessing components.
7. templates/index.html: HTML template for the Flask application, providing a form to input features for prediction.
8. requirements.txt: Lists the Python packages required to run the project.
9. .gitignore: Specifies files and directories to be ignored by Git.
10. README.md: Project documentation (this file)
11.  Joblib Files: These files contain the trained label encoders, scaler, and model.
## Installation
1. Clone the Repository
   
`git clone https://github.com/EstherBami/Student-Performance-Prediction-Flask.git`

`cd Student-Performance-Prediction-Flask`

3. Create and Activate Virtual Environment
   
`python -m venv student_prediction`

`source student_prediction/bin/activate  # On Windows use `student_prediction\Scripts\activate`

5. Install Dependencies
   
`pip install -r requirements.txt`

## Running the Application
1. Start the Flask Application
`python app.py`
2. Open in Browser
`Navigate to http://127.0.0.1:5000 to access the web interface.`

## Usage - Web interface
Fill in the form with the required features:

Gender

Race/Ethnicity

Parental Level of Education

Lunch

Test Preparation Course

Reading Score

Writing Score

Click the "Predict" button to get the predicted math score.

## Contributing
Fork the repository.

Create a new branch (git checkout -b feature-branch).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature-branch).

Create a new Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
