# Liver Disease Prediction App

This is an academic machine-learning project using the UCI Indian Liver Patient Dataset (ILPD).

## What this version does

1. Creates a simple registration/login system.
2. Shows a dashboard.
3. Lets you enter LFT-related values.
4. Automatically downloads the UCI ILPD dataset the first time the model is needed.
5. Trains Logistic Regression, SVM and Random Forest.
6. Selects the model with the best F1-score on the held-out test set.
7. Saves the trained model as `liver_model.joblib`.
8. Gives a model-based screening indication.
9. Lets you upload an LFT report image for record keeping, but asks you to verify values manually before prediction.

## Windows setup

Install Python 3.11 or newer from the official Python website.

Open Command Prompt in this project folder.

Create a virtual environment:

    python -m venv venv

Activate it:

    venv\Scripts\activate

Install libraries:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Run:

    python app.py

Open in your browser:

    http://127.0.0.1:5000

## If `python` is not recognized

Try:

    py -m venv venv
    venv\Scripts\activate
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    py app.py

## To see the model metrics

Run:

    python train_model.py

It creates:

    liver_model.joblib
    model_results.txt

## Important

This is a classroom/academic screening project, not a medical diagnostic system.
The UCI ILPD dataset contains 583 records and 10 features and has a liver-disease class label. See the UCI dataset page for details and citation.
