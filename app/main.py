from fastapi import FastAPI
from app.schemas import TextInput
import numpy as np
import joblib
import os

from app.features import extract_features, explain_violations
from scipy.sparse import csr_matrix, hstack

# Load model and vectorizer
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "APA_model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.pkl"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "APA Reference Classifier API"}

@app.post("/predict")
def predict_text(input_data: TextInput):
    raw_text = input_data.text

    # Extract features
    rule_features, violations = explain_violations(raw_text)
    feature_names = list(rule_features.keys())
    rule_vector = np.array([[rule_features[f] for f in feature_names]])

    # Vectorize the text
    text_vector = vectorizer.transform([raw_text])

    # combine text and rules

    final_text = hstack([text_vector, csr_matrix(rule_vector)])

    # Prediction
    prediction = model.predict(final_text)[0]
    probability = model.predict_proba(final_text)[0].tolist()
    label = 'APA' if prediction == 1 else 'notAPA'

    return {
        "input": raw_text,
        "prediction": label,
        "voilations": violations if label == 'notAPA' else [],
        "raw_features": rule_features,
        "probabilities": probability
    }

    # test data
        
    # Smith, J. (2020). The psychology of learning. Academic Press.
