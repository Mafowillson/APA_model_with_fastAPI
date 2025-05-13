from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import joblib
import os
from typing import List
from scipy.sparse import csr_matrix, hstack
from app.utils.features import explain_violations
from app.utils.extract_refernces import extract_references_from_file
from fastapi.responses import JSONResponse

# Load model and vectorizer
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "../app/model/APA_model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "../app/model/vectorizer.pkl"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "APA Reference Classifier API"}

@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="File must be a PDF or DOCX.")

    # Extract references from file
    references = await extract_references_from_file(file)

    results = []

    for ref in references:
        rule_features, violations = explain_violations(ref)
        feature_names = list(rule_features.keys())
        rule_vector = np.array([[rule_features[f] for f in feature_names]])
        text_vector = vectorizer.transform([ref])
        final_vector = hstack([text_vector, csr_matrix(rule_vector)])
        prediction = model.predict(final_vector)[0]
        probability = model.predict_proba(final_vector)[0].tolist()
        label = 'APA' if prediction == 1 else 'notAPA'

        results.append({
            "input": ref,
            "prediction": label,
            "violations": violations if label == 'notAPA' else [],
            "raw_features": rule_features,
            "probabilities": probability
        })

    return JSONResponse(content={"results": results})
