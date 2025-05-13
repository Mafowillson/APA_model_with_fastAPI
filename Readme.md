# 📘 APA Citation Classifier API

This project provides a RESTful API for detecting whether references follow APA formatting. It uses a trained machine learning model and rule-based feature extraction. The backend is built using **FastAPI**, and the model can process both direct text input and uploaded `.docx` documents.

---

## 🚀 Features

- Predict if a reference follows APA formatting (`APA` or `notAPA`)
- Extract references from uploaded `.docx` files
- Returns rule-based features and violations for non-APA references
- Swagger UI docs available
- Deployable on [Render](https://render.com/)

---

## 📁 Project Structure

APA_citation_model/
├── app/
│ ├── main_file.py # Main FastAPI app
│ ├── schemas.py # Pydantic models
│ ├── model/
│ │ ├── APA_model.pkl # Trained classifier
│ │ └── vectorizer.pkl # Text vectorizer
│ ├── utils/
│ │ └── extract_references.py # Utility to extract refs from DOCX
├── requirements.txt
├── render.yaml # Render deployment config
└── README.md

---

## 🧪 Local Development

### 🔧 Requirements

- Python 3.10+
- pip

### 🐍 Create a virtual environment

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

📦 Install dependencies

pip install -r requirements.txt

▶️ Run the app locally

uvicorn app.main_file:app --reload


🧪 Visit the API Docs
Go to: http://127.0.0.1:8000/docs

📤 API Endpoints
POST /predict

✅ Input: .docx or pdf file

file upload (.docx) or (pdf) containing a References section.

🧾 Output:

{
  "results": [
    {
      "input": "Smith, J. (2020). The psychology of learning. Academic Press.",
      "prediction": "APA",
      "violations": [],
      "raw_features": { ... },
      "probabilities": [0.1, 0.9]
    }
  ]
}

🤖 Model Info

Model: Scikit-learn classifier trained to identify APA citations

Features: Combination of vectorized text and rule-based indicators

Format: Saved using joblib

🛠️ Future Improvements

Improve reference parsing logic

Add support for batch predictions via API

📬 Contact
For questions or feedback, feel free to open an issue or reach out.

📄 License
MIT License – see LICENSE file for details.




