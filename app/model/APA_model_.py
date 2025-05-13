import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from app.utils.features import extract_features
import joblib
import re


df = pd.read_csv('test.csv')
print(df.head(5))

print(df.info())

print(df.describe())

# 1. Convert cathegorical data to numeric data

df['label'] = df['label'].map({'APA': 1, 'Not APA': 0})

print(df)

# 2. Divide data into train test and split

X_text = df['reference_text'].values
y = df['label'].values

# 3. Convert reference text numpy array

def custom_tokenizer(text):
    # Tokenize while preserving punctuation and parentheses
    tokens = re.findall(r"\w+|[\(\),.]", text)
    return tokens

cv = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),  # Captures sequences like "(202"
    max_features=2000,    # Limit feature explosion
    lowercase=False 
)
X_text_vec = cv.fit_transform(X_text)

# Add rule based features

rule_features = [extract_features(text) for text in X_text]
rule_df = pd.DataFrame(rule_features)
X_rules = csr_matrix(rule_df.values)

# Combining text and rule features

X = hstack([X_text_vec, X_rules])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


# 4. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Prediction (y_pred)

y_pred = model.predict(X_test)

# 6. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

# 7. graph

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["APA", "notAPA"], yticklabels=["notAPA", "APA"])
plt.title('Confusion Matrix') 
plt.xlabel('Predicted Value')
plt.ylabel("True Values")
plt.show()

print(classification_report(y_test, y_pred))

print(r2_score(y_test, y_pred))

# . Predict on new data

joblib.dump(model, 'app/model/APA_model.pkl')
joblib.dump(cv, 'app/model/vectorizer.pkl')


