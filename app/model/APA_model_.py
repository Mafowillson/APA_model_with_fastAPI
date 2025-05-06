import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
from app.model.features import extract_features
import joblib


df = pd.read_csv('../app/apa_reference_dataset.csv')
print(df.head(5))

print(df.info())

print(df.describe())

# 1. Convert cathegorical data to numeric data

df['label'] = df['label'].map({'APA': 1, 'notAPA': 0})

print(df)

# 2. Divide data into train test and split

X_text = df['reference_text'].values
y = df['label'].values

# 3. Convert reference text numpy array

cv = CountVectorizer()
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

joblib.dump(model, 'APA_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')


