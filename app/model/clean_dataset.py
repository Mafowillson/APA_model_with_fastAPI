import pandas as pd
import numpy as np
import re

# # === Load your dataset ===
# df = pd.read_csv("apa_dataset_2000.csv")

# # === Step 1: Normalize column names (in case they're inconsistent) ===
# df.columns = df.columns.str.strip().str.lower()

# # Ensure we have the required columns
# assert 'reference_text' in df.columns and 'label' in df.columns

# # === Step 2: Drop missing or empty entries ===
# df = df.dropna(subset=['reference_text', 'label'])
# df = df[df['reference_text'].str.strip() != '']

# # === Step 3: Standardize labels ===
# df['label'] = df['label'].str.strip().str.lower().map({'apa': 1, 'not apa': 0})

# # Drop rows with unknown/malformed labels
# df = df[df['label'].isin([0, 1])]

# # === Step 4: Strip leading/trailing spaces and fix spacing ===
# df['reference_text'] = df['reference_text'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

# # === Step 5: Remove exact duplicates ===
# df = df.drop_duplicates(subset=['reference_text', 'label'])

# # === Step 6: Optional — remove near-duplicates using basic similarity ===
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Compute similarity matrix (slow if data is huge)
# vectorizer = TfidfVectorizer().fit_transform(df['reference_text'])
# cosine_sim = cosine_similarity(vectorizer)

# # Remove near-duplicate pairs with cosine similarity > 0.95
# to_drop = set()
# for i in range(len(df)):
#     for j in range(i + 1, len(df)):
#         if cosine_sim[i, j] > 0.95:
#             to_drop.add(j)
# df = df.drop(df.index[list(to_drop)])

# # === Step 7: Balance dataset (optional but recommended) ===
# min_class_size = df['label'].value_counts().min()
# df_balanced = pd.concat([
#     df[df['label'] == 1].sample(min_class_size, random_state=42),
#     df[df['label'] == 0].sample(min_class_size, random_state=42)
# ]).reset_index(drop=True)

# # === Step 8: Save cleaned dataset ===
# df_balanced.to_csv("apa_reference_dataset_cleaned.csv", index=False)
# print("✅ Cleaned dataset saved as 'apa_reference_dataset_cleaned.csv'")
# print(df_balanced['label'].value_counts())

df = pd.read_csv("app/model/apa_reference_dataset.csv")
df['reference_text'] = df['reference_text'].str.replace('*', '', regex=False)
df.to_csv("super_smart_apa_dataset_1000_cleaned.csv", index=False)
