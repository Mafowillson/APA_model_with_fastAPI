import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, BertTokenizerFast, BertForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load and clean data
df = pd.read_csv("apa_dataset_2000.csv")
df = df[['reference_text', 'label']].dropna()
df['label'] = df['label'].str.strip().str.lower()

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['label'])  # APA → 0 or 1 

# Split dataset
dataset = Dataset.from_pandas(df[['reference_text', 'labels']])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["reference_text"], padding="max_length", truncation=True)

encoded_dataset = dataset.map(tokenize, batched=True)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./apa_model_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained("./apa_bert_model")
tokenizer.save_pretrained("./apa_bert_model")

def predict_apa(citation):
    inputs = tokenizer(citation, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prob = torch.softmax(outputs.logits, dim=1)
    return prob[0][1].item()  # Probability of being APA

test_citations = [
    "King, M. L. (2023). Transformers for citation analysis. arXiv preprint arXiv:2301.12345.",  # APA
    "King, M.L. 'Transformers for Citation Analysis.' arXiv, 2023."  # Non-APA
]

print("\nPredictions:")
for citation in test_citations:
    score = predict_apa(citation)
    print(f"[{'APA' if score > 0.5 else 'Non-APA'}] ({score:.2f}): {citation[:60]}...")