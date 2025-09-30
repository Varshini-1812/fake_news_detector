# main.py
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, Trainer, TrainingArguments

from configs.config import *             # ✅ fixed path
from utils.dataset import encode_texts   # ✅ fixed import
from models.distilbert_model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")   # ✅ lowercase 'data'
real_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
real_df["label"] = 1
df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)

X = df["text"].tolist()
y = df["label"].tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Encode datasets
train_dataset = encode_texts(tokenizer, X_train, y_train, MAX_LEN)
test_dataset = encode_texts(tokenizer, X_test, y_test, MAX_LEN)

# Model
model = get_model()
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    save_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# Interactive prediction loop
print("\n📰 Fake News Detector with BERT is ready! Type 'exit' to quit.\n")
while True:
    text = input("Enter news headline/text: ")
    if text.lower() == "exit":
        break
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    print("Prediction:", "REAL ✅" if pred == 1 else "FAKE ❌")
