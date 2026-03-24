import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# We use DistilBERT — smaller, faster, almost same accuracy as BERT
# Perfect for a college project

print("Loading data...")
df = pd.read_csv('processed_data.csv')
df = df.dropna(subset=['cleaned_text', 'label'])
df = df[df['cleaned_text'].str.strip() != '']

# Use 20% of data for speed — still 14k samples
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
print(f"Using {len(df)} samples for BERT training")

le = pickle.load(open('label_encoder.pkl', 'rb'))
classes = le.classes_.tolist()
df['label_encoded'] = le.transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'],
    test_size=0.2, random_state=42, stratify=df['label_encoded']
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Classes: {classes}")

# ── Dataset class ──────────────────────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts   = list(texts)
        self.labels  = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Load DistilBERT ────────────────────────────────────────────────────────────
print("\nLoading DistilBERT tokenizer and model...")
print("(First run downloads ~250MB — please wait)")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(classes)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# ── Dataloaders ────────────────────────────────────────────────────────────────
train_dataset = MentalHealthDataset(X_train.tolist(), y_train.tolist(), tokenizer)
test_dataset  = MentalHealthDataset(X_test.tolist(),  y_test.tolist(),  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ── Training ───────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=num_training_steps)

print(f"\nTraining DistilBERT for {num_epochs} epochs...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if (i+1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\nEvaluating DistilBERT...")
model.eval()
all_preds  = []
all_probs  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        preds   = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average='weighted')
y_bin = label_binarize(all_labels, classes=list(range(len(classes))))
roc = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='weighted')

print(f"\nDistilBERT Results:")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
print(classification_report(all_labels, all_preds, target_names=classes))

# ── Save model ─────────────────────────────────────────────────────────────────
print("\nSaving DistilBERT model...")
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')
print("✓ Saved to bert_model/")

# ── Update results.json ────────────────────────────────────────────────────────
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

model_results['results']['DistilBERT'] = {
    'accuracy': round(float(acc), 4),
    'f1_score': round(float(f1), 4),
    'roc_auc':  round(float(roc), 4),
}

best = max(model_results['results'], key=lambda m: model_results['results'][m]['f1_score'])
model_results['best_model'] = best
model_results['lstm_available'] = False
model_results['bert_available'] = True

with open('model_results.json', 'w') as f:
    json.dump(model_results, f, indent=2)

print(f"\n🏆 Best model overall: {best}")
print("✓ model_results.json updated!")
print("\nNow run: python app.py")
