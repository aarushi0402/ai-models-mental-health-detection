import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, label_binarize
import warnings
warnings.filterwarnings('ignore')

print("Loading processed data...")
df = pd.read_csv('processed_data.csv')
df = df.dropna(subset=['cleaned_text', 'label'])
df = df[df['cleaned_text'].str.strip() != '']

print(f"Dataset: {len(df)} rows, {df['label'].nunique()} classes")
print(df['label'].value_counts())

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
classes = le.classes_.tolist()
print(f"\nClasses: {classes}")

# ── Train/test split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label_encoded'],
    test_size=0.2, random_state=42, stratify=df['label_encoded']
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ── TF-IDF Vectorization ───────────────────────────────────────────────────────
print("\nVectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF shape: {X_train_tfidf.shape}")

# ── Results storage ────────────────────────────────────────────────────────────
results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average='weighted')

    # ROC-AUC
    try:
        y_bin = label_binarize(y_te, classes=list(range(len(classes))))
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_te)
        else:
            probs = model.decision_function(X_te)
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        roc = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')
    except:
        roc = 0.0

    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(classification_report(y_te, preds, target_names=classes))

    results[name] = {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'roc_auc':  round(roc, 4),
        'report':   classification_report(y_te, preds, target_names=classes, output_dict=True)
    }
    return model, preds

# ── Model 1: Logistic Regression ───────────────────────────────────────────────
lr_model, lr_preds = evaluate_model(
    'Logistic Regression',
    LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ── Model 2: SVM ───────────────────────────────────────────────────────────────
svm_model, svm_preds = evaluate_model(
    'SVM',
    LinearSVC(max_iter=2000, random_state=42, C=1.0),
    X_train_tfidf, X_test_tfidf, y_train, y_test
)

# ── Model 3: LSTM (via Keras) ──────────────────────────────────────────────────
print("\nTraining LSTM...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    MAX_WORDS = 20000
    MAX_LEN   = 150

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_te_seq = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

    lstm_model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(classes), activation='softmax')
    ])
    lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = lstm_model.fit(
        X_tr_seq, y_train,
        epochs=5, batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    lstm_preds = np.argmax(lstm_model.predict(X_te_seq), axis=1)
    lstm_probs = lstm_model.predict(X_te_seq)

    acc = accuracy_score(y_test, lstm_preds)
    f1  = f1_score(y_test, lstm_preds, average='weighted')
    y_bin = label_binarize(y_test, classes=list(range(len(classes))))
    roc = roc_auc_score(y_bin, lstm_probs, multi_class='ovr', average='weighted')

    print(f"LSTM — Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, lstm_preds, target_names=classes))

    results['LSTM'] = {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'roc_auc':  round(roc, 4),
        'report':   classification_report(y_test, lstm_preds, target_names=classes, output_dict=True)
    }

    with open('lstm_tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
    lstm_model.save('lstm_model.keras')
    LSTM_AVAILABLE = True
    print("✓ LSTM saved!")

except ImportError:
    print("TensorFlow not installed — skipping LSTM")
    print("Install with: pip install tensorflow")
    LSTM_AVAILABLE = False

# ── Confusion Matrix ───────────────────────────────────────────────────────────
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, svm_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('SVM Confusion Matrix', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved confusion_matrix.png")

# ── Model Comparison Chart ─────────────────────────────────────────────────────
print("Generating comparison chart...")
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
f1_scores   = [results[m]['f1_score'] for m in model_names]
roc_aucs    = [results[m]['roc_auc']  for m in model_names]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, accuracies, width, label='Accuracy', color='#4d9fff')
ax.bar(x,         f1_scores,  width, label='F1 Score',  color='#00ff88')
ax.bar(x + width, roc_aucs,   width, label='ROC-AUC',   color='#ff6b35')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — Mental Health Detection', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for i, (a, f, r) in enumerate(zip(accuracies, f1_scores, roc_aucs)):
    ax.text(i - width, a + 0.01, f'{a:.2f}', ha='center', fontsize=9)
    ax.text(i,         f + 0.01, f'{f:.2f}', ha='center', fontsize=9)
    ax.text(i + width, r + 0.01, f'{r:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved model_comparison.png")

# ── Save models ────────────────────────────────────────────────────────────────
print("\nSaving models...")
with open('lr_model.pkl',  'wb') as f: pickle.dump(lr_model,  f)
with open('svm_model.pkl', 'wb') as f: pickle.dump(svm_model, f)
with open('tfidf.pkl',     'wb') as f: pickle.dump(tfidf,     f)
with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)

# Find best model
best_model = max(results, key=lambda m: results[m]['f1_score'])
print(f"\n🏆 Best model: {best_model} (F1: {results[best_model]['f1_score']})")

# Save results
model_results = {
    'results': results,
    'classes': classes,
    'best_model': best_model,
    'lstm_available': LSTM_AVAILABLE if 'LSTM_AVAILABLE' in dir() else False
}
with open('model_results.json', 'w') as f:
    json.dump(model_results, f, indent=2)

print("\n✓ All models saved!")
print("✓ Results saved to model_results.json")
print("\nSummary:")
for m in results:
    print(f"  {m:25s} Acc: {results[m]['accuracy']:.4f}  F1: {results[m]['f1_score']:.4f}  ROC-AUC: {results[m]['roc_auc']:.4f}")
print("\nNow run: python app.py")
