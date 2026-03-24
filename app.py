from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
with open('svm_model.pkl','rb') as f: svm_model = pickle.load(f)
with open('lr_model.pkl','rb') as f: lr_model = pickle.load(f)
with open('tfidf.pkl','rb') as f: tfidf = pickle.load(f)
with open('label_encoder.pkl','rb') as f: le = pickle.load(f)
with open('model_results.json','r') as f: model_results = json.load(f)

# Load BERT
try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch
    bert_tokenizer = DistilBertTokenizer.from_pretrained('bert_model')
    bert_model = DistilBertForSequenceClassification.from_pretrained('bert_model')
    bert_model.eval()
    BERT_AVAILABLE = True
    print("✓ BERT loaded!")
except:
    BERT_AVAILABLE = False
    print("BERT not available")

classes = model_results['classes']
print(f"Classes: {classes}")
print("All models loaded!")

# ── Text preprocessing ─────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'no','not','never','nothing','nobody','nowhere','neither','nor'}
stop_words = stop_words - keep_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── Predict with traditional models ───────────────────────────────────────────
def predict_traditional(text, model, model_name):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])

    pred_idx = model.predict(vectorized)[0]
    label = le.inverse_transform([pred_idx])[0]

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vectorized)[0]
    else:
        scores = model.decision_function(vectorized)[0]
        probs = np.exp(scores) / np.sum(np.exp(scores))

    confidence = round(float(np.max(probs)) * 100, 1)
    all_probs = {cls: round(float(p)*100, 1) for cls, p in zip(classes, probs)}

    return {
        'prediction': label,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'model': model_name,
        'cleaned_text': cleaned
    }

# ── Predict with BERT ──────────────────────────────────────────────────────────
def predict_bert(text):
    import torch
    inputs = bert_tokenizer(
        text, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

    pred_idx = int(np.argmax(probs))
    label = classes[pred_idx]
    confidence = round(float(np.max(probs)) * 100, 1)
    all_probs = {cls: round(float(p)*100, 1) for cls, p in zip(classes, probs)}

    return {
        'prediction': label,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'model': 'DistilBERT',
        'cleaned_text': text
    }

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/dashboard')
def dashboard():
    import os
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindscan_dashboard.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', 'SVM')

        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400

        if model_name == 'SVM':
            result = predict_traditional(text, svm_model, 'SVM')
        elif model_name == 'Logistic Regression':
            result = predict_traditional(text, lr_model, 'Logistic Regression')
        elif model_name == 'DistilBERT' and BERT_AVAILABLE:
            result = predict_bert(text)
        else:
            result = predict_traditional(text, svm_model, 'SVM')

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-all', methods=['POST'])
def predict_all():
    """Run all models and return comparison"""
    try:
        data = request.json
        text = data.get('text', '')
        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400

        results = {}
        results['SVM'] = predict_traditional(text, svm_model, 'SVM')
        results['Logistic Regression'] = predict_traditional(text, lr_model, 'Logistic Regression')
        if BERT_AVAILABLE:
            results['DistilBERT'] = predict_bert(text)

        return jsonify({
            'results': results,
            'text': text,
            'models_used': list(results.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-results', methods=['GET'])
def get_model_results():
    return jsonify(model_results)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'bert_available': BERT_AVAILABLE,
        'classes': classes,
        'best_model': model_results.get('best_model', 'SVM')
    })

if __name__ == '__main__':
    print("\n✓ MindScan API running!")
    print("✓ Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5001)