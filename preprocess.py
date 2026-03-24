import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep some negative stopwords — important for mental health text
keep_words = {'no', 'not', 'never', 'nothing', 'nobody', 'nowhere',
              'neither', 'nor', 'hardly', 'barely', 'scarcely'}
stop_words = stop_words - keep_words

def clean_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)            # remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()         # remove extra spaces
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("Loading datasets...")

# ── Dataset 1: Combined Mental Health Data ─────────────────────────────────────
df1 = pd.read_csv('Combined Data.csv')
df1 = df1[['statement', 'status']].copy()
df1.columns = ['text', 'label']
df1['label'] = df1['label'].str.strip()

# ── Dataset 2: Suicide Detection (Reddit) ──────────────────────────────────────
df2 = pd.read_csv('Suicide_Detection.csv')
df2 = df2[['text', 'class']].copy()
df2.columns = ['text', 'label']
df2['label'] = df2['label'].map({'suicide': 'Suicidal', 'non-suicide': 'Normal'})

# ── Combine ────────────────────────────────────────────────────────────────────
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip() != '']

print(f"Combined dataset: {len(df)} rows")
print("Before balancing:")
print(df['label'].value_counts())

# ── Balance classes ────────────────────────────────────────────────────────────
# Cap majority classes, upsample minority classes
TARGET = 10000

balanced_dfs = []
for label in df['label'].unique():
    subset = df[df['label'] == label]
    if len(subset) >= TARGET:
        subset = resample(subset, n_samples=TARGET, random_state=42, replace=False)
    else:
        subset = resample(subset, n_samples=TARGET, random_state=42, replace=True)
    balanced_dfs.append(subset)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balancing: {len(df_balanced)} rows")
print(df_balanced['label'].value_counts())

# ── Clean text ─────────────────────────────────────────────────────────────────
print("\nCleaning text... (this may take a few minutes)")
df_balanced['cleaned_text'] = df_balanced['text'].apply(clean_text)

# Remove empty rows after cleaning
df_balanced = df_balanced[df_balanced['cleaned_text'].str.strip() != '']
df_balanced = df_balanced.reset_index(drop=True)

print(f"Final dataset: {len(df_balanced)} rows")

# ── Save ───────────────────────────────────────────────────────────────────────
df_balanced.to_csv('processed_data.csv', index=False)
print("\n✓ Saved as processed_data.csv")
print("\nSample cleaned text:")
print(df_balanced[['text', 'cleaned_text', 'label']].head(5).to_string())
print("\nDone! Now run: python train_models.py")
