# ===============================================
#  Yoga Pratama (312210042)
#  UTS Kecerdasan Buatan - Deteksi Hate Speech
# ===============================================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords

# --- Setup stopwords bahasa Indonesia ---
nltk.download('stopwords')
stopwords_id = set(stopwords.words('indonesian'))

# --- 1. Load Dataset ---
# Ganti path sesuai lokasi file hasil ekstrak dari archive (8).zip
df = pd.read_csv("archive/data.csv", encoding='latin1')

# Tampilkan kolom untuk memastikan
print("Kolom dalam dataset:", list(df.columns))

# --- 2. Cek dan sesuaikan nama kolom ---
# otomatis deteksi kolom teks dan label
if 'tweet' in df.columns:
    df['text'] = df['tweet']
elif 'Tweet' in df.columns:
    df['text'] = df['Tweet']
elif 'content' in df.columns:
    df['text'] = df['content']
elif 'Text' in df.columns:
    df['text'] = df['Text']
else:
    raise ValueError("Kolom teks tidak ditemukan! Cek nama kolom di dataset.")

if 'HS' in df.columns:
    df['label'] = df['HS']
elif 'label' in df.columns:
    df['label'] = df['label']
elif 'Label' in df.columns:
    df['label'] = df['Label']
elif 'hate_speech' in df.columns:
    df['label'] = df['hate_speech']
else:
    raise ValueError("Kolom label tidak ditemukan! Cek nama kolom di dataset.")

# Pilih hanya dua kolom utama
df = df[['text', 'label']].dropna()

# Pastikan label hanya 0 dan 1
if df['label'].nunique() > 2:
    df['label'] = df['label'].apply(lambda x: 1 if int(x) > 0 else 0)

print("\nJumlah data:", len(df))
print(df.head())

# --- 3. Preprocessing Teks ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)       # hapus URL
    text = re.sub(r'[^a-z\s]', ' ', text)      # hapus angka & tanda baca
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords_id]
    return " ".join(tokens)

print("\nSedang melakukan preprocessing teks...")
df['clean'] = df['text'].apply(clean_text)

# --- 4. Representasi Fitur (TF-IDF) ---
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean'])
y = df['label']

# --- 5. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. Model: Logistic Regression ---
print("\nSedang melatih model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- 7. Evaluasi Model ---
y_pred = model.predict(X_test)

print("\n=== HASIL EVALUASI MODEL ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 8. Visualisasi Distribusi Label ---
plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.title("Distribusi Label (0 = Tidak Hate Speech, 1 = Hate Speech)")
plt.xlabel("Label")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig("distribusi_label.png")
plt.show()

# --- 9. Word Cloud untuk Hate Speech ---
hate_text = " ".join(df[df['label'] == 1]['clean'])
wc = WordCloud(width=800, height=400, background_color='white').generate(hate_text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Hate Speech (Label = 1)")
plt.tight_layout()
plt.savefig("wordcloud_hate.png")
plt.show()

print("\n✅ Selesai! Grafik disimpan sebagai:")
print(" - distribusi_label.png")
print(" - wordcloud_hate.png")
