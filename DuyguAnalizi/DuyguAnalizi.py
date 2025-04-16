import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier



# 1. Veri Yükleme
file_path = "magaza_yorumlari_duygu_analizi.csv"  # Dosya yolunuza göre düzenleyin
data = pd.read_csv(file_path, encoding="utf-16")

# Durum sütunundaki her kategorinin sayısını hesapla
kategori_sayilari = data['Durum'].value_counts()

# Sonuçları yazdır
print("Kategori Sayıları:")
print(kategori_sayilari)


# 3. Tekrar Eden Değerlerin Kontrolü ve Silinmesi
print(f"\nTekrar Eden Değerler: {data.duplicated().sum()}")
data = data.drop_duplicates()

# 4. Eksik Verilerin Kontrolü ve Temizlenmesi
print(f"\nEksik Değerler:\n{data.isnull().sum()}")
data = data.dropna()

# 5. Metin Temizleme Fonksiyonu
def clean_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # Özel karakterleri kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    # Rakamları kaldırma
    text = re.sub(r'\d+', '', text)
    # Gereksiz boşlukları temizleme
    text = text.strip()
    return text

# Metin temizleme işlemini uygulama
data['Görüş'] = data['Görüş'].apply(clean_text)

# 6. Stop Words (Durak Kelimeler) Temizleme
stop_words = set(stopwords.words('turkish'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)

data['Görüş'] = data['Görüş'].apply(remove_stopwords)

# 7. Eğitim ve Test Verilerine Ayırma
X = data['Görüş']
y = data['Durum']  # Duygu etiketlerini içeren sütunun adı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. TF-IDF ile Özellik Çıkarımı
vectorizer = TfidfVectorizer(max_features=4000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 9. Özet Bilgi
print("\nEğitim Seti Boyutu:", X_train.shape)
print("Test Seti Boyutu:", X_test.shape)
print("TF-IDF Özellik Boyutu:", X_train_tfidf.shape)

# Modellerin tanımlanması
models = {
    "Logistic Regression": LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Performans ölçümleri için tabloyu başlat
results = []

# Her model için eğitim ve değerlendirme
for model_name, model in models.items():
    # Modeli eğit
    model.fit(X_train_tfidf, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_test_tfidf)
    
    # Performans ölçümleri
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Karışıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nKarışıklık Matrisi ({model_name}):")
    print(cm)
    
    # Karışıklık Matrisini Görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Karışıklık Matrisi - {model_name}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.show()
    
    
    # Sonuçları kaydet
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# Sonuçları bir DataFrame'e dönüştür
results_df = pd.DataFrame(results)

# Sonuçları tablo olarak yazdır
print("\nModel Karşılaştırmaları:")
print(results_df)


results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 6))
plt.title("Makine Öğrenimi Modellerinin Performans Karşılaştırması")
plt.ylabel("Skor")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.show()

# 10. Manuel Yorum ile Tahminler
test_comment = "çok az farkla beğendim."  # Örnek manuel yorum
test_comment_cleaned = clean_text(test_comment)
test_comment_cleaned = remove_stopwords(test_comment_cleaned)
test_comment_tfidf = vectorizer.transform([test_comment_cleaned])

manual_results = []
print("\nManuel Yorum:", test_comment)
print("Ön İşlemden Geçmiş Hali:", test_comment_cleaned)

for model_name, model in models.items():
    prediction = model.predict(test_comment_tfidf)[0]
    manual_results.append({"Model": model_name, "Tahmin": prediction})

# Manuel tahmin sonuçları
manual_results_df = pd.DataFrame(manual_results)
print("\nManuel Yorum için Model Tahminleri:")
print(manual_results_df)

