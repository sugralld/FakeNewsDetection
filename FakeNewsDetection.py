import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import string

# Muat dataset berita benar
true_news = pd.read_csv('True.csv')
true_news['label'] = 0  # Beri label 0 untuk berita benar

# Muat dataset berita palsu
fake_news = pd.read_csv('Fake.csv')
fake_news['label'] = 1  # Beri label 1 untuk berita palsu

# Gabungkan kedua dataset
data = pd.concat([true_news, fake_news], ignore_index=True)

# Preprocessing Teks
def text_preprocessing(text):
    # Ubah teks menjadi huruf kecil
    text = text.lower()

    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Hapus stopwords manual
    stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                  "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
                  "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
                  "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                  "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
                  "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
                  "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                  "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
                  "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
                  "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
                  "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
                  "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
                  "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
                  "you've", "your", "yours", "yourself", "yourselves"}

    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

data['text'] = data['text'].apply(text_preprocessing)

# Pisahkan data menjadi fitur (X) dan label (y)
X = data['text']
y = data['label']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat pipeline untuk memproses data dan melatih model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Latih model
pipeline.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = pipeline.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy}')

# Tampilkan classification report
print(classification_report(y_test, y_pred))

# Buat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Tampilkan confusion matrix dengan heatmap
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Buat dataframe hasil prediksi
result_df = pd.DataFrame({'Text': X_test, 'Actual Label': y_test, 'Predicted Label': y_pred})

# Tambahkan kolom untuk mendeskripsikan apakah berita palsu atau tidak
result_df['Prediction Result'] = result_df['Predicted Label'].apply(lambda x: 'Fake' if x == 1 else 'True')

# Tampilkan beberapa baris pertama dari hasil prediksi
print(result_df.head())