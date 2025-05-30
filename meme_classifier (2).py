import os
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase, remove punctuation, filter stopwords
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

# 1. Load the CSV file
df = pd.read_csv('meme_dataset/meme_data.csv')

# 2. Preprocess the captions
df['clean_caption'] = df['caption'].apply(preprocess_text)

# 3. TF-IDF vectorization and train/test split
X = df['clean_caption']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# 4. Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. (Optional) Test on 3 new meme captions
new_memes = [
    "When you finally finish your assignment",
    "Why does this always happen to me?",
    "Just another boring day"
]
for caption in new_memes:
    processed = preprocess_text(caption)
    prediction = pipeline.predict([processed])[0]
    print(f"Caption: \"{caption}\" => Predicted Sentiment: {prediction}")
import matplotlib.pyplot as plt
import os

# Display 3 meme images with their predicted sentiment
for idx, row in df.iloc[:3].iterrows():
    img_path = os.path.join('meme_dataset', row['image'])
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        pred = pipeline.predict([row['clean_caption']])[0]
        plt.title(f"Caption: {row['caption']}\nPredicted: {pred}")
        plt.show()
    else:
        print(f"Image {img_path} not found.")

# Save all predictions to meme_predictions.csv
df['prediction'] = pipeline.predict(df['clean_caption'])
df[['image', 'caption', 'prediction']].to_csv('meme_dataset/meme_predictions.csv', index=False)
print("Predictions saved to meme_dataset/meme_predictions.csv")