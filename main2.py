import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data (run only once, then comment out)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("punkt_tab")
# spacy.cli.download("en_core_web_sm")  # Run only once if not downloaded

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Set pandas to display full text for better inspection
pd.set_option('display.max_colwidth', None)

# Load the dataset from the CSV file
try:
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Make sure it's in the same folder as this script.")
    df = None

if df is not None:
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())

    # Ensure the columns exist
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")

    # Map sentiment to 1/0 BEFORE sampling
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Sample 500 rows
    df = df.sample(500, random_state=42).copy()

    # Download stopwords if not already done
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text).lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        doc = nlp(' '.join(tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_tokens)

    # Save original reviews for comparison
    original_reviews = df['review'].copy()
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # Display comparison
    comparison_df = pd.DataFrame({
        'Original': original_reviews,
        'Cleaned': df['cleaned_review']
    })
    print(comparison_df.head())

    # Prepare data for ML
    X = df['cleaned_review']
    y = df['sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set size:", len(X_train))
    print("Test set size:", len(X_test))

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Shape of the training TF-IDF matrix: {X_train_tfidf.shape}")
    print(f"Shape of the testing TF-IDF matrix: {X_test_tfidf.shape}")

    # Remove any NaN values from y_train and corresponding X_train_tfidf
    valid_indices = ~y_train.isna()
    X_train_tfidf_cleaned = X_train_tfidf[valid_indices.values]
    y_train_cleaned = y_train[valid_indices]

    # Train the model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_tfidf_cleaned, y_train_cleaned)
    print("Model training complete!")

    # --- Sentiment Prediction Function ---
    def predict_sentiment(text):
        """
        Takes a raw text string and predicts its sentiment using the trained model.
        """
        cleaned_text = preprocess_text(text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        return 'Positive' if prediction[0] == 1 else 'Negative'

    # --- Test Cases ---
    review_1 = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
    review_2 = "I was so bored throughout the entire film. It was a complete waste of time and money."
    review_3 = "The film was okay, not great but not terrible either. Some parts were good."

    print(f"Review: '{review_1}'\nPredicted Sentiment: {predict_sentiment(review_1)}\n")
    print(f"Review: '{review_2}'\nPredicted Sentiment: {predict_sentiment(review_2)}\n")
    print(f"Review: '{review_3}'\nPredicted Sentiment: {predict_sentiment(review_3)}\n")

    # --- Model Evaluation ---
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("DataFrame is None. Exiting.")