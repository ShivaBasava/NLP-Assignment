import pandas as pd
import numpy as np
import re
import sys
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, confusion_matrix, precision_score, recall_score
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Check if nltk data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EmotionClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = MultiOutputClassifier(LogisticRegression(
            C=1.0,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        
    def preprocess_text(self, text):
        # Text preprocessing pipeline
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path):
        # Load CSV data and preprocess text
        print("\nLoading data...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        # Check if required columns exist
        required_cols = ['text'] + self.emotion_labels
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"columns not found: {missing_cols}")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'] != '']
        
        return df
    
    def train_model(self, df, test_size=0.20, random_state=42):
        # Train the multi-label emotion classification model
        
        # Features (text) and labels (emotions)
        X = df['processed_text']
        y = df[self.emotion_labels]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)}(80%) samples")
        print(f"Test set: {len(X_test)}(20%) samples")
        
        # Vectorize the text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_vec, y_train)
        
        # Make predictions on test data (only unseen data)
        y_pred = self.model.predict(X_test_vec)
        
        # # to check test label distribution
        # print("\nBefore eval, Test Label Distribution(20%):\n")
        # print(y_test.sum(axis=0))  # Counts per emotion in test set

        # Evaluate the model
        self.evaluate_model(y_test, y_pred)
        
        return X_test, y_test, y_pred  # return the test data and predictions

    def evaluate_model(self, y_true, y_pred):
        # Evaluate the multi-label classification model
        print("\nMODEL EVALUATION")
        print("="*50)
        
        # Overall metrics
        print(f"Hamming Loss: {hamming_loss(y_true, y_pred):.4f}")
        print(f"Jaccard Score: {jaccard_score(y_true, y_pred, average='macro'):.4f}")
        
        # Micro and Macro average F1 scores
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\nMicro average F1 score: {micro_f1:.4f}")
        print(f"Macro average F1 score: {macro_f1:.4f}")
        
        # Micro and Macro average Precision
        micro_precision = precision_score(y_true, y_pred, average='micro')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        print(f"\nMicro average Precision: {micro_precision:.4f}")
        print(f"Macro average Precision: {macro_precision:.4f}")
        
        # Micro and Macro average Recall
        micro_recall = recall_score(y_true, y_pred, average='micro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        print(f"\nMicro average Recall: {micro_recall:.4f}")
        print(f"Macro average Recall: {macro_recall:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true.values.argmax(axis=1), y_pred.argmax(axis=1))
        print("\nConfusion Matrix:")
        print(cm)
        
        # Accuracy (Exact Match)
        subset_accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy (Exact Match): {subset_accuracy:.4f} ({subset_accuracy*100:.2f}%)")
        
        # Calculate Average Accuracy
        per_emotion_accuracy = []
        for i, emotion in enumerate(self.emotion_labels):
            acc = accuracy_score(y_true.iloc[:, i], y_pred[:, i])
            per_emotion_accuracy.append(acc)
        
        # Average Accuracy (mean accuracy across all emotions)
        average_accuracy = np.mean(per_emotion_accuracy)
        print(f"Average Accuracy: {average_accuracy:.4f} ({average_accuracy*100:.2f}%)")

        ## # to check test label distribution
        ## print("\nTest Label Distribution (20%):")
        ## print(y_true.sum(axis=0))  # Counts per emotion in test set

    def predict(self, X_test_vec):
        # Predict using trained model on test data
        y_pred = self.model.predict(X_test_vec)
        return y_pred  # return predictions for the test data

def main(test_file_path):
    # Initialize the classifier
    classifier = EmotionClassifier()
    
    # Load and preprocess data
    df = classifier.load_and_preprocess_data(test_file_path)
    
    # Train the model
    X_test, y_test, y_pred = classifier.train_model(df)
    
    # Only print predictions for the test set (unseen data)
    print("\nPredictions for unseen test data:")
    for i, row in enumerate(y_pred):
        print(f"Sample {i+1}: {row}")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python script.py <filename>")
            sys.exit(1)
        file_name = sys.argv[1]
        print(f"filename: {file_name}")
        main(file_name)
    except FileNotFoundError:
        print("File not found")
