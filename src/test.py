import pandas as pd
import re
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def predict_genre(new_summary, model_path="/Users/usuari/Desktop/Ironhack/BOOTCAMP/projects/final_project/src/xgboost_2.pkl", label_encoder_path="/Users/usuari/Desktop/Ironhack/BOOTCAMP/projects/final_project/src/label_encoder.pkl", vectorizer_path="/Users/usuari/Desktop/Ironhack/BOOTCAMP/projects/final_project/src/tfidf_vectorizer.pkl"):
    try:
        # Load the pre-trained model
        loaded_model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    try:
        # Load the label encoder
        label_encoder = joblib.load(label_encoder_path)
    except Exception as e:
        print(f"Error loading the label encoder: {e}")
        return

    try:
        # Load the vectorizer
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print(f"Error loading the vectorizer: {e}")
        return

    # Define a mapping between numerical labels and genre names
    label_mapping = {
        0: 'Fantasy',
        1: 'Historical Novel',
        2: 'Literary Fiction',
        3: 'Science Fiction',
        4: 'Thriller'
       
    }

    # Tokenization, lowercasing, stop word removal, lemmatization, and special character/number removal
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token]
        return ' '.join(tokens)

    # Apply text preprocessing steps to the new summary
    new_preprocessed_summary = preprocess_text(new_summary)

    # Vectorize the new summary using the loaded vectorizer
    new_tokens = vectorizer.transform([new_preprocessed_summary])

    # Make predictions on the new data
    try:
        predictions = loaded_model.predict(new_tokens)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return

    # Map the numerical label to the corresponding genre name
    predicted_genre = label_mapping.get(predictions[0], 'Unknown Genre')

    # Display the predicted genre for the new data
    print(f"Predicted Genre: {predicted_genre}")



new_summary = '''

Though he battled for years to marry her, Henry VIII has become disenchanted with the audacious Anne Boleyn. She has failed to give him a son, and her sharp intelligence and strong will have alienated his old friends and the noble families of England.

When the discarded Katherine, Henry's first wife, dies in exile from the court, Anne stands starkly exposed, the focus of gossip and malice, setting in motion a dramatic trial of the queen and her suitors for adultery and treason.

At a word from Henry, Thomas Cromwell is ready to bring her down. Over a few terrifying weeks, Anne is ensnared in a web of conspiracy, while the demure Jane Seymour stands waiting her turn for the poisoned wedding ring. But Anne and her powerful family will not yield without a ferocious struggle. To defeat the Boleyns, Cromwell must ally himself with his enemies. What price will he pay for Annie's head?"
'''

predict_genre(new_summary)
