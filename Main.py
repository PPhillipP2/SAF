import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re

# load dataset file
df = pd.read_csv(r'Data/IMDB Dataset MINIMIZED.csv')

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', 100)  # Show full width of each column

# Load the English tokenizer and language model
activated = spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
spacy_text_blob = SpacyTextBlob(nlp)
nlp.add_pipe('spacytextblob')


# Clean up data
def clean_text(text):
    # remove html line breaks
    text = re.sub(r'<br\s*/?>', '', text)
    #remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# Function to analyze sentiment using SpaCyTextBlob
def analyze_sentiment(text):
    doc = nlp(text)
    # Return the polarity score
    return doc._.blob.sentiment.polarity


# Load data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['review'] = data['review'].apply(clean_text)
    return data


def perform_sentiment_analysis(filepath):
    data = load_data(filepath)
    kf = KFold(n_splits=5)  # 5-fold cross-validation
    accuracies = []
    f1_scores = []

    # Prepare data
    sentiments = [analyze_sentiment(review) for review in data['review']]
    data['predicted_sentiment'] = ['positive' if score > 0 else 'negative' for score in sentiments]
    labels = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
    predicted_labels = data['predicted_sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(data):
        y_test = labels[test_index]
        predictions = predicted_labels[test_index]

        # Calculate metrics
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracies.append(acc)
        f1_scores.append(f1)

    # Print results
    print(f"Average Accuracy: {np.mean(accuracies):.2f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.2f}")

# If this script is the main program being executed
if __name__ == "__main__":
    perform_sentiment_analysis('Data/IMDB Dataset MINIMIZED.csv')




"""
review_set = df.iloc[:, 0].tolist()

# spacy default pipeline
print("Default spacy tokens:")
for review in review_set:
    print(review)
    doc = nlp(clean_text(review))
    #doc = nlp(review)
    # all desired properties from tokens picked here
    print([[token.text, token.pos_] for token in doc if not token.is_stop and not token.is_punct])

    #for sentence in doc.sents:
    #    print(sentence._.blob.sentiment)

    print(doc._.blob.sentiment)

    print()
"""