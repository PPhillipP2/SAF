import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re

# load dataset file

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', 100)  # Show full width of each column

# Load the English tokenizer and language model
activated = spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')
spacy_text_blob = SpacyTextBlob(nlp)
nlp.add_pipe('spacytextblob')


# Clean up data
def clean_text(text):
    # remove html line breaks
    text = re.sub(r'<br\s*/?>', '', text)
    #remove non-alphanumeric characters
    #text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization with SpaCy
    cleaning_doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in cleaning_doc])
    return lemmatized_text



# Function to analyze sentiment using SpaCyTextBlob
def analyze_sentiment(text):
    doc = nlp(text)
    # filter features/token properties here


    # Return the sentiment
    return doc._.blob.sentiment.polarity


# Load data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['review'] = data['review'].apply(clean_text)
    return data


def perform_sentiment_analysis(filepath):
    data = load_data(filepath)
    kf = KFold(n_splits=7)  # 5-fold cross-validation
    accuracies = []
    f1_scores = []
    all_predictions = []  # List to store all predictions

    # Prepare data
    for train_index, test_index in kf.split(data):
        test_reviews = data.iloc[test_index]['review']
        y_test = data.iloc[test_index]['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
        test_sentiments = [analyze_sentiment(review) for review in test_reviews]
        predictions = ['positive' if score > 0 else 'negative' for score in test_sentiments]
        predicted_labels = [1 if pred == 'positive' else 0 for pred in predictions]

        # Store predictions
        for review, actual, predicted, rating in zip(test_reviews, y_test, predicted_labels, test_sentiments):
            all_predictions.append({'Review': review, 'Actual': actual, 'Predicted': predicted, 'Rating': ((rating+1)/2)})

        # Calculate metrics
        acc = accuracy_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        accuracies.append(acc)
        f1_scores.append(f1)


    # Convert predictions list to DataFrame and save or display
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('movie_review_predictions.csv', index=False)
    print(predictions_df.head(15))  # Optionally print the first few rows

    # Print results
    print(f"Average Model Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Model F1 Score: {np.mean(f1_scores):.4f}")

# If this script is the main program being executed
if __name__ == "__main__":
    perform_sentiment_analysis('Data/IMDB Dataset - 10000.csv')




"""
df = pd.read_csv(r'Data/IMDB Dataset MINIMIZED.csv')


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