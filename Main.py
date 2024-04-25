import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re

df = pd.read_csv(r'Data/IMDB_Dataset.csv')

# Load the English tokenizer and language model
nlp = spacy.load('en_core_web_sm')

# Test for tokenization
test = "I love love, what the frick, dog water, long-short, crapbucket, U.K., the is"
tokens = [token.text for token in nlp(test)]
print(tokens)

# Clean up data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens2 = nlp(text)
    tokens2 = [token.text for token in tokens2 if token.text]
    return tokens2

df['clean_review'] = df['review'].apply(clean_text)


# Print the tokens
print(df['clean_review'].iloc[0])

