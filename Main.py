import spacy
import pandas as pd

# df = pd.read_csv(r'C:\Users\phanp\Downloads\IMDB')

# Load the English tokenizer and language model
nlp = spacy.load('en_core_web_trf')

# Test for tokenization
test = "I love love, what the frick, dog water, long-short, crapbucket, U.K., the is"
tokens = [token.text for token in nlp(test)]
print(tokens)

# cleanTest = [token.text.lower() for token in test if not token.is_punct and not token.is_stop]
# print(cleanTest)

# Process the text with spaCy
# doc = nlp(df['IMDB Dataset'])

# Clean tokens
