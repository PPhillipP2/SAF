import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import spacy_cleaner
from spacy_cleaner import processing, Cleaner
from spacy_cleaner.processing import removers, replacers, mutators
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re
from spacy.pipeline import TextCategorizer
from spacy.training import Example

# load dataset file
df = pd.read_csv(r'Data/IMDB Dataset.csv')

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


# Test for tokenization
#test = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."

review_set = df.iloc[:, 0].tolist()

# spacy default pipeline
print("Default spacy tokens:")
for review in review_set:
    print(review)
    #doc = nlp(clean_text(review))
    doc = nlp(review)
    # all desired properties from tokens picked here
    print([[token.text, token.pos_] for token in doc if not token.is_stop and not token.is_punct])

    #for sentence in doc.sents:
    #    print(sentence._.blob.sentiment)

    print(doc._.blob.sentiment)

    print()


"""
pipeline = Cleaner(
    nlp,
    processing.remove_stopword_token,
    processing.remove_punctuation_token,
    processing.mutate_lemma_token,
)


test2 = ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."]

# custom spacy_cleaner pipeline
processed = pipeline.clean(test2)
print("Custom spacy_cleaner pipeline:")
print(processed)




# clean and preprocess the text review data column in df
reviews = df.iloc[:, 0].tolist()

df['clean_text'] = pipeline.clean(reviews)


print("Movie Review Dataframe (custom pipeline:")
print(df['clean_text'])


# tokenize the CLEANED text
#df['tokens'] = df['clean_text'][0:5].apply(tokenize)

# print for verification
#print(df[['clean_text', 'tokens']].head(5))
"""

# Everything below is WIP

"""
textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True})

# Add labels
textcat.add_label("amazing")
textcat.add_label("good")
textcat.add_label("okay")
textcat.add_label("bad")
textcat.add_label("terrible")

# Train only textcat
training_excluded_pipes = [
    pipe for pipe in nlp.pipe_names if pipe != "textcat"
]
with nlp.disable_pipes(training_excluded_pipes):
    optimizer = nlp.begin_training()
    # Training loop
    print("Beginning training")
    batch_sizes = compounding(
        4.0, 32.0, 1.001
    )
    for i in range(iterations):
        loss = {}
        random.shuffle(training_data)
        batches = minibatch(training_data, size=batch_sizes)
        for batch in batches:
            text, labels = zip(*batch)
            nlp.update(
                text,
                labels,
                drop=0.2,
                sgd=optimizer,
                losses=loss
            )
"""
