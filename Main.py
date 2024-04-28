import spacy
import spacy_cleaner
from spacy_cleaner import processing, Cleaner
from spacy_cleaner.processing import removers, replacers, mutators
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re
from spacy.pipeline import TextCategorizer
from spacy.training import Example

# load dataset file
df = pd.read_csv(r'Data/IMDB Dataset MINIMIZED.csv')

# Load the English tokenizer and language model
activated = spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')

# Test for tokenization
test = ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. Greg's a shitter!"]
tokens = [token.text for token in nlp(test[0])]
print(tokens)

# Clean up data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    return text

def tokenize(text):
    doc = nlp(text)
    doc = [token.lemma_ for token in doc if not token.is_stop]
    return doc

pipeline = Cleaner(
    nlp,
    removers.remove_stopword_token,
    removers.remove_punctuation_token,
    mutators.mutate_lemma_token,
)

print(pipeline.clean(test))


# clean and preprocess the text review data column in df
#df['clean_text'] = df.iloc[:, 0].apply(clean_text)

# tokenize the CLEANED text
#df['tokens'] = df['clean_text'].apply(tokenize)

# print for verification
#print(df[['clean_text', 'tokens']].head())


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