import spacy
import spacy_cleaner
from spacy_cleaner.processing import removers, replacers, mutators
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re

# load dataset file
df = pd.read_csv(r'Data/IMDB Dataset MINIMIZED.csv')

# Load the English tokenizer and language model
activated = spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')

# Test for tokenization
test = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
dirty_tokens = [token.text for token in nlp(test)]
print(dirty_tokens)

# Clean up data
def clean_text(text):
    # lowercase
    text = text.lower()
    # HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Replace periods at the ends of sentences with a space
    text = re.sub(r'\.\s*', ' ', text)
    # punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    doc = nlp(text)
    doc = [token.lemma_ for token in doc if not token.is_stop]
    return doc

print(test)
cleaned_text = clean_text(test)
print(cleaned_text)

# clean and preprocess the text review data column in df
#df['clean_text'] = df.iloc[:, 0].apply(clean_text)

# tokenize the CLEANED text
#df['tokens'] = df['clean_text'].apply(tokenize)

# print for verification
#print(df[['clean_text', 'tokens']].head())