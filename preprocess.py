import re
from nltk.tokenize import word_tokenize

def remove_punctuation(text):
    translation_table = str.maketrans('-',' ',custom_exclude)
    new_text = text.translate(translation_table)
    return new_text

def remove_non_ascii(text):
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_non_ascii(text)
    text = word_tokenize(text)
    return text
