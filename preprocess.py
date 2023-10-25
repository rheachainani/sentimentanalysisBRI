import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

custom_exclude = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
stop_words = set(stopwords.words('english'))
custom_stop_words = stop_words - {'no' , 'not' , 'with' , 'against'}

def remove_punctuation(text):
    translation_table = str.maketrans('-',' ',custom_exclude)
    new_text = text.translate(translation_table)
    return new_text

def remove_non_ascii(text):
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text

def remove_stop_words(text):
    text = [word for word in text if word not in custom_stop_words]
    return text

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_non_ascii(text)
    text = word_tokenize(text)
    text = remove_stop_words(text)
    return text