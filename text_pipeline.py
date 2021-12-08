# NLTK is our Natural-Language-Took-Kit
from nltk.corpus import stopwords
from nltk import word_tokenize

# Libraries for helping us with strings
import string
# Regular Expression Library
import re

stopwords = stopwords.words('english')

# Lowercase all words
def make_lower(a_string):
    return a_string.lower()

# Remove all punctuation
def remove_punctuation(a_string):
    a_string = re.sub(r'[^\w\s]','',a_string)
    return a_string

def remove_number(a_string):
    a_string = re.sub(r'[0-9]', '', a_string)
    return a_string

def remove_stopwords(a_string):
    #break the sentence into a list of words
    words = word_tokenize(a_string)
    #make a list to append valid words into
    valid_words = []
    #loop through all the words
    for word in words:
        if word not in stopwords:
            valid_words.append(word)
    a_string = ' '.join(valid_words)
    return a_string

def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)
    input_string = remove_number(input_string)
    input_string = remove_stopwords(input_string)
    return input_string