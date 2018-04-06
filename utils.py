from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def seg_sentences(text):
    sents = sent_tokenize(text)
    return sents

def seg_words(sent):
    words = word_tokenize(sent)
    return words
