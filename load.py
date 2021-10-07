
# Run in python console
import nltk; nltk.download('stopwords')

# Run in terminal or command prompt
# pip install spacy

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim_models # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

import os

def main():

    def probability(elem):
        return elem[1]
    
    lda_model = gensim.models.ldamodel.LdaModel.load("test_model.model")
    
    new_doc = 'this book describes windows software'.split()
    new_doc_bow = lda_model.id2word.doc2bow(new_doc)
    new_doc_topics = lda_model.get_document_topics(new_doc_bow)
    # pprint(new_doc_topics)

    new_doc_topics.sort(key=probability, reverse=True)
    # pprint(new_doc_topics)

    if len(new_doc_topics) > 0 :
        topic_no = new_doc_topics[0][0]
        topic_prop = new_doc_topics[0][1]
        # pprint(topic_prop)

if __name__ == "__main__":
    main()
