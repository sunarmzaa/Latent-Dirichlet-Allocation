from pprint import pp, pprint
import pandas as pd

# Creating a pandas dataframe from json file
data = pd.read_json('http://lda.depa.cloud/data.json')
data.head()
pprint(data)

import re
# Define a function to clean the text
def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text) 
    return text

# Cleaning the text in the review column
data['cleaned_content'] = data['content'].apply(clean)
data.head()
pprint(data)

import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet

# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

data['POS_tagged'] = data['cleaned_content'].apply(token_stop_pos)
data.head()
pprint(data)

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos: 
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:  
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew
    
data['Lemma'] = data['POS_tagged'].apply(lemmatize)
data.head()
pprint(data)

pprint(data[['content', 'Lemma']])

#Sentiment analysis using TextBlob¶
from textblob import TextBlob

# function to calculate subjectivity 
def getSubjectivity(content):
    return TextBlob(content).sentiment.subjectivity

# function to calculate polarity
def getPolarity(content):
    return TextBlob(content).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

fin_data = pd.DataFrame(data[['content', 'Lemma']])

# fin_data['Subjectivity'] = fin_data['Lemma'].apply(getSubjectivity) 
fin_data['Polarity'] = fin_data['Lemma'].apply(getPolarity) 
fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)
fin_data.head()
pprint(fin_data)

tb_counts = fin_data.Analysis.value_counts()
pprint(tb_counts)

import matplotlib.pyplot as plt
# %matplotlib inline

tb_count= fin_data.Analysis.value_counts()
plt.figure(figsize=(10, 7))
plt.pie(tb_counts.values, labels = tb_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
plt.legend()
plt.show()
