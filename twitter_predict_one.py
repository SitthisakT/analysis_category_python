# import common library
import os
import json
import warnings
import string
import pandas as pd

# import natural language toolkit
import nltk
from nltk.corpus import stopwords # such as 'the', 'a', 'an', 'in'
from nltk.tokenize import RegexpTokenizer # split words
from nltk.stem import WordNetLemmatizer # convert words to root words
from nltk.stem.porter import PorterStemmer # e.g. convert is, am, are -> be

# import beautiful soup for pulling data out of HTML and XML files
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split # split data to train and test
from sklearn.externals import joblib # for save or load model

nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings("ignore") # ignroe any warning

def predict_twitter_text(textContent):

  data = pd.DataFrame({'text': textContent}) # input text and convert to dataFrame

  # declare variable for text processing
  tokenizer = RegexpTokenizer(r'\w+')
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()

  # remove html tag in text and get data
  def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    html_free = soup.get_text()
    return html_free

  # remove punctuation such as !, ?, #, $
  def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

  # remove stopwords
  def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

  # lemmatization
  def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text] 
    return lem_text

  # stemming
  def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

  # doing all text processing process
  data['text_parse'] = data['text'].str.replace(r'@[\w]*', '',regex=True) # remove twttier account name
  data['text_parse'] = data['text_parse'].str.replace(r'(http.*)?', '',regex=True) # remove http... such as https://www.google.co.th
  data['text_parse'] = data['text_parse'].str.replace(r'RT', '', regex=True) # remove 'RT'
  data['text_parse'] = data['text_parse'].apply(lambda x: remove_html(x))
  data['text_parse'] = data['text_parse'].apply(lambda x: remove_punctuation(x))
  data['text_parse'] = data['text_parse'].apply(lambda x: tokenizer.tokenize(x.lower()))
  data['text_parse'] = data['text_parse'].apply(lambda x: remove_stopwords(x))
  data['text_parse'] = data['text_parse'].apply(lambda x: word_lemmatizer(x))
  data['text_parse'] = data['text_parse'].apply(lambda x: word_stemmer(x))
  data = data[['text', 'text_parse']] # now data have original data and text processing-ed data

  # load countvectorizer model and apply to text parsed
  cv = joblib.load('./Models/cv.clf')
  X_test_cv = cv.transform(data['text_parse'])

  # load naivebayes model and predict it
  model = joblib.load('./Models/naive_bayes.clf')
  predictions = model.predict(X_test_cv)

  # return predicted value
  if predictions[0] == 0:
    return 'Entertainment'
  elif predictions[0] == 1:
    return 'Food'
  elif predictions[0] == 2:
    return 'Sport'
  elif predictions[0] == 3:
    return 'Technology'
  else :
    return 'Travel'

  # return error!
  return 'Error!'