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

data = pd.read_csv('twitter_data.csv') # import dataset to pandas dataFrame

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

# set category code for create label code
category_codes = {
    'Entertainment': 0,
    'Food': 1,
    'Sport': 2,
    'Technology': 3,
    'Travel': 4
}

# set label code
data['label_code'] = data['label']
data = data.replace({'label_code':category_codes})

# split data for training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data[['text','text_parse']], data['label_code'], test_size=0.20)
# /////////////////////////////////////////////////////////////
dirName = 'Models' # declare directory name
 
try:
    # create target directory
    os.mkdir(dirName)
    print("[system]\tDirectory " , dirName ,  " Created ") 
except FileExistsError: # already exists
    print("[system]\tDirectory " , dirName ,  " already exists")
# /////////////////////////////////////////////////////////////
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english') # declare countvector model
X_train_cv = cv.fit_transform(X_train['text_parse']) # fit and transform model to x_train
X_test_cv = cv.transform(X_test['text_parse']) # transform x_test

joblib.dump(cv, './Models/cv.clf') # save model for use later
# /////////////////////////////////////////////////////////////
# word_freq_data = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names()) # word frequency for exploration and visualize
# top_words_data = pd.DataFrame(word_freq_data.sum()).sort_values(0, ascending=False) # top word frequency for exploration and visualize
# /////////////////////////////////////////////////////////////
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB() # declare naive bayes model
naive_bayes.fit(X_train_cv, y_train) # fit to x_train_cv and y_train
predictions = naive_bayes.predict(X_test_cv) # predict x_test_cv

joblib.dump(naive_bayes, './Models/naive_bayes.clf') # save model for use later
# /////////////////////////////////////////////////////////////

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# print('Accuracy score: ', accuracy_score(y_test, predictions))
# print('Precision score: ', precision_score(y_test, predictions, average='weighted'))
# print('Recall score: ', recall_score(y_test, predictions, average='weighted'))