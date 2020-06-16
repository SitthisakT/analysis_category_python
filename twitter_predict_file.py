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

path = './Datasets/' # set path
file_list = ['Entertainment', 'Food', 'Sport', 'Technology', 'Travel'] # set filelist

df = [None] * len(file_list) # declare empty list filelist size

# open connection to file
for i in range(len(file_list)):
  # Initialize empty list to store tweets
  tweets_data = []
  with open(path + file_list[i] +'.txt', 'r') as tweets_file:
      # Read in tweets and store in list
      for line in tweets_file:
          tweet = json.loads(line)
          tweets_data.append(tweet)
  df[i] = pd.DataFrame(tweets_data, columns=['created_at','lang', 'text', 'source']) # set json data to dataframe with 4 columns
  df[i]['label'] = file_list[i] # set label

data = pd.concat(df).reset_index(drop=True).drop(columns=['created_at', 'lang', 'source']) # drop all columns except 'text'
data.to_csv('twitter_data.csv', sep=',', index=False) # export data for 'twitter_predict_file.py'

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

# load countvectorizer model and apply to text parsed
cv = joblib.load('./Models/cv.clf')
X_test_cv = cv.transform(data['text_parse'])

# load naivebayes model and predict it
model = joblib.load('./Models/naive_bayes.clf')
predictions = model.predict(X_test_cv)

testing_predictions = [] # declare testing_predictions list

for i in range(len(data)):
    if predictions[i] == 0:
        testing_predictions.append('Entertainment')
    elif predictions[i] == 1:
        testing_predictions.append('Food')
    elif predictions[i] == 2:
        testing_predictions.append('Sport')
    elif predictions[i] == 3:
        testing_predictions.append('Technology')
    else :
        testing_predictions.append('Travel')

check_df = pd.DataFrame({'actual_label': list(data['label']), 'prediction': testing_predictions, 'text':list(data['text'])}) # convert data to dataframe

# replace label_code to label
check_df.replace(to_replace=0, value='Entertainment', inplace=True)
check_df.replace(to_replace=1, value='Food', inplace=True)
check_df.replace(to_replace=2, value='Sport', inplace=True)
check_df.replace(to_replace=3, value='Technology', inplace=True)
check_df.replace(to_replace=4, value='Travel', inplace=True)

# export to csv
check_df.to_csv('prediction_output.csv', sep=',', index=False)