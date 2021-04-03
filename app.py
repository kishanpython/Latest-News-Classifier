from flask import Flask, flash, request, redirect, render_template,url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np 
import pickle
import time
import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import punkt,word_tokenize
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)

app = Flask(__name__)

# Custom Stop words
stop = ['has', 'its', "needn't", 'm', "wouldn't", 'but', 'he', "mustn't", 'his', 'there', 'or', "won't", 'can', 'd', "hadn't", 'how', 'hasn', 'very', 'wouldn', 'own', "doesn't", 'their', "isn't", 'an', "haven't", "wasn't", 'those', 'once', "shan't", 'when', "aren't", 've', 'it', "it's", 'of', "don't", 'and', 'down', 'yours', 'to', 'over', "she's", 'we', 'they', 'haven', 'having', 'ain', 'no', 'her', 'you', 'then', 'just', 'didn', 'into', 'before', 'shouldn', 'here', 'yourselves', 's', 'will', 'which', 'are', 'who', 'with', "you'd", 'this', 'me', 'themselves', "you've", 'hadn', 'mightn', 'she', 'o', 'more', 'whom', 'for', 'him', 'again', 'below', 'few', 'most', 'been', 'such', 'shan', 'is', 'ourselves', 'y', 'by', 'being', 'in', 'mustn', "you'll", 'herself', 'yourself', 'ours', 'between', 'had', 'other', "should've", 't', 'isn', 'them', 'himself', 're', 'doing', 'only', 'where', 'your', 'after', 'so', 'll', 'against', 'the', 'about', 'each', 'aren', 'wasn', "couldn't", 'have', 'ma', 'i', 'my', "mightn't", 'as', 'from', 'itself', 'under', 'same', 'why', 'any', 'our', 'be', 'off', "hasn't", 'through', "you're", 'was', 'did', "shouldn't", 'myself', 'some', 'theirs', 'hers', 'further', 'do', 'now', 'than', 'too', 'during', 'at', 'because', 'doesn', 'needn', "weren't", 'don', "didn't", 'couldn', 'what', 'does', 'if', 'up', 'on', 'these', 'should', 'all', "that'll", 'above', 'weren', 'that', 'a', 'while', 'both', 'until', 'were', 'am']

# MODEL-LOG
path_models = "models/"
path_log = path_models + 'best_log.pickle'
with open(path_log, 'rb') as data:
    log_model = pickle.load(data)

# TFIDF
path_tfidf = "models/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)


# Category mapping
category_codes = {
    'entertainment': 0,
    'technology': 1,
    'politics': 2,
    'sports': 3,
    'other':5,
}


# Inshort Data
def inshort_news():
    
    # url definition
    news_urls = ['https://inshorts.com/en/read/technology',
                 'https://inshorts.com/en/read/sports',
                 'https://inshorts.com/en/read/world',
                 'https://inshorts.com/en/read/politics',
                 'https://inshorts.com/en/read/entertainment']
    
    # Extracted Data Storage
    news_contents = []
    
    for url in news_urls:
        
        # Request's
        data = requests.get(url)
        
        # Beautify Data
        soup = BeautifulSoup(data.content, 'html.parser')
        
        # Live Data Extraction
        for article in soup.find_all('div',class_=["news-card-content news-right-box"]):
            news_article = article.find('div', attrs={"itemprop": "articleBody"}).string
            news_contents.append(news_article)

    # Randomly shuffle data        
    random.shuffle(news_contents)
    # df_features
    df = pd.DataFrame({'Content': news_contents })
    return df    




# Mix Data from different cat - 25
def inshort_news_mix():
    
    # url definition
    news_urls = ['https://inshorts.com/en/read']
    
    # Extracted Data Storage
    news_contents = []
    for url in news_urls:
        
        # Request's
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')
        
        # Live Data Extraction
        for article in soup.find_all('div',class_=["news-card-content news-right-box"]):
            news_article = article.find('div', attrs={"itemprop": "articleBody"}).string
            news_contents.append(news_article)

    # Randomly suffle data
    random.shuffle(news_contents)
    # df_features
    df = pd.DataFrame({'Content': news_contents })
    return df    



# Custom Data Input
def custom_data_input(custom_data):
    # Making DataFrame
    custom_data_inp = {'Content':[custom_data]}
    df = pd.DataFrame(data=custom_data_inp)
    return df



# Link, white spaces, bracket etc removal from raw data
def preprocessing_text(text):
    # Remove link
    sentence = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',text)
    # Remove number
    sentence = re.sub(r'\d+',' ',sentence)
    # Remove white space
    sentence = re.sub(r'\s+',' ',sentence)
    # Remove single character
    sentence = re.sub(r"\b[a-zA-Z]\b", ' ', sentence)
    # Remove bracket
    sentence = re.sub(r'\W+',' ',sentence)
    # Make sentence lowercase
    sentence = sentence.lower()
    return sentence


# Removing the stopwords from raw data
def stop_word_removal(sen):
    # tokenize the sentence
    x = word_tokenize(sen)
    
    # remove stop words
    new_x_list = [word for word in x if word not in stop]
    
    # Making sentence again
    pre_proces_sen = ' '.join(new_x_list)
    return pre_proces_sen


# Creating Feature from Raw Data
def create_feature_from_df(df):
    
    # Preprocessing the data 
    df['pre-processed'] = df['Content'].apply(preprocessing_text)
    
    # Rmoving stop words from pre-processed data
    df['pre-processed-stop-word'] = df['pre-processed'].apply(stop_word_removal)
    
    # TF-IDF
    features = tfidf.transform(df['pre-processed-stop-word']).toarray()
    return features


# Category matching function
def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category


def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each article
    predictions_proba = log_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = log_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .50:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories


# Prediction data 
def complete_df(df,categories):
    df['Prediction'] = categories
    return df


def result(data,type_of_met):

	# Custom Data
	if type_of_met=="CD":
		data_created = custom_data_input(data)

	# Real Time Data
	elif type_of_met == 'RT':
		data_created = inshort_news()

	# Mix data	
	else:
		data_created = inshort_news_mix()

	# Create features
	features = create_feature_from_df(data_created)
	# Predict
	predictions = predict_from_features(features)
	# Put into dataset
	results = complete_df(data_created, predictions)

	return results


# Home Section
@app.route('/',strict_slashes=False)
def home():
	return render_template('home.html')

# @app.route('/home',strict_slashes=False)
# def ret_home():
# 	return render_template('home.html')


# Custom Input Handler
@app.route('/custom_input', methods=['POST'])
def custom_input():

	# Request Method
	if request.method == 'POST':

		# Taking data from form
		if request.form['custom_text']:
			text_data = request.form.get('custom_text')
			meth_typ = 'CD'
			final_result = result(text_data,meth_typ)

		# Rendring Results
		content_data = final_result['Content']
		pre_data = final_result['Prediction']

		# Result
		return render_template('results_show.html', sender_data=zip(content_data,pre_data), zip=zip)

	else:
		return render_template('results_show.html')

# Real Time Data Handler
@app.route('/real_time_data/')
def real_time_data():

	# type of data --> Real Time Data
	meth_typ = "RT"

	# Result handle
	final_result = result(None,meth_typ)

	# Result formation
	content_data = final_result['Content']
	pre_data = final_result['Prediction']

	# Render Results
	return render_template('results_show.html', sender_data=zip(content_data,pre_data), zip=zip)


# Mix Data Type Handler
@app.route('/mix_data/')
def mix_data():

	# type of data --> Mixed
	meth_typ = "MX"

	# results 
	final_result = result(None,meth_typ)

	# Making response data
	content_data = final_result['Content']
	pre_data = final_result['Prediction']
	
	# Rendring Results
	return render_template('results_show.html', sender_data=zip(content_data,pre_data), zip=zip)


if __name__=='__main__':
	app.run(debug=False)

#python app.py FLASK_DEBUG=1 flask run	