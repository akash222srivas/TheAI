# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:53:33 2023

@author: akash
"""

from flask import Flask, render_template
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def display_news():
    # Read the Excel file into a DataFrame
    df = pd.read_excel('D:/Work/python/My_news_sentiment_app/news_data_no_duplicates.xlsx')

    # Tokenize the titles
    df['tokens'] = df['title'].apply(lambda x: word_tokenize(x.lower()))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tokens'] = df['tokens'].apply(lambda x: [token for token in x if token not in stop_words])

    # Perform sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Filter positive and negative news articles
    positive_articles = df.loc[df['sentiment'] > 0, 'title'].tolist()
    negative_articles = df.loc[df['sentiment'] < 0, 'title'].tolist()

    return render_template('news_table.html', positive_articles=positive_articles, negative_articles=negative_articles)

if __name__ == '__main__':
    app.run()
