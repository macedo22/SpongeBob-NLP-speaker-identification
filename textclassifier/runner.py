from . import sentiment as sentimentinterface
from . import classify
import timeit
import numpy as np
import importlib
importlib.reload(sentimentinterface)
importlib.reload(classify)
import pickle
import random
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import zscore

number_choices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
characters = ['Anchovies', 'DoodleBob', 'Gary', 'Narrator', 'Mr. Krabs',
              'Patrick', 'Plankton', 'Sandy', 'SpongeBob', 'Squidward']
characters_possible_lower = ['anchovies', 'doodlebob', 'doodle bob', 'gary',
                             'narrator', 'mr. krabs', 'mr krabs', 'patrick',
                             'plankton', 'sandy', 'spongebob', 'sponge bob',
                             'squidward']


def setup():
    import os

    results_dir = './textclassifier/static/textclassifier/images/results/'
    if not os.path.isdir(results_dir):
       os.makedirs(results_dir)

    class Data: pass
    sentiment = Data()
    
    sentiment.train_data, sentiment.train_labels = (
        sentimentinterface.read_tsv_without_tar('textclassifier/data/expanded_reduced.tsv'))
    sentiment.dev_data, sentiment.dev_labels = (
        sentimentinterface.read_tsv_without_tar('textclassifier/data/dev_reduced.tsv'))
    sentiment.test_data, sentiment.test_labels = (
        sentimentinterface.read_tsv_without_tar('textclassifier/data/test_reduced.tsv'))
    sentiment.count_vect = joblib.load('textclassifier/tfidf.pkl')
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    sentiment.testX = sentiment.count_vect.transform(sentiment.test_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    sentiment.testy = sentiment.le.transform(sentiment.test_labels)
        
    cls = joblib.load('textclassifier/joblib_model.pkl')
    
    return (sentiment, cls)

        
def run_random(sentiment, cls):
    randomIndex = random.randint(0, len(sentiment.test_data))
    text = sentiment.test_data[randomIndex]
    true_label = sentiment.test_labels[randomIndex]
    context = classify.predict_with_explanation([text], sentiment, cls, true_label)
    return context
    
def run_given_label(sentiment, cls, desired_label):
    randomIndex = random.randint(0, len(sentiment.test_data))
    text = sentiment.test_data[randomIndex]
    true_label = sentiment.test_labels[randomIndex]
    
    while desired_label != true_label:
        randomIndex = random.randint(0, len(sentiment.test_data))
        text = sentiment.test_data[randomIndex]
        true_label = sentiment.test_labels[randomIndex]
    
    context = classify.predict_with_explanation([text], sentiment, cls, true_label)
    return context

def run_given_both(sentiment, cls, text, true_label):
    context = classify.predict_with_explanation([text], sentiment, cls, true_label)
    return context
