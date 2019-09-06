#!/bin/python

import csv
import copy

def read_files(tarfname, tfidf=True):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()

    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    if tfidf is False:
        sentiment.count_vect = CountVectorizer(
            ngram_range=(1, 3), token_pattern = r"(?u)\b[\w']+\b")
    else:
        sentiment.count_vect = TfidfVectorizer(
            ngram_range=(1, 3), token_pattern = r"(?u)\b[\w']+\b")
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_unlabeled_from_test(testfname, sentiment):
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    with open(testfname, encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            label = line[0]
            text = line[1]
            unlabeled.data.append(text)          
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)

    return unlabeled


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def read_tsv_without_tar(fname):   
    data = []
    labels = []
    
    with open(fname, encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            label = line[0]
            text = line[1]
            labels.append(label)
            data.append(text)
            
    return data, labels

def read_tsv_files_without_tar(train_fname='data/train.tsv', dev_fname='data/dev.tsv', test_fname='data/test.tsv'):        
            
    class Data: pass
    sentiment = Data()
    sentiment.train_data, sentiment.train_labels = read_tsv_without_tar(train_fname)
    sentiment.dev_data, sentiment.dev_labels = read_tsv_without_tar(dev_fname)
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentiment.count_vect = TfidfVectorizer(ngram_range=(1, 3), token_pattern = r"(?u)\b[\w']+\b")
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)

    return sentiment
