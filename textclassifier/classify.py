#!/bin/python

from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import copy
from . import sentiment as sentimentinterface
from subprocess import call
import csv
from scipy.stats import zscore


def get_label_from_index(index):
    if index == 0:
        return 'Anchovies'
    elif index == 1:
        return 'DoodleBob'
    elif index == 2:
        return 'Gary'
    elif index == 3:
        return 'Mr. Krabs'
    elif index == 4:
        return 'Narrator'
    elif index == 5:
        return 'Patrick'
    elif index == 6:
        return 'Plankton'
    elif index == 7:
        return 'Sandy'
    elif index == 8:
        return 'SpongeBob'
    else:
        return 'Squidward'


def get_index_from_label(label):
    if label == 'Anchovies':
        return 0
    elif label == 'DoodleBob':
        return 1
    elif label == 'Gary':
        return 2
    elif label == 'Mr. Krabs':
        return 3
    elif label == 'Narrator':
        return 4
    elif label == 'Patrick':
        return 5
    elif label == 'Plankton':
        return 6
    elif label == 'Sandy':
        return 7
    elif label == 'SpongeBob': 
        return 8
    else:
        return 9


def get_text_color_from_label(label):
    label = (label.get_text().lower())[:12]
    if 'anchovies' in label:
        return (1.0, 0.63921568627, 0.4)
    elif 'doodlebob' in label:
        return  (0.36078431372, 0.36078431372, 0.23921568627)
    elif 'gary' in label:
        return (0.2, 1.0, 1.0)
    elif 'mr. krabs' in label:
        return (0.83529411764, 0.0, 0.0)
    elif  'narrator'  in label:
        return (0.6, 0.0, 1.0)
    elif 'patrick'  in label:
        return (1.0, 0.5, 0.5)
    elif 'plankton' in label:
        return (0.0, 0.30196078431, 0.0)
    elif 'sandy' in label:
        return (0.8, 0.4, 0.0)
    elif 'spongebob' in label: 
        return (1.0, 0.85882352941, 0.01960784313)
    elif 'squidward' in label:
        return (0.0, 1.0, 0.6)
    else:
        return (0.0, 0.0, 0.0)


def get_confidence_statement(percent):
    if percent >= .5:
        return "very confident"
    elif percent >=.25:
        return "fairly confident"
    else:
        return "not very sure"


def get_prediction_img(pred_label):
    fname = '../../static/textclassifier/images/%s/normal.png' % pred_label
    return fname  


def get_actual_img(pred_label, true_label):
    if pred_label == true_label:
        fname = '../../static/textclassifier/images/%s/positive.gif' % true_label
    else:
        fname = '../../static/textclassifier/images/%s/negative.gif' % true_label
    return fname


def train_classifier(X, y, Cs=10):
    """Train a classifier using the given training data.

    Trains logistic regression on the input data with default parameters.
    """
    cls = LogisticRegressionCV(Cs=Cs, random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls


def evaluate(X, yt, cls, name='data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    return acc


def get_mistakes(X, yt, cls, documents):
    """Get incorrect guesses by classifier."""
    yp = cls.predict(X)
    yp_probs = cls.predict_proba(X)
    mistakes = []
    
    for i in range(len(yp)):
        if yp[i] != yt[i]:
            mistakes.append([documents[i], yp_probs[i], yt[i]])
    return mistakes


def tune_classifier(trainX, trainy, devX, devy, cls):
    """Tune the regularization strength of the classifier using a
    development data set."""
    
    train_C = cls.C_[0]
    num_train_Cs = len(cls.Cs_)
    C_step_size = num_train_Cs / 5.0
    lower_C = 5.0
    upper_C = train_C
    dev_Cs = np.linspace(lower_C, upper_C, num=50, endpoint=False)
    train_accuracy = evaluate(devX, devy, cls, name='dev data')
    
    opt_cls = None
    opt_C = train_C
    max_accuracy = train_accuracy
    
    for C in dev_Cs:
        current_cls = train_classifier(trainX, trainy, Cs=[C])
        current_C = current_cls.C_[0]  # sanity check
        current_accuracy = evaluate(devX, devy, current_cls, name='%f' % current_C)
        
        if current_accuracy >= max_accuracy:
            opt_cls = copy.deepcopy(current_cls)
            max_accuracy = current_accuracy
            opt_C = current_C
    
    cls = copy.deepcopy(opt_cls)
    
    return cls


def expand_labeled_data(sentiment, unlabeled, cls, accuracy_threshold, percent_of_labeled, outfname):
    call(['cp', 'data/train_reduced.tsv', outfname])
    
    labeled_X = sentiment.trainX
    labeled_data = sentiment.train_data

    unlabeled_X = unlabeled.X
    unlabeled_data = unlabeled.data
    number_of_samples = int(percent_of_labeled * len(labeled_data))
    random_unlabeled_indices = random.sample(range(0, len(unlabeled_data)), number_of_samples)

    for i in range(len(random_unlabeled_indices)):
        random_index = random_unlabeled_indices[i]
        pred = cls.predict_proba(unlabeled_X[random_index])
        
        max_index= np.argmax(pred[0])
        if pred[0][max_index] >= accuracy_threshold:
            with open(outfname,'a', newline='', encoding="utf8") as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_writer.writerow(
                    [get_label_from_index(max_index), '%s' % unlabeled_data[random_index]])
            
    sentiment.train_data, sentiment.train_labels = sentimentinterface.read_tsv_without_tar(outfname)

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
        
    cls = train_classifier(sentiment.trainX, sentiment.trainy)
        
    return cls

    
def predict_with_explanation(input_text, sentiment, cls, true_label):    
    text_as_str = input_text[0]
    input_vectorized = sentiment.count_vect.transform(input_text)   
    idfs = (np.nonzero(input_vectorized))[1]
    pred = cls.predict_proba(input_vectorized)[0]
    pred_with_labels = []
    
    for i, prob in enumerate(pred):
        label = get_label_from_index(i)
        pred_with_labels.append([label, prob])
        
    pred_with_labels_sorted = sorted(
        pred_with_labels, key=lambda x: abs(x[1]), reverse=True)
    best_label = pred_with_labels_sorted[0][0]
    best_label_prob = pred_with_labels_sorted[0][1]
    second_best_label = pred_with_labels_sorted[1][0]
    second_best_label_prob = pred_with_labels_sorted[1][1]
    
    pred_index = get_index_from_label(best_label)
    second_best_index = get_index_from_label(second_best_label)
    true_index = get_index_from_label(true_label)
    true_label_prob = pred_with_labels[true_index][1]
    
    confidence_statement = get_confidence_statement(best_label_prob)
    
    feature_names = sentiment.count_vect.get_feature_names()
    
    pred_zscores = zscore(cls.coef_[pred_index][:])
    second_best_zscores = zscore(cls.coef_[second_best_index][:])
    true_zscores = zscore(cls.coef_[true_index][:])
    
    pred_ngram_zscores = []
    second_ngram_best_zscores = []
    true_ngram_zscores = []
    
    for index in idfs:
        ngram = feature_names[index]
        pred_zscore = pred_zscores[index]
        pred_ngram_zscores.append([ngram, pred_zscore])
        second_best_zscore = second_best_zscores[index]
        second_ngram_best_zscores.append([ngram, second_best_zscore])
        true_zscore = true_zscores[index]
        true_ngram_zscores.append([ngram, true_zscore])   
    
    pred_ngram_zscores_sorted = sorted(
        pred_ngram_zscores, key=lambda x: abs(x[1]), reverse=True)
    pred_ngram_zscores_sorted_not_abs = sorted(
        pred_ngram_zscores, key=lambda x: x[1], reverse=True)
    second_best_ngram_zscores_sorted = sorted(
        second_ngram_best_zscores, key=lambda x: abs(x[1]), reverse=True)
    second_best_ngram_zscores_sorted_not_abs = sorted(
        second_ngram_best_zscores, key=lambda x: x[1], reverse=True)
    true_ngram_zscores_sorted = sorted(
        true_ngram_zscores, key=lambda x: abs(x[1]), reverse=True)
    true_ngram_zscores_sorted_not_abs = sorted(
        true_ngram_zscores, key=lambda x: x[1], reverse=True)
       
    if best_label == true_label and confidence_statement != 'not very sure':
        correctness = 'correct'
    elif best_label == true_label and confidence_statement == 'not very sure':
        correctness = 'correct, despite its lack of confidence'
    elif best_label != true_label and confidence_statement != 'not very sure':
        correctness = 'overconfident and incorrect'
    else:
        correctness = 'incorrect, which is not surprising due to its lack of confidence'
        
    if best_label == 'SpongeBob':
        isSpongeBob = True
    else:
        isSpongeBob = False
    
    prediction_statement = ("The model was <strong>%s</strong> that the most likely label is"
                            " <strong>%s</strong> with a <strong>%f%%</strong> probability.\n"
                            % (confidence_statement, best_label, best_label_prob*100))
    
    firstTextSpaces = 40
    secondTextSpaces= firstTextSpaces + 40
    predicted_character = best_label
    actual_character = true_label
    
    pred_img_name = get_prediction_img(best_label)
    predicted_character_img_url = pred_img_name
    actual_img_name = get_actual_img(best_label, true_label)
    actual_character_img_url = actual_img_name

    predicted_character_contributions_img_url = 'pred_ngram_contributions.png'
    comparison_character_contributions_img_url = 'comparison_ngram_contributions.png'
    predicted_character_strengths_img_url = 'pred_ngram_strengths.png'
    comparison_character_strengths_img_url = 'comparison_ngram_strengths.png'
    predicted_wordcloud_title = "%s's wordcloud" % best_label
    predicted_wordcloud_img_url = '../../static/textclassifier/images/%s/wordcloud.png' % predicted_character

    if len(pred_ngram_zscores) == 0:
        plot_explanation = (("In this case, none of the words or phrases in the quote had previouly"
                             + " been seen by the model. This means that the prediction was only based"
                             + " on which character speaks most often, in general, which is"
                             + " <strong>%s</strong>.") % best_label)
        comparison_statement = ''
        comparison_wordcloud_title = ''
        comparison_wordcloud_img_url = ''

    else:
        plot_explanation = (("In this case, the model was %s. The plot below shows the words and phrases"
                             + " that contributed the most to the model's guess. <strong>RED</strong> represents"
                             + " those believed to be things that a character would typically NOT say"
                             + " often compared to other characters. <strong>BLUE</strong> represents those believed"
                             + " to be things that the character would say often. The size of the"
                             + " red and blue bars correspond to how unlikely or likely a character"
                             + " is thought to say a word or phrase.") % correctness)
    
        if best_label != true_label:
            if isSpongeBob:
                added_msg = (" As a disclaimer, it's important to remember that most of the quotes that the"
                             + " model learned from were by SpongeBob. So it's likely that fact contributed"
                             + " to the model predicting SpongeBob. The analysis of these results should"
                             + " keep this in mind.")
            else:
                added_msg = ""            

            comparison_statement = (("Because the model predicted that <strong>%s</strong> said the quote when, in fact,"
                                     + " it was <strong>%s</strong>, both characters and their believed associations with the"
                                     + " words and phrases in the quote are compared below. <strong>%s</strong> was predicted"
                                     + " to have said this quote with only a <strong>%f%%</strong> probability.%s")
                                     % (best_label, true_label, true_label, true_label_prob, added_msg))

            comparison_wordcloud_title = "%s's wordcloud" % true_label
            comparison_wordcloud_img_url = '../../static/textclassifier/images/%s/wordcloud.png' % true_label
            
            print_ngram_zscores(
                best_label, pred_ngram_zscores_sorted, true_label, true_ngram_zscores, True,
                predicted_character_contributions_img_url)
            print_ngram_zscores(
                true_label, true_ngram_zscores_sorted, best_label, pred_ngram_zscores, True,
                comparison_character_contributions_img_url)
            print_ngram_zscores(
                best_label, pred_ngram_zscores_sorted_not_abs, true_label,
                true_ngram_zscores, False, predicted_character_strengths_img_url)
            print_ngram_zscores(
                true_label, true_ngram_zscores_sorted_not_abs, best_label,
                pred_ngram_zscores, False, comparison_character_strengths_img_url)

        else:
            if isSpongeBob:
                added_msg = (" As a disclaimer, it's important to remember that most of the quotes that the"
                             + " model learned from were by SpongeBob. So although SpongeBob was correctly"
                             + " predicted, the analysis of these results should keep this in mind")
            else:
                added_msg = ""

            comparison_statement = (("Because the model predicted that <strong>%s</strong> said the quote and was correct,"
                                     + " their believed associations with the words and phrases in the quote"
                                     + " are compared below with <strong>%s</strong>, which is the character who was predicted"
                                     + " to be second most likely to have said this quote, with a probability"
                                     + " of <strong>%f%%</strong>.%s")
                                     % (best_label, second_best_label, second_best_label_prob*100, added_msg))

            comparison_wordcloud_title = "%s's wordcloud" % second_best_label
            comparison_wordcloud_img_url = '../../static/textclassifier/images/%s/wordcloud.png' % second_best_label

            print_ngram_zscores(
                best_label, pred_ngram_zscores_sorted, second_best_label,
                second_ngram_best_zscores, True, predicted_character_contributions_img_url)
            print_ngram_zscores(
                second_best_label, second_best_ngram_zscores_sorted, best_label,
                pred_ngram_zscores, True, comparison_character_contributions_img_url)
            print_ngram_zscores(
                best_label, pred_ngram_zscores_sorted_not_abs, second_best_label,
                second_ngram_best_zscores, False, predicted_character_strengths_img_url)
            print_ngram_zscores(second_best_label, second_best_ngram_zscores_sorted_not_abs,
                best_label, pred_ngram_zscores, False, comparison_character_strengths_img_url)

    context = {
        'input_text': text_as_str,
        'prediction_statement': prediction_statement,
        'predicted_character': predicted_character,
        'actual_character': actual_character,
        'predicted_character_img_url': predicted_character_img_url,
        'actual_character_img_url': actual_character_img_url,
        'plot_explanation': plot_explanation,
        'comparison_statement': comparison_statement,
        'predicted_character_contributions_img_url': predicted_character_contributions_img_url,
        'comparison_character_contributions_img_url': comparison_character_contributions_img_url,
        'predicted_character_strengths_img_url': predicted_character_strengths_img_url,
        'comparison_character_strengths_img_url': comparison_character_strengths_img_url,
        'predicted_wordcloud_title': predicted_wordcloud_title,
        'predicted_wordcloud_img_url': predicted_wordcloud_img_url,
        'comparison_wordcloud_title': comparison_wordcloud_title,
        'comparison_wordcloud_img_url': comparison_wordcloud_img_url
    }

    return context



def print_ngram_zscores(pred_label, pred_ngram_zscores_sorted, true_label, true_ngram_zscores_unsorted, isContribution, fname):
    pred_ngram_count = len(pred_ngram_zscores_sorted)
    max_length = 5
    length = 5
    
    if pred_ngram_count > max_length:
        length = max_length
    else:
        length = pred_ngram_count
    
    reordered_true_ngrams = []
        
    for i, pred_ngram_pair in enumerate(pred_ngram_zscores_sorted):
        pred_ngram = pred_ngram_pair[0]
        for j, true_ngram_pair  in enumerate(true_ngram_zscores_unsorted):
            true_ngram = true_ngram_pair[0]
            true_zscore = true_ngram_pair[1]
            if true_ngram == pred_ngram:
                reordered_true_ngrams.append([true_ngram, true_zscore])
                break
        
    
    pred_x_values = list(zip(*pred_ngram_zscores_sorted))[1][::-1][-length:]
    pred_y_values = list(zip(*pred_ngram_zscores_sorted))[0][::-1][-length:]
    
    true_x_values = list(zip(*reordered_true_ngrams))[1][::-1][-length:]
    true_y_values = list(zip(*reordered_true_ngrams))[0][::-1][-length:]    
    
    if abs(pred_x_values[-1]) > abs(max(true_x_values, key=abs)):
        max_value = abs(pred_x_values[-1])
    else:
        max_value = abs(max(true_x_values, key=abs))

    y_axis = np.arange(1, len(pred_y_values) * 3, 1)

    colors = []
    x_values = []
    y_values = []
    for i in range(len(pred_x_values)):
        
        true_value = true_x_values[i]
        true_ngram = true_y_values[i]
        
        x_values.append(true_value)
        y_values.append('%s: %s' % (true_label.upper(), true_ngram))
        if true_value < 0.0:
            colors.append((1.0, 0.0, 0.0))
        else:
            colors.append((0.0, 0.0, 1.0))
        
        pred_value = pred_x_values[i]
        pred_ngram = pred_y_values[i]
        
        x_values.append(pred_value)
        y_values.append('%s: %s' % (pred_label.upper(), pred_ngram))
        if pred_value < 0.0:
            colors.append((1.0, 0.0, 0.0))
        else:
            colors.append((0.0, 0.0, 1.0))
            
        x_values.append(0)
        y_values.append('')
        colors.append('black')
        
    x_values = x_values[:-1]
    y_values = y_values[:-1]
    colors = colors[:-1]
            

    plot_height = 6 * (len(pred_y_values) * 3 / 10)

    fig = plt.figure(figsize=(10, plot_height))
    plt.barh(y_axis, x_values, align='center', color=colors)
    x_axis_ticks = plt.xticks()

    plt.xticks(x_axis_ticks[0], ("" for x in range(len(x_axis_ticks[0]))))
    plt.yticks(y_axis, y_values)

    ax = fig.gca()
    ax.tick_params(labelsize=12)
    blue_patch = mpatches.Patch(color='blue', label='Positive')
    red_patch = mpatches.Patch(color='red', label='Negative')
    ax.legend(fontsize='large', handles=[blue_patch, red_patch],
              bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axvline(x=0.0, linestyle=':', color='black')
    
    ticklabels = [t for t in ax.get_yticklabels()]
    for i, label in enumerate(ticklabels):
        ticklabels[i].set_color(get_text_color_from_label(label))
        ticklabels[i].set_fontweight("bold")

    if isContribution:
        the_title = "Top Words and Phrases Ranked by Contribution to Modeling %s" % pred_label.upper()
    else:
        the_title = "Top Words and Phrases Ranked by Strength of Association with %s" % pred_label.upper()
    
    
    plt.title(the_title, fontsize=16)
    plt.tight_layout()
    plt.savefig('textclassifier/static/textclassifier/images/results/%s' % fname)
    plt.close()
