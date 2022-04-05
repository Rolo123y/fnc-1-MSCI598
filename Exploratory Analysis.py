import pandas as pd
from sklearn.svm import SVC
import sys
import numpy as np
import plotly
from plotly import express as px

import feature_engineering as fe
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features

from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

import nltk
nltk.download('omw-1.4')


def generate_features(stances, dataset, name):
    h, b, y = [], [], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(
        word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(
        refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(
        polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(
        hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X, y


if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    TrainSet = DataSet()

    train_stance_df = pd.DataFrame(TrainSet.stances)
    #train_body_df = pd.DataFrame(TrainSet.articles)

    # Generate histogram plots of Stances
    stance_hist = px.histogram(train_stance_df, x="Stance", text_auto=True)
    #stance_hist.show()

    ## Additional introductory plots
    #x_features, y_features = generate_features(TrainSet.stances[:100], TrainSet, "full_data")

    # sequence length by word
    head_sent = [nltk.word_tokenize(fe.clean(sen)) for sen in train_stance_df['Headline']]
    head_len = []

    for head in head_sent:
        head_len.append(len(head))

    head_len_hist = px.histogram(head_len, labels={'value':'Sentence Length'})
    head_len_hist.show()

    head_len_threshold = np.percentile(head_len,90)
    print("The 90th percentile of sentence length is {}".format(round(head_len_threshold,2)))


    # Load/Precompute all features now
    #X_holdout, y_holdout = generate_features(hold_out_stances, d, "holdout")
    #for fold in fold_stances:
    #    Xs[fold], ys[fold] = generate_features(
    #        fold_stances[fold], d, str(fold))

    # best_score = 0
    # best_fold = None

 
