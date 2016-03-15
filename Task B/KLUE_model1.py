from __future__ import division
from pickling import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import numpy as np

#Cross_Validated
cross_validated = True

#Load training features
[train_Y, train_tweets, train_polarity, train_p_polarity, train_n_polarity, train_length, train_p_emo, train_n_emo, train_embedding] = pickleLoad("processed_train_data.pickle")

#Load development features
[dev_Y, dev_tweets, dev_polarity, dev_p_polarity, dev_n_polarity, dev_length, dev_p_emo, dev_n_emo, dev_embedding] = pickleLoad("processed_dev_data.pickle")

#Initialise vectoriser
vectoriser = CountVectorizer(ngram_range = (1,2), min_df = 5, max_df = 0.8)

#Apply vectoriser to obtain BOW representations for training set
train_X = vectoriser.fit_transform(train_tweets)

#Construct training dataset feature matrices
train_polarity = csr_matrix(train_polarity)
train_p_polarity = csr_matrix(train_p_polarity)
train_n_polarity = csr_matrix(train_n_polarity)
train_length = csr_matrix(train_length)
train_p_emo = csr_matrix(train_p_emo)
train_n_emo = csr_matrix(train_n_emo)
train_embedding = csr_matrix(train_embedding)
#Concatenate feature matrices
train_X = hstack([train_X, train_polarity, train_p_polarity, train_n_polarity, train_length, train_p_emo, train_n_emo, train_embedding])

#Apply vectoriser to obtain BOW representations for development set
dev_X = vectoriser.transform(dev_tweets)

#Construct development dataset feature matrices
dev_polarity = csr_matrix(dev_polarity)
dev_p_polarity = csr_matrix(dev_p_polarity)
dev_n_polarity = csr_matrix(dev_n_polarity)
dev_length = csr_matrix(dev_length)
dev_p_emo = csr_matrix(dev_p_emo)
dev_n_emo = csr_matrix(dev_n_emo)
dev_embedding = csr_matrix(dev_embedding)
#Concatenate feature matrices
dev_X = hstack([dev_X, dev_polarity, dev_p_polarity, dev_n_polarity, dev_length, dev_p_emo, dev_n_emo, dev_embedding])

#Build Score Function for Cross Validation
f1_cv = make_scorer(f1_score, average = "macro", labels = [1,2])

#Build model - MaxEnt
classifier = LogisticRegression()
params = {
    'penalty': ['l1', 'l2'],
    'C':[1, 0.1, 0.01],
}

#Fit Model if already cross validated
if cross_validated:
    #Set new best parameters and score on test set
    classifier.set_params(penalty = 'l2', C = 0.1)
    classifier.fit(train_X, train_Y)
    predicted = classifier.predict(dev_X)
    predicted_probabilities = classifier.predict_proba(dev_X)
    print "MaxEnt"
    print "macro f1 = " + str(f1_score(dev_Y, predicted, average = 'macro', labels = [1,2]))
    print "accuracy = " + str(accuracy_score(dev_Y, predicted))
    pickleSave("klue_model1_probabilities.pickle", predicted_probabilities)
    pickleSave("klue_model1.pickle", classifier)
    pickleSave("klue_vectoriser.pickle", vectoriser)

#Cross-validate model if not already done
else:
    grid_search = GridSearchCV(classifier, params, scoring = f1_cv, n_jobs = -1, verbose = 1, cv = 10)
    grid_search.fit(train_X, train_Y)

    bestParams = grid_search.best_estimator_.get_params()
    bestScore = grid_search.best_score_
    print bestScore
    #Print best parameters
    new_params = {}
    for p in sorted(params.keys()):
        new_params[p] = bestParams[p]
    print new_params
