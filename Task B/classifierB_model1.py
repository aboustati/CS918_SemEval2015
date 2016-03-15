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

#Load training data
[train_Y, train_tweets, train_length, train_embedding, train_affin, train_senti140, train_nrc] = pickleLoad("classifierB_processed_train_data.pickle")

#Load development data
[dev_Y, dev_tweets, dev_length, dev_embedding, dev_affin,
    dev_senti140, dev_nrc] = pickleLoad("classifierB_processed_dev_data.pickle")

#Initialise vectoriser
vectoriser = CountVectorizer(ngram_range = (1,2), min_df = 5, max_df = 0.8)

#Create BOW representations for training data
train_X = vectoriser.fit_transform(train_tweets)
#Create feature matrices
train_length = csr_matrix(train_length)
train_embedding = csr_matrix(train_embedding)
train_affin = csr_matrix(train_affin)
train_senti140 = csr_matrix(train_senti140)
train_nrc = csr_matrix(train_nrc)
#Concatenate feature matrices
train_X = hstack([train_X, train_length, train_embedding, train_affin, train_senti140, train_nrc])

#Create BOW representations for development data
dev_X = vectoriser.transform(dev_tweets)
#Create feature matrices
dev_length = csr_matrix(dev_length)
dev_embedding = csr_matrix(dev_embedding)
dev_affin = csr_matrix(dev_affin)
dev_senti140 = csr_matrix(dev_senti140)
dev_nrc = csr_matrix(dev_nrc)
#Concatenate feature matrices
dev_X = hstack([dev_X, dev_length, dev_embedding, dev_affin, dev_senti140, dev_nrc])

#Build Score Function for Cross Validation
f1_cv = make_scorer(f1_score, average = "macro", labels = [1,2])

#Build model - MaxEnt
classifier = LogisticRegression()
params = {
    'penalty': ['l1', 'l2'],
    'C':[1, 0.1, 0.01],
}

#Train model if cross-validated
if cross_validated:
    #Set new best parameters and score on test set
    classifier.set_params(penalty = 'l2', C = 0.1)
    classifier.fit(train_X, train_Y)
    predicted = classifier.predict(dev_X)
    predicted_probabilities = classifier.predict_proba(dev_X)
    print "MaxEnt"
    print "macro f1 = " + str(f1_score(dev_Y, predicted, average = 'macro', labels = [1,2]))
    print "accuracy = " + str(accuracy_score(dev_Y, predicted))
    pickleSave("classifierB_model1_probabilities.pickle", predicted_probabilities)
    pickleSave("classifierB_model1.pickle", classifier)
    pickleSave("classifierB_vectoriser.pickle", vectoriser)

#Cross validate model if not done
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
