from __future__ import division
from pickling import *
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import numpy as np

#Labels dictionary
labels_dict = {0:"neutral", 1:"positive", 2:"negative"}

#Load test data
[ID1, ID2, processed_tweets, target_length, embedding, afinn, senti140, nrc] = pickleLoad("processed_test_data.pickle")

#Load vectoriser
vectoriser = pickleLoad("vectoriser.pickle")

#Load models
model1 = pickleLoad("model1.pickle")
model2 = pickleLoad("model2.pickle")
model3 = pickleLoad("model3.pickle")

#Create BOW representations for training data
test_X = vectoriser.transform(processed_tweets)
#Create feature matrices
test_length = csr_matrix(target_length)
test_embedding = csr_matrix(embedding)
test_affin = csr_matrix(afinn)
test_senti140 = csr_matrix(senti140)
test_nrc = csr_matrix(nrc)
#Concatenate feature matrices
test_X = hstack([test_X, test_length, test_embedding, test_affin, test_senti140, test_nrc])

probabilities1 = model1.predict_proba(test_X)
probabilities2 = model2.predict_proba(test_X)
probabilities3 = model3.predict_proba(test_X)

#Calculate ensemble probabilities
ensemble = (probabilities1 + probabilities2 + probabilities3)/3
#Predict
predicted = [np.argmax(row) for row in ensemble]
#Relabel
predicted_labels = [labels_dict[p] for p in predicted]

#Write to file
with open("task_A_submission.txt", "w") as f:
    for entry in zip(ID1, ID2, predicted_labels):
        f.write(entry[0])
        f.write("\t")
        f.write(entry[1])
        f.write("\t")
        f.write(entry[2])
        f.write("\n")

