from __future__ import division
from pickling import *
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import numpy as np

#Labels dictionary
labels_dict = {0:"neutral", 1:"positive", 2:"negative"}

#Load test data for classifier B
[ID1, ID2, classifierB_tweets, classifierB_length, classifierB_embedding, classifierB_afinn, classifierB_senti140, classifierB_nrc] = pickleLoad("classifierB_processed_test_data.pickle")

#Load test data for KLUE
[ID1, ID2, klue_tweets, klue_polarity, klue_p_polarity, klue_n_polarity, klue_length, klue_p_emo, klue_n_emo, klue_embedding] = pickleLoad("klue_processed_test_data.pickle")


#Load vectoriser for classifier B
classifierB_vectoriser = pickleLoad("classifierB_vectoriser.pickle")

#Load vectoriser for klue
klue_vectoriser = pickleLoad("klue_vectoriser.pickle")

#Load models
model1 = pickleLoad("classifierB_model1.pickle")
model2 = pickleLoad("classifierB_model2.pickle")
model3 = pickleLoad("klue_model1.pickle")
model4 = pickleLoad("klue_model2.pickle")

#Create BOW representations for testing data
classifierB_test_X = classifierB_vectoriser.transform(classifierB_tweets)
#Create feature matrices
classifierB_test_length = csr_matrix(classifierB_length)
classifierB_test_embedding = csr_matrix(classifierB_embedding)
classifierB_test_affin = csr_matrix(classifierB_afinn)
classifierB_test_senti140 = csr_matrix(classifierB_senti140)
classifierB_test_nrc = csr_matrix(classifierB_nrc)
#Concatenate feature matrices
classifierB_test_X = hstack([classifierB_test_X, classifierB_test_length, classifierB_test_embedding, classifierB_test_affin, classifierB_test_senti140, classifierB_test_nrc])

#Create BOW representations for testing set
klue_test_X = klue_vectoriser.transform(klue_tweets)

#Construct training dataset feature matrices
klue_test_polarity = csr_matrix(klue_polarity)
klue_test_p_polarity = csr_matrix(klue_p_polarity)
klue_test_n_polarity = csr_matrix(klue_n_polarity)
klue_test_length = csr_matrix(klue_length)
klue_test_p_emo = csr_matrix(klue_p_emo)
klue_test_n_emo = csr_matrix(klue_n_emo)
klue_test_embedding = csr_matrix(klue_embedding)
#Concatenate feature matrices
klue_test_X = hstack([klue_test_X, klue_test_polarity, klue_test_p_polarity, klue_test_n_polarity, klue_test_length, klue_test_p_emo, klue_test_n_emo, klue_test_embedding])

probabilities1 = model1.predict_proba(classifierB_test_X)
probabilities2 = model2.predict_proba(classifierB_test_X)
probabilities3 = model3.predict_proba(klue_test_X)
probabilities4 = model4.predict_proba(klue_test_X)

#Calculate ensemble probabilities
ensemble = (probabilities1 + probabilities2 + probabilities3 + probabilities4)/4
#Predict
predicted = [np.argmax(row) for row in ensemble]
#Relabel
predicted_labels = [labels_dict[p] for p in predicted]

#Write to file
with open("task_B_submission.txt", "w") as f:
    for entry in zip(ID1, ID2, predicted_labels):
        f.write(entry[0])
        f.write("\t")
        f.write(entry[1])
        f.write("\t")
        f.write(entry[2])
        f.write("\n")

