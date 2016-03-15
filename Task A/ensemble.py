from __future__ import division
from pickling import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Load development data labels
[dev_Y, _, _, _, _, _, _] = pickleLoad("processed_dev_data.pickle")

#Load model probabilities
model1 = pickleLoad("model1_probabilities.pickle")
model2 = pickleLoad("model2_probabilities.pickle")
model3 = pickleLoad("model3_probabilities.pickle")

#Calculate ensemble probabilities
ensemble = (model1 + model2 + model3)/3
#Predict
predicted = [np.argmax(row) for row in ensemble]

print "Ensemble"
print "macro f1 = " + str(f1_score(dev_Y, predicted, average = 'macro', labels = [1,2]))
print "accuracy = " + str(accuracy_score(dev_Y, predicted))
