from __future__ import division
from pickling import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

[dev_Y, _, _, _, _, _, _, _, _] = pickleLoad("processed_dev_data.pickle")

model1 = pickleLoad("klue_model1_probabilities.pickle")
model2 = pickleLoad("klue_model2_probabilities.pickle")
model3 = pickleLoad("classifierB_model1_probabilities.pickle")
model4 = pickleLoad("classifierB_model2_probabilities.pickle")

ensemble = (model1 + model2 + model3 + model4)/4

predicted = [np.argmax(row) for row in ensemble]

print "ensemble"
print "macro f1 = " + str(f1_score(dev_Y, predicted, average = 'macro', labels = [1,2]))
print "accuracy" + str(accuracy_score(dev_Y, predicted))
