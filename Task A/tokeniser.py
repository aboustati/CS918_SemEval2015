from __future__ import division
from pickling import *
import numpy as np
import csv, twokenize

#Tokenise a tweet
def tokenise(tweet):
    return twokenize.tokenize(tweet)

#Tokenise a list of tweets
def tokenise_tweets(stream):
    output = []
    for i in xrange(len(stream)):
        output.append(tokenise(stream[i]))
    return output


if __name__ == "__main__":
    trainingData = []
    with open("./Exercise2_data/twitter-train-cleansed-B.tsv", "r") as tsvin:
        tsvin = csv.reader(tsvin, delimiter = "\t")
        for row in tsvin:
            trainingData.append(row)

    tokenised_tweets = tokenise_tweets(trainingData[:][3])
    labels = []
    for i in xrange(len(trainingData)):
        labels.append(trainingData[i][2])
        tokenised_tweets.append(tokenise(trainingData[i][3]))

    pickleSave("trainB.pickle", (tokenised_tweets, labels))

