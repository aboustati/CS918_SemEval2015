from __future__ import division
from pickling import *
from tokeniser import *
from preprocessing import *
import numpy as np
import random

#Help Functions
def average_polarity(a):
    if a:
        return sum(a)/len(a)
    else:
        return 0

#AFINN dictionary
afinn_dict = pickleLoad("afinn_111_dict.pickle")

#Labels dictionary
label_dict = {'neutral':0, 'positive':1, 'negative':2}

#Embedding dictionary
embedding_dict = pickleLoad("embedding_dict.pickle")

#Load data
data = []
with open("./Exercise2_data/twitter-test-B.tsv", "r") as tsvin:
    tsvin = csv.reader(tsvin, delimiter = "\t")
    for row in tsvin:
        data.append(row)

ID1 = [data[i][0] for i in xrange(len(data))]
ID2 = [data[i][1] for i in xrange(len(data))]
labels = [data[i][2] for i in xrange(len(data))]
tweets = [data[i][3] for i in xrange(len(data))]
tokenised_tweets = tokenise_tweets(tweets)

#Process data and map to lexicons and word embeddings
embeddings = []
mean_pol = []
pos_pol = []
neg_pol = []
length = []
processed_tweets = []
for tweet in tokenised_tweets:
    embedding = []
    polarity = []
    new_tweet = []
    for token in tweet:
        token = token.lower()
        token = replaceURLs(token)
        token = replaceUserMentions(token)
        token = replacePositiveEmoji(token)
        token = replaceNegativeEmoji(token)
        token = replaceHashtag(token)
        token = replaceRest(token)
        if token:
            if token:
                try:
                    polarity.append(afinn_dict[token])
                except:
                    pass
                try:
                    embedding.append(embedding_dict[token])
                except:
                    pass
            new_tweet.append(token)
    embeddings.append(np.vstack(embedding))
    mean_pol.append(average_polarity(polarity))
    pos_pol.append(len([pos for pos in polarity if pos > 0]))
    neg_pol.append(len([neg for neg in polarity if neg < 0]))
    length.append(len(new_tweet))
    processed_tweets.append(" ".join(new_tweet))

#Count emoticons
positive_emojis = [len(re.findall("POSITIVE_EMOJI",tweet)) for tweet in processed_tweets]
negative_emojis = [len(re.findall("NEGATIVE_EMOJI",tweet)) for tweet in processed_tweets]

#Calculate mean embedding
embeddings = [np.mean(emb, axis = 0) for emb in embeddings]

#Save data
mean_pol = np.vstack(mean_pol)
pos_pol = np.vstack(pos_pol)
neg_pol = np.vstack(neg_pol)
length = np.vstack(length)
positive_emojis = np.vstack(positive_emojis)
negative_emojis = np.vstack(negative_emojis)
embeddings = np.vstack(embeddings)

obj = [ID1, ID2, processed_tweets, mean_pol, pos_pol, neg_pol, length, positive_emojis, negative_emojis, embeddings]

pickleSave("klue_processed_test_data.pickle", obj)
