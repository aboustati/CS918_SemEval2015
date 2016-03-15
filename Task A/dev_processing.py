from __future__ import division
from tokeniser import *
from preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import numpy as np
import random

#Helper Functions
def average_polarity(tweet):
    if tweet:
        return sum(tweet)/len(tweet)
    else:
        return 0

def lookup(tweet, dict, replacement):
    out = []
    for token in tweet:
        try:
            out.append(dict[token])
        except:
            pass
    if out:
        return np.mean(np.vstack(out), axis = 0)
    else:
        return np.array(replacement)

#Labels dictionary
label_dict = {'neutral':0, 'positive':1, 'negative':2}

#AFINN dictionary
afinn_dict = pickleLoad("afinn_111_dict.pickle")

#Senti140 dictionary
senti140_dict = pickleLoad("senti140_dict.pickle")

#NRC_Sentiment dictionary
nrc_sentiment_dict = pickleLoad("nrc_hashtag_sentiment_dict.pickle")

#Embedding dictionary
embedding_dict = pickleLoad("embedding_dict.pickle")

#Load dataset
data = []
with open("./Exercise2_data/twitter-dev-gold-A.tsv", "r") as tsvin:
    tsvin = csv.reader(tsvin, delimiter = "\t")
    for row in tsvin:
        data.append(row)

#Extract data from dataset
begin = [int(data[i][2]) for i in xrange(len(data))]
end = [int(data[i][3]) for i in xrange(len(data))]
labels = [data[i][4] for i in xrange(len(data))]
tweets = [data[i][5] for i in xrange(len(data))]

#Preprocess the tweets
preprocessed_tweets = []
for tweet in tweets:
    tweet = tweet.lower()
    tweet = replaceURLs(tweet)
    tweet = replaceUserMentions(tweet)
    preprocessed_tweets.append(tweet)
tokenised_tweets = [tweet.split(" ") for tweet in preprocessed_tweets]

#Extract Target phrase and tokens around it
target = []
target_length = []
pre_target = []
post_target = []
for i in xrange(len(tokenised_tweets)):
    tweet = tokenised_tweets[i]
    b_phrase = begin[i]
    e_phrase = end[i]
    target_tweet = " ".join(tweet[b_phrase:e_phrase+1])
    target_tweet = tokenise(target_tweet)
    target.append(target_tweet)
    target_length.append(len(target_tweet))
    pre_target_tweet = " ".join(tweet[:b_phrase][-4:])
    pre_target_tweet = tokenise(pre_target_tweet)
    pre_target.append(pre_target_tweet)
    post_target_tweet = " ".join(tweet[e_phrase+1:][:4])
    post_target_tweet = tokenise(post_target_tweet)
    post_target.append(post_target_tweet)

target_length = np.vstack(target_length)

#Map to word embeddings
pre_target_embedding = []
target_embedding  = []
post_target_embedding = []
for element in zip(pre_target, target, post_target):
    pre_target_embedding.append(lookup(element[0], embedding_dict, [0.0]*50))
    target_embedding.append(lookup(element[1], embedding_dict, [0.0]*50))
    post_target_embedding.append(lookup(element[2], embedding_dict, [0.0]*50))

pre_target_embedding = np.vstack(pre_target_embedding)
target_embedding = np.vstack(target_embedding)
post_target_embedding = np.vstack(post_target_embedding)

embedding = np.hstack([pre_target_embedding, target_embedding, post_target_embedding])

#Map to AFINN-111 Lexicon
pre_target_afinn = []
target_afinn = []
post_target_afinn = []
for element in zip(pre_target, target, post_target):
    pre_target_afinn.append(lookup(element[0], afinn_dict, [0.]))
    target_afinn.append(lookup(element[1], afinn_dict, [0.]))
    post_target_afinn.append(lookup(element[2], afinn_dict, [0.]))

pre_target_afinn = np.vstack(pre_target_afinn)
target_afinn = np.vstack(target_afinn)
post_target_afinn = np.vstack(post_target_afinn)

afinn = np.hstack([pre_target_afinn, target_afinn, post_target_afinn])

#Map to Sentiment140 Lexicon
pre_target_senti140 = []
target_senti140 = []
post_target_senti140 = []
for element in zip(pre_target, target, post_target):
    pre_target_senti140.append(lookup(element[0], senti140_dict, [0.]))
    target_senti140.append(lookup(element[1], senti140_dict, [0.]))
    post_target_senti140.append(lookup(element[2], senti140_dict, [0.]))

pre_target_senti140 = np.vstack(pre_target_senti140)
target_senti140 = np.vstack(target_senti140)
post_target_senti140 = np.vstack(post_target_senti140)

senti140 = np.hstack([pre_target_senti140, target_senti140, post_target_senti140])

#Map to NRC Hashtag Sentiment Lexicon
pre_target_nrc = []
target_nrc = []
post_target_nrc = []
for element in zip(pre_target, target, post_target):
    pre_target_nrc.append(lookup(element[0], nrc_sentiment_dict, [0.]))
    target_nrc.append(lookup(element[1], nrc_sentiment_dict, [0.]))
    post_target_nrc.append(lookup(element[2], nrc_sentiment_dict, [0.]))

pre_target_nrc = np.vstack(pre_target_nrc)
target_nrc = np.vstack(target_nrc)
post_target_nrc = np.vstack(post_target_nrc)

nrc = np.hstack([pre_target_nrc, target_nrc, post_target_nrc])

#Rejoin Tweets
processed_tweets = []
for element in zip(pre_target, target, post_target):
    tweet = " ".join(element[0]+element[1]+element[2])
    processed_tweets.append(tweet)

#Convert labels from string to numeric
Y = [label_dict[label] for label in labels]

#Create data object
obj = [Y, processed_tweets, target_length, embedding, afinn, senti140, nrc]

#Save data object
pickleSave("processed_dev_data.pickle", obj)
