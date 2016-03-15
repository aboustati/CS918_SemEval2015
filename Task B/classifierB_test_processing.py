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

def split_tweet(tweet, parts):
    avg = len(tweet)/parts
    out = []
    cursor = 0.
    while cursor < len(tweet):
        out.append(tweet[int(cursor):int(cursor+avg)])
        cursor += avg
    return out

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

#Load development dataset
data = []
with open("./Exercise2_data/twitter-test-B.tsv", "r") as tsvin:
    tsvin = csv.reader(tsvin, delimiter = "\t")
    for row in tsvin:
        data.append(row)

ID1 = [data[i][0] for i in xrange(len(data))]
ID2 = [data[i][1] for i in xrange(len(data))]
labels = [data[i][2] for i in xrange(len(data))]
tweets = [data[i][3] for i in xrange(len(data))]

#Preprocess tweets
preprocessed_tweets = []
for tweet in tweets:
    tweet = tweet.lower()
    tweet = replaceURLs(tweet)
    tweet = replaceUserMentions(tweet)
    preprocessed_tweets.append(tweet)

#Tokenise tweets
tokenised_tweets = tokenise_tweets(preprocessed_tweets)

#Calculate tweet length
tweet_length = [len(tweet) for tweet in tokenised_tweets]
tweet_length = np.vstack(tweet_length)

#Split tweet into three parts
start = []
middle = []
end = []
for tweet in tokenised_tweets:
    part1, part2, part3 = split_tweet(tweet, 3)
    start.append(part1)
    middle.append(part2)
    end.append(part3)

#Map to word embeddings
start_embedding = []
middle_embedding  = []
end_embedding = []
for element in zip(start, middle, end):
    start_embedding.append(lookup(element[0], embedding_dict, [0.0]*50))
    middle_embedding.append(lookup(element[1], embedding_dict, [0.0]*50))
    end_embedding.append(lookup(element[2], embedding_dict, [0.0]*50))

start_embedding = np.vstack(start_embedding)
middle_embedding = np.vstack(middle_embedding)
end_embedding = np.vstack(end_embedding)

embedding = np.hstack([start_embedding, middle_embedding, end_embedding])

#Map to AFINN-111 Lexicon
start_afinn = []
middle_afinn = []
end_afinn = []
for element in zip(start, middle, end):
    start_afinn.append(lookup(element[0], afinn_dict, [0.]))
    middle_afinn.append(lookup(element[1], afinn_dict, [0.]))
    end_afinn.append(lookup(element[2], afinn_dict, [0.]))

start_afinn = np.vstack(start_afinn)
middle_afinn = np.vstack(middle_afinn)
end_afinn = np.vstack(end_afinn)

afinn = np.hstack([start_afinn, middle_afinn, end_afinn])

#Map to Sentiment140 Lexicon
start_senti140 = []
middle_senti140 = []
end_senti140 = []
for element in zip(start, middle, end):
    start_senti140.append(lookup(element[0], senti140_dict, [0.]))
    middle_senti140.append(lookup(element[1], senti140_dict, [0.]))
    end_senti140.append(lookup(element[2], senti140_dict, [0.]))

start_senti140 = np.vstack(start_senti140)
middle_senti140 = np.vstack(middle_senti140)
end_senti140 = np.vstack(end_senti140)

senti140 = np.hstack([start_senti140, middle_senti140, end_senti140])

#Map to NRC Hashtag Sentiment Lexicon
start_nrc = []
middle_nrc = []
end_nrc = []
for element in zip(start, middle, end):
    start_nrc.append(lookup(element[0], nrc_sentiment_dict, [0.]))
    middle_nrc.append(lookup(element[1], nrc_sentiment_dict, [0.]))
    end_nrc.append(lookup(element[2], nrc_sentiment_dict, [0.]))

start_nrc = np.vstack(start_nrc)
middle_nrc = np.vstack(middle_nrc)
end_nrc = np.vstack(end_nrc)

nrc = np.hstack([start_nrc, middle_nrc, end_nrc])

#Rejoin tweets
processed_tweets = [" ".join(tweet) for tweet in tokenised_tweets]

#Create data object
obj = [ID1, ID2, processed_tweets, tweet_length, embedding, afinn, senti140, nrc]

#Save data object
pickleSave("classifierB_processed_test_data.pickle", obj)
