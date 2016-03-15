from __future__ import division
from pickling import *
import re, csv

nrc_hashtag_sentiment = {}
with open("./NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt", "r") as lex:
        rows = lex.readlines()
        for row in rows:
            row = re.sub("\n", "", row)
            row = row.split("\t")
            nrc_hashtag_sentiment[row[0]] = float(row[1])

pickleSave("nrc_hashtag_sentiment_dict.pickle", nrc_hashtag_sentiment)
