from __future__ import division
from pickling import *
import re, csv

senti140 = {}
with open("./Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt", "r") as lex:
        rows = lex.readlines()
        for row in rows:
            row = re.sub("\n", "", row)
            row = row.split("\t")
            senti140[row[0]] = float(row[1])

pickleSave("senti140_dict.pickle", senti140)
