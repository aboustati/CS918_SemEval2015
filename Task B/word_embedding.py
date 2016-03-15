from __future__ import division
from pickling import *
import re

embeddings = {}
with open("./embedding-results/sswe-u.txt", "r") as embed:
        rows = embed.readlines()
        for row in rows:
            row = re.sub("\r\n", "", row)
            row = row.split("\t")
            embeddings[row[0]] = [float(feature) for feature in row[1:]]

pickleSave("embedding_dict.pickle", embeddings)
