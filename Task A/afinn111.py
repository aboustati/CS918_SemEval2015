from __future__ import division
from pickling import *
import re

afinn = {}
with open("./AFINN/AFINN-111.txt", "r") as lex:
        rows = lex.readlines()
        for row in rows:
            row = re.sub("\n", "", row)
            row = row.split("\t")
            afinn[row[0]] = int(row[1])

pickleSave("afinn_111_dict.pickle", afinn)
