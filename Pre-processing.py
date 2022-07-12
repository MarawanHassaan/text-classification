import string
import csv
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize(s):

    #Remove punctuation
    s = s.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))

    #Strip and convert all letters to lowercase
    s = s.strip().lower()

    #replace \n with space
    s = s.replace("\n", " ")

    return s

f = open("lyrics.csv", encoding = "utf8")
reader = csv.reader(f)
#skip header
next(reader)

classes = ["Rock", "Pop", "Metal", "Hip-Hop"]
genres = dict()

i = 0
for row in reader:
  genre = row[4]
  lyrics = normalize(row[5])
  
  #if there are lyrics and label
  if (lyrics!="" and genre!="Other" and genre!="Not Available" and 
      re.sub(r'[][}{)( .*]', '', lyrics) != "instrumental"):

    lyrics= re.sub(r'[^;:&$?!\'0-9a-zA-Z ]', '', lyrics)

    if (genre in classes):
      if (genre not in genres):
        genres[genre] = (i, [])
        i += 1

      genres[genre][1].append(lyrics)
      
data_text = []
data_label = []

for g in genres:
  lyrics = genres[g][1]
  data_text += lyrics[:23000]
  data_label += [genres[g][0]]*23000

#To use fastText, RCNN:
df = pd.DataFrame({"text": data_text, "label": data_label})
df.to_pickle("data/df_4.pkl")

#To use simple network:
vec = TfidfVectorizer()
X = vec.fit_transform(data_text)
#print(X.shape)

out = "data/vectorizer_balanced_4.sav"
out_label = "data/labels_balanced_4.sav"
pickle.dump(X, open(out, "wb"))
pickle.dump(data_label, open(out_label, "wb"))
