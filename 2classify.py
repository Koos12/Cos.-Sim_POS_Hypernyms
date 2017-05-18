import sklearn
import os
import sys
import re
import pandas
import numpy as np
import scipy
import unicodedata
import nltk 
nltk.download()
from nltk.corpus import wordnet as wn
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import StringIO
from os import path
import cPickle as pickle
import sklearn.preprocessing as pp
import string
import math
from collections import Counter
from decimal import Decimal
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def put(self,key,data):
      hashvalue = self.hashfunction(key,len(self.slots))

      if self.slots[hashvalue] == None:
        self.slots[hashvalue] = key
        self.data[hashvalue] = data
      else:
        if self.slots[hashvalue] == key:
          self.data[hashvalue] = data  #replace
        else:
          nextslot = self.rehash(hashvalue,len(self.slots))
          while self.slots[nextslot] != None and \
                          self.slots[nextslot] != key:
            nextslot = self.rehash(nextslot,len(self.slots))

          if self.slots[nextslot] == None:
            self.slots[nextslot]=key
            self.data[nextslot]=data
          else:
            self.data[nextslot] = data #replace

    def hashfunction(self,key,size):
         return key%size

    def rehash(self,oldhash,size):
        return (oldhash+1)%size

    def get(self,key):
      startslot = self.hashfunction(key,len(self.slots))

      data = None
      stop = False
      found = False
      position = startslot
      while self.slots[position] != None and  \
                           not found and not stop:
         if self.slots[position] == key:
           found = True
           data = self.data[position]
         else:
           position=self.rehash(position,len(self.slots))
           if position == startslot:
               stop = True
      return data

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)


def isNotBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return True
    #myString is None OR myString is empty or blank
    else:
         return False

def definition():
	global T   #Here you give a value to bilbodog (even None)
	T=HashTable()



#WORD = re.compile(r'\w+')
f = open("/Users/Koos/Anaconda2/multipleCosineTriples")
tripleWords = []
lines = f.readlines()
for line in lines:
	#print line
	line2=line.split("|")
	line2Length=len(line2)
	for i,a in enumerate(line2):
		if (isNotBlank(a)):
			T=HashTable()
			T[i]=a
			tripleWord=T[i]
			tripleWords.append(tripleWord.lower())
#for element in tripleWords:
#	print element
				
				
	
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator



def text_to_vector(text, tripleWords):
		gramWords=[]
		WORD = re.compile(r'\w+')
		words = WORD.findall(text)
		#print Counter(words)
		c=Counter(words)
		mylist=list(c.elements())
		myset = set(mylist)
		mynewlist = list(myset)
		length=len(mynewlist)
		#for element in mynewlist:
			#print element
		#print "******"
		for el in tripleWords:
			if el in myset:
				#print "Gram Word found!!"
				#print el
				gramWords.append(el.lower())
		#print Counter(gramWords)
		return Counter(gramWords)
		
		
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
count = 0
totalTxt=""
full_file_paths = get_filepaths("/Python27/training/C01")
allLemmas=""

for file in full_file_paths:
#f = os.path.basename(file)
    note = open(file, "r")
    txt=note.read()
    totalTxt+=txt
    count+=1
    note.close() 
for c in string.punctuation:
     totalTxt= totalTxt.replace(c,"")   
totalTxt=re.sub(r'\s+',' ',totalTxt)
totalTxt='\''+totalTxt+'.\''
#print totalTxt1
tokens = word_tokenize(totalTxt)
tagged_tokens = pos_tag(tokens)
for tagged_token in tagged_tokens:
         word = tagged_token[0]
         for c in string.punctuation:
             word= word.replace(c,"") 
         word_pos = tagged_token[1]
         lemma = wordnet_lemmatizer.lemmatize(word)
         lemma= lemma+'/'+word_pos         
         allLemmas=lemma+' '+allLemmas
allLemmas='\''+allLemmas+'.\''         
allLemmas=re.sub(r'\s+',' ',allLemmas)
#print allLemmas

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
count = 0
totalTxt1=""
full_file_paths = get_filepaths("/Python27/training/C02")


allLemmas1=""

for file in full_file_paths:
#f = os.path.basename(file)
    note = open(file, "r")
    txt=note.read()
    totalTxt1+=txt
    count+=1
    note.close() 
for c in string.punctuation:
     totalTxt1= totalTxt1.replace(c,"")   
totalTxt1=re.sub(r'\s+',' ',totalTxt1)
totalTxt1='\''+totalTxt1+'.\''
#print totalTxt1
tokens = word_tokenize(totalTxt1)
tagged_tokens = pos_tag(tokens)
for tagged_token in tagged_tokens:
         word = tagged_token[0]
         for c in string.punctuation:
             word= word.replace(c,"") 
         word_pos = tagged_token[1]
         lemma = wordnet_lemmatizer.lemmatize(word)
         lemma= lemma+'/'+word_pos         
         allLemmas1=lemma+' '+allLemmas1
allLemmas1='\''+allLemmas1+'.\''         
allLemmas1=re.sub(r'\s+',' ',allLemmas1)
#print allLemmas


path = "/Python27/test/C01"


def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
count = 0

full_file_paths = get_filepaths("/Python27/test/C01")

correctlyClassified=0
incorrectlyClassified=0

fileList=[]

textLemmas=""
lemma=""
count=-1
for file in full_file_paths:
    f = os.path.basename(file)
    note = open(file, "r")
    text=note.read()
    #print f
    H=HashTable()
    H[count]=text
    #print count
    #print H[count]
    if count < 400:
        count+=1
        for c in string.punctuation:
           text= text.replace(c,"")
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        for tagged_token in tagged_tokens:
            word = tagged_token[0]
            for c in string.punctuation:
                word= word.replace(c,"") 
            word_pos = tagged_token[1]
            if ((word_pos=="JJ") or (word_pos=="NN") or (word_pos=="VB")
                or (word_pos=="VBD") or (word_pos=="RB") or (word_pos=="NP")
                or (word_pos=="NNS") or (word_pos=="NP") or (word_pos=="BER")
                or (word_pos=="BEZ") or (word_pos=="MD") or (word_pos=="PRP")
                or (word_pos=="RB") or (word_pos=="VBG") or(word_pos=="VBN")
                or  (word_pos=="VBP") or (word_pos=="RBR") or (word_pos=="RBS")
                or (word_pos=="JJR") or (word_pos=="JJS") or (word_pos=="PDT")
                or (word_pos=="DT") or (word_pos=="IN") or (word_pos=="PRP")
                or (word_pos=="PRP$") or (word_pos=="PRP") or (word_pos=="VBZ")
                or (word_pos=="WRB") or (word_pos=="PRP$") or (word_pos=="TO")
                or (word_pos=="WP") or (word_pos=="WRB")):
                lemma = wordnet_lemmatizer.lemmatize(word)
                #print lemma
                textLemmas=textLemmas+' '+lemma
    textLemmas='\''+textLemmas+' .\'\n'
    textLemmas=re.sub(r'\'\s+','\'',textLemmas)
    holdLemmas=textLemmas
    if (len(textLemmas)!=3):
         print textLemmas
    textLemmas=""
    vector1 = text_to_vector(allLemmas, tripleWords)
    vector2 = text_to_vector(holdLemmas,tripleWords)
    #print vector1
    cosine = get_cosine(vector1, vector2)
    if (cosine>0):
        print 'Cosine', cosine
    vector1 = text_to_vector(allLemmas1, tripleWords )
    #print vector1
    cosine1 = get_cosine(vector1, vector2)
    if (cosine1>0):
        print 'Cosine1', cosine1
    if ((cosine>cosine1)&(cosine!=0)&(cosine1!=0)):
        print "This document was incorrectly classified as belonging in training set C02 (HIV infections)"
        incorrectlyClassified+=1
    elif ((cosine!=0)&(cosine1!=0)):
        print "This document was correctly classified as belonging in training set C01 (Tuberculosis)"
        correctlyClassified+=1
        print "Number of correctly classified documents (parser): ",correctlyClassified
        print f
        fileList.append(f)
    note.close() 

testfilesPOSTuberculosis=open('/Python27/testfPosTub', 'w')
for item in fileList:
    testfilesPOSTuberculosis.write("%s\n" % item)

