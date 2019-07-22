#! /usr/bin/env python3

# use NLTK to summarize an input file

import nltk
import collections
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

#file = input("enter a text file to summary: ")

file = "sample.txt"

# for testing, read this file
#file = "test-text.txt"

# read contents of file
content = open(file).read()

# lemmatize all words in content
lemmatizer = WordNetLemmatizer()

# tokenize content into words, then lemmatize; save all lemmas
lemmatized = []
for word in word_tokenize(content.lower()):
    lemmatized.append(lemmatizer.lemmatize(word, pos='v'))

# tag all lemmas with their parts of speech, then remove all the
# verbs (POS='VB')
tagged = nltk.pos_tag(lemmatized)
no_verbs = []

# exclude these non-informative words from our top choices
exclude = ['the','a', 'of','that', 'and', '.']

for word_tuple in tagged:
    # get rid of verbs and non-informative words
    if word_tuple[1] != 'VB' and word_tuple[0] not in exclude:
        no_verbs.append(word_tuple[0])

# calculate word frequency of remaining words
WORD_COUNTS = collections.Counter(no_verbs)

# get top five words as tuple containing word and the number of times it was seen
top_words = WORD_COUNTS.most_common(5)
print("top words:", top_words)
print()

# go back through original text and keep sentences that have our
# most common words
for sentence in sent_tokenize(content):
    for top_word in top_words:
        if top_word[0] in sentence:
            print("Sentences with most common words:",sentence)
            break