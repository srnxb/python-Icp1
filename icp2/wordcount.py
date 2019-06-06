file = open('text.txt')
wordcount = {}

for word in file.read().split():
    if word.lower() not in wordcount:
        wordcount[word.lower()] = 1
    else:
        wordcount[word.lower()] += 1

for k, v in wordcount.items():
    print (k, v)
