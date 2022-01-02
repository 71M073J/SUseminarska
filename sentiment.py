# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import html.parser as ht
import pickle

ps = PorterStemmer()


def get_stem_dict(folder="aclImdb_v1/aclImdb/train", fr="pos"):
    worddict = {}
    stemdict = {}
    dire = f"./{folder}/{fr}/"
    words = 0
    for i, file in enumerate(os.listdir(dire)):
        with open(dire + file, 'r', encoding="utf8") as f:
            for line in f:
                # test = "ASDF./<<<"
                line = line.replace("<br />", "").translate({ord(c): " " for c in './<>,\\\"()!?'}).lower()
                # line = line.strip("./<>,").lower()
                # line = line.lower()
                col = line.split()
                le = len(col)
                for word in col:
                    d = 1 / le
                    words += d
                    if word in worddict:
                        worddict[word] += d
                    else:
                        worddict[word] = d
            # print(worddict)
        if i % 1000 == 0:
            print(i)
        # break
    stop_words = set(stopwords.words('english'))
    # stop_words.add("/>")
    stop_words = stop_words.union(['-', 'br', "i'm", "he'", "i'v", "&"])
    # print(worddict)
    worddict2 = {}
    stemmedwords = 0
    for k, v in worddict.items():
        if k not in stop_words:
            worddict2[k] = v
            stemmedwords += v

    # print(worddict)
    for word in worddict2:
        stem = ps.stem(word)
        if stem in stemdict:
            stemdict[stem] += worddict2[word]
        else:
            stemdict[stem] = worddict2[word]
    # print(stemdict)
    print(f"we have had {words} word score before stemming and {stemmedwords} word score after")
    return stemdict


def sentimentScore(sent):
    folder = "aclImdb_v1/aclImdb/train"
    fr = "pos"
    dire = f"./{folder}/{fr}/"
    for i, file in enumerate(os.listdir(dire)):
        with open(dire + file, 'r', encoding="utf8") as f:
            score = 0
            for line in f:
                # test = "ASDF./<<<"
                line = line.replace("<br />", "").translate({ord(c): " " for c in './<>,\\\"()!?'}).lower()
                # line = line.strip("./<>,").lower()
                # line = line.lower()
                col = line.split()
                le = len(col)
                for word in col:
                    stem = ps.stem(word)
                    if stem in sent:
                        score += sent[stem]
            print(score)


def getSentiment(stemdictpos={}, stemdictneg={}, cutoff=1024):
    diff = {}
    print("setting positives...")
    for i, pos in enumerate(stemdictpos):
        if i % 1000000 == 0: print(i)
        # if pos in stemdictneg:
        if pos in diff:
            diff[pos] += stemdictpos[pos]
        else:
            diff[pos] = stemdictpos[pos]
    print("setting negatives...")
    for i, neg in enumerate(stemdictneg):
        if i % 1000000 == 0: print(i)
        # if neg in stemdictpos:
        if neg in diff:
            diff[neg] -= stemdictneg[neg]
        else:
            diff[neg] = -stemdictneg[neg]
    num = len(diff) - 1
    print("calculating cutoff")
    sentiment = [x for i, x in enumerate(sorted([(k, diff[k]) for k in diff], key=lambda x: x[1]))
                 if (i < (cutoff // 2) or (i + (cutoff // 2)) > num)]
    print(len(sentiment), sentiment)
    return sentiment


def get_sopojavitve(folder="aclImdb_v1/aclImdb/"):
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(['-', 'br', "i'm", "he'", "i'v", "&"])
    for fr in ["pos", "neg"]:
        if os.path.exists(f'sopojavitve{fr}.pkl'):
            continue
        sopojavitve = {}
        for case in ["train"]:
            dire = f"./{folder}/{case}/{fr}/"
            for i, file in enumerate(os.listdir(dire)):
                with open(dire + file, 'r', encoding="utf8") as f:
                    for line in f:
                        line = line.replace("<br />", "").translate({ord(c): " " for c in './<>,\\\"()!?'}).lower()
                        col = [ps.stem(word) for word in line.split() if word not in stop_words]
                        le = len(col)
                        for i2 in range(le):
                            for i3 in range(i2 + 1, min(i2 + 20, le)):

                                stem1 = col[i2]
                                stem2 = col[i3]
                                # distscore = 2 ** (i3 - i2)
                                distscore = (i3 - i2)
                                # print(stem1, stem2)

                                if stem1 < stem2:
                                    if (stem1, stem2) in sopojavitve:
                                        sopojavitve[(stem1, stem2)] += 1 / distscore
                                    else:
                                        sopojavitve[(stem1, stem2)] = 1 / distscore
                                else:
                                    if (stem2, stem1) in sopojavitve:
                                        sopojavitve[(stem2, stem1)] += 1 / distscore
                                    else:
                                        sopojavitve[(stem2, stem1)] = 1 / distscore
                # break
                if i % 50000 == 0: print(i)
        print(f"writing '{fr}' pickle")
        with open(f'sopojavitve{fr}.pkl', 'wb') as f:
            pickle.dump(sopojavitve, f)


def getSingleSent():
    print("getting pos")
    stemdictpos = get_stem_dict(fr="pos")
    # print(sorted([(k, stemdictpos[k]) for k in stemdictpos], key=lambda x: x[1]))
    print("getting neg")
    stemdictneg = get_stem_dict(fr="neg")
    # print(sorted([(k, stemdictneg[k]) for k in stemdictneg], key=lambda x: x[1]))


def get_pair_sentiment(cutoff=1024):
    fr = "pos"
    # read
    print("reading pos")
    with open(f'sopojavitve{fr}.pkl', 'rb') as f:
        datapos = pickle.load(f)
    fr = "neg"
    print("reading neg")
    with open(f'sopojavitve{fr}.pkl', 'rb') as f:
        dataneg = pickle.load(f)
    return getSentiment(datapos, dataneg, cutoff=cutoff)


def savePairSent(cutoff=1024):
    if os.path.exists(f"pairSentiment{cutoff}.pkl"):
        return
    sent = get_pair_sentiment(cutoff)  # OKAY this will be the input vector
    sent = [x[0] for x in sent]
    #print("new values saved:", sent)
    with open(f'pairSentiment{cutoff}.pkl', 'wb') as f:
        pickle.dump(sent, f)
