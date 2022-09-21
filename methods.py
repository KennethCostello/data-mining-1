import glob
import re
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import pandas as pd
import json
from pathlib import Path
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))

def extractFilePath(filepath):
    path1 = '//**//**/*.json'
    path2 = '//**/*.xml'

    if Path.is_file(Path(path2)):
        keys = glob.glob(f'{filepath}//**/*.kwd')
        texts = glob.glob(f'{filepath}//**/*.xml')

        key_paths = []
        text_paths = []
        key_number_list = []

        for i in range(len(keys)):
            keypaths = keys[i].split("\\")
            value = keypaths[-1].split(".")[0]
            key_number_list.append(value)

        for i in range(len(texts)):
            textpath = texts[i].split("\\")
            value = textpath[-1].split(".")[0]
            if value in key_number_list:
                text_paths.append(texts[i])

        df1 = pd.DataFrame()

        df1['key_paths'] = keys
        df1['text_paths'] = text_paths

    else:
        jsonFiles = glob.glob(f'{filepath}//**//**/*.json')

        body_text_list = []
        paper_id = []
        for filepath in jsonFiles[:1000]:
            with open(filepath) as files:
                content = json.load(files)
                if content['body_text']:
                    text = content['body_text'][0]['text']
                    paper_id.append(content['paper_id'])
                    body_text_list.append(text)

        df1 = pd.DataFrame()

        df1['key_paths'] = paper_id
        df1['text_paths'] = body_text_list

    return df1

def extractDocLines(path):
    with codecs.open(path, "r", encoding="utf8", errors="ignore") as f:
        doc = f.read()
        doc = doc.split("\n")
        doc = "".join(doc)

        sectionArray = []
        sectionText = ""
        for result in re.findall("<SECTION(.*?)</SECTION>", doc):
            sectionArray.append(result)
            sectionText += result

        sectionHeaders = []
        for section in sectionArray:
            for result in re.findall('header=(.*?)>', section):
                sectionHeaders.append(result)

    return sectionText

def extractKeywords(path):
    with codecs.open(path, "r", encoding="utf8", errors="ignore") as f:
            doc = f.read()
            doc = doc.split("\n")
            doc = [x.lower() for x in doc if x]
    return doc

def cleanText(row):
    sent = []
    for term in row.split():
        term = re.sub('[^a-zA-Z]', " ", term.lower())
        sent.append(term)
    sent = [word for word in sent if word not in stop]

    return " ".join(sent)

def applyTFIDF(data):
    tfidf_vectoriser = TfidfVectorizer(max_df = 0.9, min_df = 0.1,
                                                ngram_range =(1,4))

    tfidf_matrix = tfidf_vectoriser.fit_transform(list(data))

    return tfidf_vectoriser, tfidf_matrix

def exploreMatrix(matrix, model):
    terms = model.vocabulary_

    dict_values = dict(zip(matrix.indices, matrix.data))

    dict_index = dict(zip(terms.values(), terms.keys()))

    return dict_index, dict_values

def evaluateModel(dataSet):
    precision_list = []
    recall_list = []
    fscore_list = []
    for i, row, in dataSet.df.iterrows():
        y_pred =  topCorpusTerms(dataSet.matrix, dataSet.dict_index, i)

        y_true = dataSet.df.keywords[i]

        precision, recall, fscore = evaluateResults(y_pred, y_true)

        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)

    print("precision: ", sum(precision_list)/len(precision_list)*100)
    print("recall: ", sum(recall_list)/len(recall_list)*100)
    print("fscore: ", sum(fscore_list)/len(fscore_list)*100)


def evaluateResults(y_pred, y_true):
    #determine correct
    correct = [1 for x in y_true if x in y_pred]
    correct = sum(correct)
    # precision
    try:
        precision = float(correct/len(y_pred))
    except:
        precision =  0
    # recall
    try:
        recall = float(correct/len(y_true))
    except:
        recall = 0
    #fscore
    try:
        fscore = 2*(precision*recall)/(precision+recall)
    except:
        fscore = 0

    return precision, recall, fscore


def topCorpusTerms(matrix, model_terms, row_id, top_n=10):
    row = np.squeeze(matrix[row_id].toarray())
    topn_ids = np.argsort(row)[::-1]
    top_terms = [model_terms[i] for i in topn_ids]

    return top_terms[:top_n]


def bolsterNgrams(dataSet):
    terms = dataSet.dict_index.values()
    lexemes = []
    for term in terms:
        if len(term.split()) > 1:
            lexemes.append(term)

    for i, row in dataSet.df.iterrows():
        for lex in lexemes:
            if lex in dataSet.df.vsm[i]:
                count = dataSet.df.vsm[i].count(lex)
                dataSet.matrix[i, dataSet.model.vocabulary_[lex]] *= count

    return lexemes
