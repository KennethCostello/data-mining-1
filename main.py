import time
import glob
import re
import numpy as np
from methods import *
from dataClass import DataSet
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer



start = time.time()

#https://github.com/snkim/AutomaticKeyphraseExtraction

#first create argparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", required = False, help="please specify filepath" )
parser.add_argument("-g", required = False, help="run target function" )
args = parser.parse_args()

root = "C:/Week5/Week5 Assignment/data"

def main():
    if args.g:
        filepath = args.g
    else:
        filepath = root

    dataSet = DataSet()

    # extracting the paths
    dataSet.df = extractFilePath(filepath)

    print(dataSet.df.head())
    #
    # path = dataSet.df.text_paths[2]
    # text = extractDocLines(path)
    #
    # dataSet.df['rawText'] = dataSet.df.text_paths.apply(extractDocLines)
    # dataSet.df['keywords'] = dataSet.df.key_paths.apply(extractKeywords)
    #
    # print(dataSet.df.head())
    #
    # #clean text
    # dataSet.df['vsm'] = dataSet.df.rawText.apply(cleanText)
    #
    # #apply tfidf
    # dataSet.model, dataSet.matrix = applyTFIDF(dataSet.df['vsm'])
    #
    #
    # dataSet.dict_index, dataSet.dict_values= exploreMatrix(dataSet.matrix, dataSet.model)
    #
    # bolsterNgrams(dataSet)
    #
    # # evaluate result
    # evaluateModel(dataSet)




if __name__ == "__main__":
    main()

print("\n"+ 40*"*")
print(time.time()-start)
print( 40*"*"+ "\n")
