import gensim
from gensim.test.utils import common_texts
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

def count_files(directory):
    total_files = 0
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = root + "/" + file
            file_paths.append(relative_path)
            total_files += 1
    return total_files, sorted(file_paths)

def load_dataset(no_of_files,start=0):
    # read the json files batch wise
    total_files,file_paths = count_files(r"./misbelief-challenge/answers")
    if start + no_of_files > total_files:
        raise Exception("Number of files requested exceeds the number of files that exist")
    df = pd.DataFrame()
    print(file_paths)
    for i in range(start,no_of_files):
        file_path = file_paths[i]
        print(f"Reading file: {file_path}")
        temp_df = pd.read_json(file_path)
        df = pd.concat([df,temp_df],axis=1)
    return df
# Load the dataset of questions and answers
def nlp_processing():
    no_of_files = 2
    df = load_dataset(no_of_files)
    vectorizer = TfidfVectorizer()
    # Train Word2Vec models for each question
    question_word2vec_models = []
    for col in df.columns:
        question = df[col]['question']
        question_word2vec = gensim.models.Word2Vec(sentences=[question], size=100, min_count=2)
        question_word2vec_models.append(question_word2vec)

    for col in zip(df.columns,question_word2vec_models):
        correct_answers = df[col]['correct_answers']
        incorrect_answers = df[col]['incorrect_answers']
        reddit_answers = df[col]['answers']

def save_model():
    model = gensim.models.Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")

def load_model():
    model = gensim.models.Word2Vec.load(r"misbelief-challenge/word2vec.model")
    print(model.wv['computer'])

if __name__ == "__main__":
    load_model()

    