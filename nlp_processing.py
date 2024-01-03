import gensim
from gensim.test.utils import common_texts
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader as downloader

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
    for i in range(start,no_of_files):
        file_path = file_paths[i]
        print(f"Reading file: {file_path}")
        temp_df = pd.read_json(file_path)
        df = pd.concat([df,temp_df],axis=1)
    return df

# training model on common_texts, questions, correct answers, incorrect answers and reddit answers
def approach_1(df,col):
    correct_answers = pd.Series(df[col]['correct_answers']).apply(lambda x: x.lower().split())
    incorrect_answers = pd.Series(df[col]['incorrect_answers']).apply(lambda y: y.lower().split())
    reddit_answers = pd.Series(df[col]['answers']).apply(lambda z: z.lower().split())
    question = df[col]['question'][0].lower()
    sentences = [question,*correct_answers,*incorrect_answers,*reddit_answers,*common_texts]
    model = gensim.models.Word2Vec(workers=8, min_count=10, window=10, vector_size=300)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=model.corpus_count, epochs=10)
    return model, correct_answers, incorrect_answers, reddit_answers

# using pretrained model
def approach_2(df,col):
    model = downloader.load("glove-wiki-gigaword-300")
    correct_answers = pd.Series(df[col]['correct_answers']).apply(lambda x: model.get_mean_vector(x.lower().split()))
    incorrect_answers = pd.Series(df[col]['incorrect_answers']).apply(lambda y: model.get_mean_vector(y.lower().split()))
    # reddit answers are already in lowercase
    reddit_answers = pd.Series(df[col]['answers']).apply(lambda z: model.get_mean_vector(z.split()))
    return model, correct_answers, incorrect_answers, reddit_answers

def cosine_similarity(correct_answer, reddit_answer):
    return np.dot(correct_answer, reddit_answer) / (np.linalg.norm(correct_answer) * np.linalg.norm(reddit_answer))

def similarity_criteria(correctness,incorrectness,threshold=0.7):
    return 1 if sum([x > threshold for x in correctness]) > sum([y > threshold for y in incorrectness]) else 0

def nlp_processing():
    
    no_of_files = 2
    df = load_dataset(no_of_files)
    for question in df.columns:
        model, correct_answers, incorrect_answers, reddit_answers = approach_2(df,question)
        for answer in reddit_answers:
            # check similarity
            correctness = [model.wv.similarity(answer,correct_ans) for correct_ans in correct_answers]
            incorrectness = [model.wv.similarity(answer,incorrect_ans) for incorrect_ans in incorrect_answers]
            answer_value = similarity_criteria(correctness,incorrectness)
            print(answer_value)

if __name__ == "__main__":
    nlp_processing()