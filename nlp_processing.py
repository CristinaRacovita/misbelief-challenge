import gensim
from gensim.test.utils import common_texts
import numpy as np
from InstructorEmbedding import INSTRUCTOR
import pandas as pd
import os
import torch
import gensim.downloader as downloader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity
from flair.embeddings import DocumentRNNEmbeddings, WordEmbeddings, TransformerDocumentEmbeddings, StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence
import json

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

def approach_3(df,col):
    '''
    The sentences to be embedded should be in the format of 
    [["instruction prompt 0", "text to be embedded 0], ["instruction prompt 1", "text to be embedded 1], ...]
    '''
    '''
    Model from: https://huggingface.co/hkunlp/instructor-large
    '''
    model = INSTRUCTOR('hkunlp/instructor-large')
    correct_answers = pd.Series(df[col]['correct_answers'])
    incorrect_answers = pd.Series(df[col]['incorrect_answers'])
    reddit_answers = pd.Series(df[col]['answers'])
    
    correct_answers = correct_answers.apply(lambda x: model.encode(["Represent the correct answer:",x.lower()]))
    incorrect_answers = incorrect_answers.apply(lambda y: model.encode(["Represent the incorrect answer:",y.lower()]))
    # reddit answers are already in lowercase
    reddit_answers = reddit_answers.apply(lambda z: model.encode(["Represent the reddit answer:",z]))
    return correct_answers,incorrect_answers,reddit_answers

def approach_4(df,col):
    r'''Installed at: C:\Users\saran\AppData\Local\Temp\tmpwa4zvu1f'''
    flair_embedding_forward = FlairEmbeddings('news-forward',fine_tune=True)
    flair_embedding_backward = FlairEmbeddings('news-backward',fine_tune=True)
    glove_embedding = WordEmbeddings('glove') # [ar,glove]

    document_embeddings = DocumentRNNEmbeddings([glove_embedding],rnn_type='lstm')
    # could possibly add more models, similarity scores tend to fall as more models are added
    model = StackedEmbeddings([
    document_embeddings,
    flair_embedding_backward,
    flair_embedding_forward
    ])
    correct_answers = [Sentence(x.lower()) for x in df[col]['correct_answers']]
    incorrect_answers = [Sentence(y.lower()) for y in pd.Series(df[col]['incorrect_answers'])]
    reddit_answers = [Sentence(z) for z in pd.Series(df[col]['answers'])]
    model.embed([*correct_answers,*incorrect_answers,*reddit_answers])

    return correct_answers, incorrect_answers, reddit_answers

def approach_5(df,col):
    # TF-IDF approach
    correct_answers = pd.Series(df[col]['correct_answers']).apply(lambda x: x.lower())
    incorrect_answers = pd.Series(df[col]['incorrect_answers']).apply(lambda y: y.lower())
    reddit_answers = pd.Series(df[col]['answers'])
    # correct_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # incorrect_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # correct_matrix = correct_vectorizer.fit_transform(correct_answers).toarray()
    # incorrect_matrix = incorrect_vectorizer.fit_transform(incorrect_answers).toarray()
    all_answers = pd.concat([correct_answers,incorrect_answers])
    matrix = vectorizer.fit_transform(all_answers)
    correct_matrix = matrix[:len(df[col]['correct_answers']), :]
    incorrect_matrix = matrix[len(df[col]['correct_answers']):, :]
    # correctness = reddit_answers.apply(lambda x: correct_vectorizer.transform([x]).toarray())
    # incorrectness = reddit_answers.apply(lambda y: incorrect_vectorizer.transform([y]).toarray())
    reddit_matrix = vectorizer.transform(reddit_answers)
    return correct_matrix, incorrect_matrix, reddit_matrix #correctness, incorrectness

def similarity_criteria(correctness,incorrectness):
    return 1 if np.max(correctness) > np.max(incorrectness) else 0

def series_to_json(series:pd.Series,attribute_name,attribute_val):
    """Converts a Pandas Series object to a JSON object and adds an attribute to it"""
    json_object = {}
    for index, value in series.items():
        json_object[index] = value
    json_object[attribute_name] = attribute_val
    return json_object

def nlp_processing(batch_start,batch_end): 
    file_path = f"misbelief-challenge/answers/answers_{batch_start}-{batch_end}.json"
    df = pd.read_json(file_path)
    data_store = {}
    #cos = CosineSimilarity(dim=0)
    for question in df.columns:
        answer_vector = []
        correct_matrix, incorrect_matrix, reddit_matrix = approach_5(df,question)
        for i in range(reddit_matrix.shape[0]):
            # check similarity
            # correct_vector = reddit_correct[i]
            # incorrect_vector = reddit_incorrect[i]
            reddit_ans = reddit_matrix[i]
            correctness = [cosine_similarity(correct_ans,reddit_ans)[0] for correct_ans in correct_matrix]
            incorrectness = [cosine_similarity(incorrect_ans,reddit_ans)[0] for incorrect_ans in incorrect_matrix]
            answer_vector.append(similarity_criteria(correctness,incorrectness))
        #df[question]['predicted_answers'] = answer_vector
        data_store[question] = series_to_json(df[question],"predicted_answers",answer_vector)
    with open(file_path, 'w') as f:
        json.dump(data_store, f, indent=4)
            

def torch_similarity(sentence,sentence2,cos):
    shape = sentence[0].embedding.shape
    avg_1 = torch.zeros(shape)
    avg_2 = torch.zeros(shape)
    for token in  sentence:
        avg_1 += token.embedding
    for token in sentence2:
        avg_2 += token.embedding
    avg_1 /= shape[0]
    avg_2 /= shape[0]
    
    return round(cos(avg_1,avg_2).item(),3)

if __name__ == "__main__":
    nlp_processing(20,40)



