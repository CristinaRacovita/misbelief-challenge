import numpy as np
import pandas as pd
from nlp_processing import approach_1, approach_2, approach_3, approach_4, approach_5, torch_similarity, similarity_criteria
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score
from flair.embeddings import FlairEmbeddings,DocumentRNNEmbeddings, WordEmbeddings, StackedEmbeddings
import gensim.downloader as downloader
import json
from InstructorEmbedding import INSTRUCTOR
from torch.nn import CosineSimilarity

def label_answers(file_path):
    evaluation_set = pd.read_json(file_path)
    data_store = {}
    for question_index in evaluation_set:
        answer_vec = []
        question = evaluation_set[question_index]['question']
        answers = evaluation_set[question_index]['answers']
        correct_answers = evaluation_set[question_index]['correct_answers']
        print(f"Correct answers: {correct_answers}")
        for ans in answers:
            print(f"For Question: {question}")
            print(f"Answer: {ans}")
            answer_vec.append(int(input(f"Label: ")))
        print("\n----------------End of Question ---------------\n")
        data_store[question_index] =  json.loads(evaluation_set[question_index].to_json())
        data_store[question_index]['true_labels'] = answer_vec
    with open(r"misbelief-challenge\evaluation\answers_evaluation.json","w") as file:
        json.dump(data_store,file,indent=4)

def accuracy(true_labels,predicted_labels):
    num_correct = sum(true_labels)
    num_incorrect = len(true_labels) - num_correct
    if num_correct != 0:
        correctness_accuracy = np.sum(np.logical_and(true_labels,predicted_labels))/num_correct
    else:
        correctness_accuracy = 0
    incorrectness_accuracy = np.sum(np.logical_not(np.logical_or(true_labels,predicted_labels)))/num_incorrect
    overall_accuracy = np.sum(np.logical_not(np.logical_xor(true_labels,predicted_labels)))/len(true_labels)
    return {'correctness_accuracy':correctness_accuracy,
            'incorrectness_accuracy':incorrectness_accuracy,
            'overall_accuracy':overall_accuracy}
    

def run_approach_1(df,col,true_value):
    '''
    Approach uses Word2Vec model
    '''
    model, correct_answers, incorrect_answers, reddit_answers = approach_1(df,col)
    answer_vector = []
    for ans in reddit_answers:
        correctness = [cosine_similarity(model.wv.similarity(ans,correct_ans)) for correct_ans in correct_answers]
        incorrectness = [cosine_similarity(model.wv.similarity(ans,incorrect_ans)) for incorrect_ans in incorrect_answers]
        answer_vector.append(similarity_criteria(correctness,incorrectness))
    return accuracy(true_value,answer_vector)

def run_approach_2(df,col,true_value,pretrained_model):
    '''
    Approach uses a pretrainedWord2Vec model
    '''
    correct_answers, incorrect_answers, reddit_answers = approach_2(pretrained_model,df,col)
    # correct_answers = df[col]['correct_answers'] + [df[col]['best_answer']]
    # incorrect_answers = df[col]['incorrect_answers']
    # reddit_answers = df[col]['answers']
    answer_vector = []
    for answer in reddit_answers:
        # check similarity
        correctness = [cosine_similarity(answer,correct_ans) for correct_ans in correct_answers]
        incorrectness = [cosine_similarity(answer,incorrect_ans) for incorrect_ans in incorrect_answers]
        answer_vector.append(similarity_criteria(correctness,incorrectness))
    return accuracy(true_value,answer_vector)

def run_approach_3(df,col,true_value,pretrained_model):
    correct_answers,incorrect_answers,reddit_answers = approach_3(df,col,pretrained_model)
    answer_vector = []
    for answer in reddit_answers:
        correctness = [np.mean(cosine_similarity(answer,correct_ans)) for correct_ans in correct_answers]
        incorrectness = [np.mean(cosine_similarity(answer,incorrect_ans)) for incorrect_ans in incorrect_answers]
        answer_vector.append(similarity_criteria(correctness,incorrectness))
    return accuracy(true_value,answer_vector)

def run_approach_4(df,col,true_value):
    cos = CosineSimilarity(dim=0)
    correct_answers, incorrect_answers, reddit_answers = approach_4(df,col)
    answer_vector = []
    for answer in reddit_answers:
        correctness = [torch_similarity(answer,correct_ans,cos) for correct_ans in correct_answers]
        incorrectness = [torch_similarity(answer,incorrect_ans,cos) for incorrect_ans in incorrect_answers]
        answer_vector.append(similarity_criteria(correctness,incorrectness))
    return accuracy(true_value,answer_vector)

def run_approach_5(df,col,true_value):
    '''
    Approach uses TfidfVectorizer
    '''
    correct_matrix, incorrect_matrix, reddit_matrix = approach_5(df,col)
    answer_vector = []
    for i in range(reddit_matrix.shape[0]):
        reddit_ans = reddit_matrix[i]
        correctness = [cosine_similarity(correct_ans,reddit_ans)[0] for correct_ans in correct_matrix]
        incorrectness = [cosine_similarity(incorrect_ans,reddit_ans)[0] for incorrect_ans in incorrect_matrix]
        answer_vector.append(similarity_criteria(correctness,incorrectness))
    return accuracy(true_value,answer_vector)

def get_approach_4_model():
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
    return model

def run_approach(approach: str, df, col, true_value, model=None):
    if approach == "approach_2":
        return run_approach_2(df, col, true_value, model)
    elif approach == "approach_3":
        return run_approach_3(df, col, true_value, model)
    elif approach == "approach_4":
        return run_approach_4(df, col, true_value, model)
    elif approach == "approach_5":
        return run_approach_5(df, col, true_value)
    else:
        raise ValueError("Invalid approach")

if __name__ == "__main__":
    evaluation_set = pd.read_json(r"misbelief-challenge\evaluation\answers_evaluation.json")
    #approach_2_model = downloader.load("glove-wiki-gigaword-300")
    #approach_3_model = INSTRUCTOR('hkunlp/instructor-large')
    #approach_4_model = get_approach_4_model()

    # data_store = {
    # approach: {
    #     question_index: run_approach(approach,evaluation_set, question_index, evaluation_set[question_index]['true_labels'], model)
    #     for question_index in evaluation_set.columns if len(evaluation_set[question_index]['answers']) != 0
    # }
    # for approach, model in [
    #     ('approach_2', approach_2_model),
    #     ('approach_3', approach_3_model),
    #     ('approach_4', approach_4_model),
    #     ('approach_5', None) 
    # ]
    # }
    # with open(r"misbelief-challenge\evaluation\evaluation_scores.json") as file:
    #     json.dump(data_store,file,indent=4)
    approach =  'approach_4'
    data_store = {approach:{}}
    for question_index in evaluation_set.columns:
        if len(evaluation_set[question_index]['answers']) != 0:
            true_value = evaluation_set[question_index]['true_labels']
            # store the accuracy values from each approach
            # accuracy_dict_2 = run_approach_2(evaluation_set,question_index,true_value,approach_2_model)
            # accuracy_dict_3 = run_approach_3(evaluation_set,question_index,true_value,approach_3_model)
            accuracy_dict = run_approach_4(evaluation_set,question_index,true_value)
            data_store[approach][question_index] = accuracy_dict
            #accuracy_dict_5 = run_approach_5(evaluation_set,question_index,true_value)
    with open(r"misbelief-challenge\evaluation\evaluation_approach_4.json",'w') as file:
        json.dump(data_store,file,indent=4)
            

    
