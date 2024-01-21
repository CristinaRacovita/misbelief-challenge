import pandas as pd
import numpy as np
import os

def load_reddit_answers():
    df_list = []
    directory_path = r"misbelief-challenge\reddit_answers"
    for filename in os.listdir(directory_path):
        file_path = directory_path + "\\" + filename
        temp_df = pd.read_json(file_path)
        df_list.append(temp_df)
    return pd.concat(df_list,axis=1)

def load_quora_answers():
    df_list = []
    directory_path = r"misbelief-challenge\quora_answers"
    for filename in os.listdir(directory_path):
        file_path = directory_path + "\\" + filename
        temp_df = pd.read_json(file_path)
        df_list.append(temp_df)
    return pd.concat(df_list,axis=1)

def reshape_df(df,df_type) -> pd.DataFrame:
    T_reddit = df.transpose()
    features = T_reddit.columns
    T_reddit = T_reddit.reset_index()
    #print(T_reddit)
    melted_df = pd.melt(T_reddit, id_vars=['index'], value_vars=features)
    melted_df['json_object'] = melted_df.apply(lambda row: {row['variable']: row['value']}, axis=1)
    melted_df = melted_df.drop(['variable', 'value'], axis=1)
    result_series =  melted_df.groupby("index")["json_object"].agg(list)
    result_df = pd.DataFrame(data=result_series).reset_index()
    result_df.columns = ["questions",df_type]
    return result_df

def merge_datasets():
    reddit_df = load_reddit_answers()
    quora_df = load_quora_answers()
    #main_df = pd.read_json(r"misbelief-challenge\truthful_qa.json")
    # result_df.index = melted_df.index
    reddit_melted = reshape_df(reddit_df,"reddit")
    quora_melted = reshape_df(quora_df,"quora")
    reddit_melted['questions'] = reddit_melted['questions'].astype(str)
    quora_melted['questions'] = quora_melted['questions'].astype(str)
    merged_df = reddit_melted.merge(quora_melted,how="inner",on="questions")
    merged_df.to_json(r"misbelief-challenge\main_df.json",indent=2)

merge_datasets()
