import streamlit as st
import pandas as pd
import numpy as np

st.header("Most genius project ever")
'''
    Business questions: \n
    How helpful are forums such as Quora and Reddit based on the number of accurate answers?\n
    Which questions have the most incorrect answers?\n
    How long do people post questions after the post has been posted?\n
'''
@st.cache_data
def load_data(no_of_files):
    file_paths = [r"misbelief-challenge/answers/answers_0-20.json",
                 r"misbelief-challenge/answers/answers_20-40.json"]
    # get timeStamp data and the answers vector for each question
    df = pd.concat([pd.read_json(f) for f in file_paths[:no_of_files]],axis=1).T
    return df

def visual_processing():
    no_of_files = 2
    df = load_data(no_of_files)
    st.write("Head of main dataframe")
    st.write(df.head())
    dimensions = pd.DataFrame(
        data=[["Time"],["Question Category"],["Location"]],
        columns = ["Dimensions"]  
        )
    facts = pd.DataFrame(
        data=[["Number of Answers"],["Number of correct/incorrect answers"]],
        columns = ["Facts"]  
        )
    col1, col2 = st.columns(2)
    # Place the dimensions table in the first column
    col1.header("Dimensions")
    col1.table(dimensions)
    # Place the facts table in the second column
    col2.header("Facts")
    col2.table(facts)
    st.header(f"Answers distribution (in {no_of_files} files)")
    correct_answers = pd.DataFrame(df['predicted_answers'].apply(lambda x: sum(x)).values,
                                   index=df['question'],
                                   columns=["correct answers"]
                                   )
    incorrect_answers = pd.DataFrame(df['predicted_answers'].apply(lambda x: len(x)-sum(x)).values,
                                    index=df['question'],
                                    columns=["incorrect answers"]
                                    )
    bar_data_1 = pd.concat([correct_answers,incorrect_answers],axis=1)
    st.bar_chart(data=bar_data_1,color=['#ffaa00',"#ffaa0088"])

    st.header("Incorrect answers: By category")
    st.bar_chart(df.groupby("category")['predicted_answers'].sum().apply(lambda x: len(x)-sum(x)))
if __name__ == "__main__":
    visual_processing()
