import streamlit as st
import pandas as pd
import numpy as np
import itertools
import datetime

def convert_bin_to_time_interval(bin_range):
    bin_start, bin_end = bin_range
    time_interval = str(int((bin_end - bin_start) // 600)) + " minutes"
    return time_interval

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

    # number of empty timeStamps
    checking_timeStamps_df = df.explode(['timeStamps'])
    st.write(f"Number of empty timeStamp values: {(checking_timeStamps_df['timeStamps'] == '').sum()}")

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

    st.header(f"Incorrect answers: By category")
    incorrect_categories_selected = st.multiselect(label='Select categories for incorrect answers',options=df['category'].unique())
    if incorrect_categories_selected != []:
        temp_df = df.groupby("category")['predicted_answers'].sum().apply(lambda x: len(x)-sum(x)).to_frame().T[incorrect_categories_selected]
        st.bar_chart(temp_df.T)
    else:
        st.error("select a category")

    st.header(f"Correct answers: By category")
    correct_categories_selected = st.multiselect(label='Select categories for correct answers',options=df['category'].unique())
    if correct_categories_selected != []:
        temp_df = df.groupby("category")['predicted_answers'].sum().apply(lambda x: sum(x)).to_frame().T[correct_categories_selected]
        st.bar_chart(temp_df.T)
    else:
        st.error("select a category")

    # Popular misbelief based on timestamp year by category
    st.header("Popular misbelief based on timestamp year by category")
    temp_df = df.explode(['predicted_answers','timeStamps'])
    replacement_year = pd.to_datetime("2020-10-27T14:13:12.111Z").year # need a better way to do this 
    temp_df['timeStamps'] = temp_df['timeStamps'].apply(lambda x: pd.to_datetime(x).year if x != "" else replacement_year)
    # group by timeStamp
    grouped_temp_df = temp_df.groupby(['timeStamps','category'])['predicted_answers'].apply(lambda x: (x == 0).sum()).to_frame()
    bar_data_2 = grouped_temp_df.pivot_table(index='timeStamps', columns='category', values='predicted_answers', fill_value=0)
    st.bar_chart(bar_data_2,color=['#ffaa00',"#ffaa0088"])
    # time_data = np.array([time_delta_data,all_answers]).T
    # time_df = pd.DataFrame(data=time_data,columns=['time_delta','answers_vector'])
    # bins = pd.cut(time_df['time_delta'], range(int(min(time_df['time_delta'])), int(max(time_df['time_delta'])) + 1, int(interval.total_seconds())))
    # grouped_data = time_df.groupby(bins).count()
    # grouped_data['time_intervals'] = range(1,len(grouped_data)+1) # could have better labelling
    # # too many 0 entries
    # st.line_chart(data=grouped_data,x='time_intervals',y='answers_vector',color="#FF0000")



        
if __name__ == "__main__":
    visual_processing()
