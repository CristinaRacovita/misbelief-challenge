import streamlit as st
import pandas as pd
from evaluation_methods import evaluate_approaches
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6']

st.header("Misbeliefs Around the World")
st.markdown('''
    **We analysed the most popular misbelief answers from Quora and Reddit. 
    We used the TruthfulQA dataset, where we had top superstitions in the world and a web crawler to retrive people's answers from two known forum websites.**
''')
st.markdown('''
    **Business questions:**
    - How helpful are forums such as Quora and Reddit based on the number of accurate answers?\n
    - Which questions have the most incorrect answers?\n
    - How many categories are and how many questions do we have based on the category?\n
    - How many correct and incorrect answers do we have based on the location (country)?
''')
@st.cache_data
def load_data():
    file_path = r"misbelief-challenge\main_df_processed.json"
    # get timeStamp data and the answers vector for each question
    df = pd.read_json(file_path).transpose()
    return df


def visual_processing():
    df = load_data()
    empty_reddit_timeStamps = (df.explode(['reddit_timeStamps'])['reddit_timeStamps'] == '').sum()
    empty_quora_timeStamps = (df.explode(['quora_timeStamps'])['quora_timeStamps'] == '').sum()
    locations_and_answers = df[["locations", "quora_predicted_answers"]].to_numpy()

    num_reddit_answers = df.explode(["reddit_predicted_answers"]).shape[0]
    num_quora_answers = df.explode(["quora_predicted_answers"]).shape[0]
    num_correct_reddit_answers = df.explode(["reddit_predicted_answers"])['reddit_predicted_answers'].sum()
    num_correct_quora_answers = df.explode(["quora_predicted_answers"])['quora_predicted_answers'].sum()
    num_incorrect_reddit_answers = num_reddit_answers-num_correct_reddit_answers
    num_incorrect_quora_answers = num_quora_answers-num_correct_quora_answers


    unique_locations = df.explode(['locations'])['locations'].nunique()
    unique_categories = df['category'].nunique()

    answers_statistics = make_subplots(1, 3, specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]])

    answers_statistics.add_trace(go.Pie(labels=['Reddit', 'Quora'], 
                         values=[num_reddit_answers, num_quora_answers],
                         title='Total Questions'),
                         1,1)
    answers_statistics.add_trace(go.Pie(labels=['Reddit', 'Quora'], 
                         values=[num_correct_reddit_answers, num_correct_quora_answers],
                         title='Correct Questions'),
                         1,2)
    answers_statistics.add_trace(go.Pie(labels=['Reddit', 'Quora'], 
    values=[num_incorrect_reddit_answers, num_incorrect_quora_answers],
    title='Incorrect Questions'),
    1,3)
    answers_statistics.update_layout(title_text="Answers distribution between Quora and Reddit")

    loc_answer_correct = defaultdict(int)
    loc_answer_wrong = defaultdict(int)
    unknown = 0

    for i in range(locations_and_answers.shape[0]):       
        for loc, answer in zip(locations_and_answers[i][0], locations_and_answers[i][1]):
            if loc == 'Unknown':
                unknown += 1
                continue
            if loc == "America":
                location = "USA"
            else:
                location = loc
            if answer == 1:
                loc_answer_correct[location] += 1
            elif answer == 0:
                loc_answer_wrong[location] += 1

    st.write("**How does our final data look?**")
    st.write(df.iloc[:df.shape[0],:5])

    st.markdown(f'''
        **Data Statistics**
        - Number of empty timeStamp values: {
           empty_quora_timeStamps + empty_reddit_timeStamps
            }
        - There are {unknown} unknown locations.
        - Identified {unique_locations} unique locations.
        - There are {unique_categories} categories.
    ''')
    # Display the figure
    st.plotly_chart(answers_statistics)

    st.subheader("Category Analysis")
    category_count = dict(df['category'].value_counts())
    st.write("**Unique Categories**")
    st.bar_chart(category_count)

    fig, ax = plt.subplots()
    ax.pie(category_count.values(), labels=category_count.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal') 
    ax.set_facecolor("#ffffff")

    st.pyplot(fig)

    # category_answers = df[["category", "predicted_answers"]].to_numpy()
    # cat_answer_correct = defaultdict(int)
    # cat_answer_wrong = defaultdict(int)

    # for i in range(category_answers.shape[0]):
    #     cat_answer_correct[category_answers[i][0]] = category_answers[i][1].count(1)
    #     cat_answer_wrong[category_answers[i][0]] = category_answers[i][1].count(0)

    # sorted_cat_answer_correct = dict(sorted(cat_answer_correct.items(), key=lambda item: item[1], reverse=True))
    # sorted_cat_answer_wrong = dict(sorted(cat_answer_wrong.items(), key=lambda item: item[1], reverse=True))
    # st.write("**Top Categories with Correct Answers**")
    # st.bar_chart(sorted_cat_answer_correct)

    # st.write("**Top Categories with Wrong Answers**")
    # st.bar_chart(sorted_cat_answer_wrong)
    
    correctness, incorrectness, overall = evaluate_approaches()
    st.subheader("NLP Models Analysis")
    st.write("Accuracy of Models for Correct Answers")
    st.bar_chart(data=correctness)
    st.write("Accuracy of Models for Incorrect Answers")
    st.bar_chart(data=incorrectness)
    st.write("Overall Accuracy of Models")
    st.bar_chart(data=overall)

    st.subheader(f"Top 5 Misbeliefs")
    reddit_correct_answers = df['reddit_predicted_answers'].apply(lambda x: sum(x)).values
    quora_correct_answers = df['quora_predicted_answers'].apply(lambda x: sum(x)).values
    reddit_incorrect_answers = df['reddit_predicted_answers'].apply(lambda x:len(x)-sum(x)).values
    quora_incorrect_answers = df['quora_predicted_answers'].apply(lambda x:len(x)-sum(x)).values
    total_correct = reddit_correct_answers + quora_correct_answers
    total_incorrect = reddit_incorrect_answers + quora_incorrect_answers
    correct_answers = pd.DataFrame(total_correct,
                                   index=df['question'],
                                   columns=["correct answers"]
                                   )
    incorrect_answers = pd.DataFrame(total_incorrect,
                                    index=df['question'],
                                    columns=["incorrect answers"]
                                    )
    bar_data_1 = pd.concat([correct_answers,incorrect_answers],axis=1)
    bar_data_1['total_answers'] = bar_data_1['correct answers'] + bar_data_1['incorrect answers']
    bar_data_1 = bar_data_1.sort_values("total_answers",ascending=False).drop("total_answers",axis=1)
    st.bar_chart(data=bar_data_1.head(),color=['#ffaa00',"#ffaa0088"],height=500,width=1000)

    st.subheader(f"Most incorrect questions")
    most_incorrect_df = pd.DataFrame(data=total_incorrect,columns=["incorrect_count"],index=df['question']).sort_values("incorrect_count",ascending=False)
    st.line_chart(most_incorrect_df)

    st.subheader("Answers based on Category")
    st.write(f"**Incorrect answers: By category**")
    reddit_incorrect_grouped = df.groupby("category")['reddit_predicted_answers'].\
        sum().apply(lambda x: len(x)-sum(x)).to_frame().T
    quora_incorrect_grouped = df.groupby("category")['quora_predicted_answers'].\
        sum().apply(lambda x: len(x)-sum(x)).to_frame().T
    incorrect_categories_selected = st.multiselect(label='Select categories for incorrect answers',options=df['category'].unique(), default=[df["category"].unique()[0], df["category"].unique()[1]])
    if incorrect_categories_selected != []:
        reddit_incorrect_grouped = reddit_incorrect_grouped[incorrect_categories_selected]
        quora_incorrect_grouped = quora_incorrect_grouped[incorrect_categories_selected]
        temp_df = pd.concat([reddit_incorrect_grouped,quora_incorrect_grouped],axis=1)
        temp_df = temp_df.T.rename({"reddit_predicted_answers":"reddit_incorrect","quora_predicted_answers":"quora_incorrect"},axis=1)
        st.bar_chart(temp_df)
    else:
        st.error("select a category")

    st.write(f"**Correct answers: By category**")
    correct_categories_selected = st.multiselect(label='Select categories for correct answers',options=df['category'].unique(), default=[df["category"].unique()[0], df["category"].unique()[1]])
    reddit_correct_grouped = df.groupby("category")['reddit_predicted_answers'].\
        sum().apply(lambda x: sum(x)).to_frame().T
    quora_correct_grouped = df.groupby("category")['quora_predicted_answers'].\
        sum().apply(lambda x: sum(x)).to_frame().T
    if correct_categories_selected != []:
        reddit_correct_grouped = reddit_correct_grouped[incorrect_categories_selected]
        quora_correct_grouped = quora_correct_grouped[incorrect_categories_selected]
        temp_df = pd.concat([reddit_correct_grouped,quora_correct_grouped],axis=1)
        temp_df = temp_df.T.rename({"reddit_predicted_answers":"reddit_correct","quora_predicted_answers":"quora_correct"},axis=1)
        st.bar_chart(temp_df)
    else:
        st.error("select a category")
    
    st.write(f"**Accuracy by Category**")
    total_reddit_grouped = df.groupby("category")['reddit_predicted_answers'].\
    sum().apply(lambda x: len(x)).to_frame()
    total_quora_grouped = df.groupby("category")['quora_predicted_answers'].\
        sum().apply(lambda x: len(x)).to_frame()
    reddit_correct_grouped = df.groupby("category")['reddit_predicted_answers'].\
        sum().apply(lambda x: sum(x)).to_frame()
    quora_correct_grouped = df.groupby("category")['quora_predicted_answers'].\
        sum().apply(lambda x: sum(x)).to_frame()
    reddit_accuracy_grouped = np.where(total_reddit_grouped['reddit_predicted_answers'] != 0,
                            reddit_correct_grouped['reddit_predicted_answers'] / total_reddit_grouped['reddit_predicted_answers'],
                            0)
    quora_accuracy_grouped = np.where(total_quora_grouped['quora_predicted_answers'] != 0,
                            quora_correct_grouped['quora_predicted_answers'] / total_quora_grouped['quora_predicted_answers'],
                            0)
    accuracy_df = pd.DataFrame({"reddit_accuracy":reddit_accuracy_grouped,"quora_accuracy":quora_accuracy_grouped},index=df['category'].unique())
    st.bar_chart(accuracy_df,color=[COLORS[4],COLORS[8]])

    
    # Popular misbeliefs from reddit based on timestamp year by category
    st.subheader("Popular misbelief from **reddit** based on timestamp year by category")
    temp_df = df.explode(['reddit_predicted_answers','reddit_timeStamps'])
    replacement_year = pd.to_datetime("2020-10-27T14:13:12.111Z").year # need a better way to do this 
    temp_df['timeStamps'] = temp_df['reddit_timeStamps'].apply(lambda x: pd.to_datetime(x).year if x != "" and x != "None" else replacement_year)
    # group by timeStamp
    grouped_temp_df = temp_df.groupby(['timeStamps','category'])['reddit_predicted_answers'].apply(lambda x: (x == 0).sum()).to_frame()
    reddit_bar_data = grouped_temp_df.pivot_table(index='timeStamps', columns='category', values='reddit_predicted_answers', fill_value=0)
    # TODO: INCREASE FIGSIZE
    st.bar_chart(reddit_bar_data,color = COLORS[:len(reddit_bar_data.columns)])

    # Popular misbeliefs from quora based on timestamp year by category
    st.subheader("Popular misbelief from **quora** based on timestamp year by category")
    temp_df = df.explode(['quora_predicted_answers','quora_timeStamps'])
    replacement_year = pd.to_datetime("2020-10-27T14:13:12.111Z").year # need a better way to do this 
    temp_df['timeStamps'] = temp_df['quora_timeStamps'].apply(lambda x: pd.to_datetime(x).year if x != "" and x != "None" else replacement_year)
    # group by timeStamp
    grouped_temp_df = temp_df.groupby(['timeStamps','category'])['quora_predicted_answers'].apply(lambda x: (x == 0).sum()).to_frame()
    quora_bar_data = grouped_temp_df.pivot_table(index='timeStamps', columns='category', values='quora_predicted_answers', fill_value=0)
    # TODO: INCREASE FIGSIZE
    st.bar_chart(quora_bar_data,color = COLORS[:len(quora_bar_data.columns)])

    sorted_loc_answer_correct = {k: v for k,v in sorted(loc_answer_correct.items(), key=lambda item: item[1], reverse=True)}
    sorted_loc_answer_wrong = dict(sorted(loc_answer_wrong.items(), key=lambda item: item[1], reverse=True)) 

    st.subheader("Location Analysis")
    data = {"Country": sorted_loc_answer_correct.keys(),
        "Value": sorted_loc_answer_correct.values()}

    df = pd.DataFrame(data)

    st.write("**Top Countries with Correct Answers** - *Map & Bar Plot Visualization*")
    # Create a choropleth map
    fig = px.choropleth(df,
                        locations="Country",
                        locationmode="country names",
                        color="Value",
                        color_continuous_scale="Magma",
                        labels={"Value": "Correct Answers"})

    # Display the map chart using Streamlit
    st.plotly_chart(fig)
    st.bar_chart(sorted_loc_answer_correct)

    st.write("**Top Countries with Wrong Answers**")
    st.bar_chart(sorted_loc_answer_wrong)



        
if __name__ == "__main__":
    visual_processing()
