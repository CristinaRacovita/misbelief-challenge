import streamlit as st
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px

st.write(
    """ 
    # Misbeliefs Challenge
         Meaningful description about DS
"""
)

@st.cache_data
def load_data(no_of_files):
    file_paths = [r"../answers/answers_0-20.json",
                 r"../answers/answers_20-40.json"]
    # get timeStamp data and the answers vector for each question
    df = pd.concat([pd.read_json(f) for f in file_paths[:no_of_files]],axis=1).T
    return df


def visualize_popular_misbelief_based_on_location():
    no_of_files = 1
    df = load_data(no_of_files)

    st.write("Quora Questions")
    st.write(df.head())

    category_count = dict(df['category'].value_counts())
    st.write("Unique Categories")
    st.bar_chart(category_count)

    # print(category_count.values())
    fig, ax = plt.subplots()
    ax.pie(category_count.values(), labels=category_count.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal') 
    ax.set_facecolor("#ffffff")

    st.pyplot(fig)

    locations_and_answers = df[["locations", "predicted_answers"]].to_numpy()

    loc_answer_correct = defaultdict(int)
    loc_answer_wrong = defaultdict(int)
    unknown = 0

    for i in range(locations_and_answers.shape[0]):       
        for loc, answer in zip(locations_and_answers[i][0], locations_and_answers[i][1]):
            if loc == 'Unknown':
                unknown += 1
                continue
            if answer == 1:
                loc_answer_correct[loc] += 1
            elif answer == 0:
                loc_answer_wrong[loc] += 1

    st.write(f"There are {unknown} unknown locations.")

    sorted_loc_answer_correct = dict(sorted(loc_answer_correct.items(), key=lambda item: item[1], reverse=True))
    sorted_loc_answer_wrong = dict(sorted(loc_answer_wrong.items(), key=lambda item: item[1], reverse=True)) 

    st.write("Top Countries with Correct Answers")
    st.bar_chart(sorted_loc_answer_correct)

    st.write("Top Countries with Wrong Answers")
    st.bar_chart(sorted_loc_answer_wrong)

    category_answers = df[["category", "predicted_answers"]].to_numpy()
    cat_answer_correct = defaultdict(int)
    cat_answer_wrong = defaultdict(int)

    for i in range(category_answers.shape[0]):
        cat_answer_correct[category_answers[i][0]] = category_answers[i][1].count(1)
        cat_answer_wrong[category_answers[i][0]] = category_answers[i][1].count(0)

    sorted_cat_answer_correct = dict(sorted(cat_answer_correct.items(), key=lambda item: item[1], reverse=True))
    sorted_cat_answer_wrong = dict(sorted(cat_answer_wrong.items(), key=lambda item: item[1], reverse=True))
    st.write("Top Categories with Correct Answers")
    st.bar_chart(sorted_cat_answer_correct)

    st.write("Top Categories with Wrong Answers")
    st.bar_chart(sorted_cat_answer_wrong)

    data = {"Country": sorted_loc_answer_correct.keys(),
        "Value": sorted_loc_answer_correct.values()}

    df = pd.DataFrame(data)

    # Create a choropleth map
    fig = px.choropleth(df,
                        locations="Country",
                        locationmode="country names",
                        color="Value",
                        color_continuous_scale="Magma",
                        title="Top Countries with Correct Answers",
                        labels={"Value": "Correct Answers"})

    # Display the map chart using Streamlit
    st.plotly_chart(fig)




if __name__ == "__main__":
    visualize_popular_misbelief_based_on_location()