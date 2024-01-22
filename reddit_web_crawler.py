import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import time
import numpy as np
import requests
from datasets import load_dataset
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


BASE_URL = "https://www.reddit.com"
# QUESTION = "What happens to you if you eat watermelon seeds?"
PARSER = "html.parser"

def get_hugging_data():
    return load_dataset("truthful_qa","generation")

def remove_extra_spaces_and_emojis(text:str):
    text = text.strip()
    # Remove all whitespaces except single spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove all emojis
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(lemmatizer:WordNetLemmatizer,sentence):
    sentence = [word for word in remove_extra_spaces_and_emojis(sentence).split(" ") if word not in stopwords.words('english')]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in sentence]
    return ' '.join(lemmatized_words).lower()

def most_similar_posts(question,post_titles,question_links,similarity_threshold):
    vectorizer = TfidfVectorizer()
    lemmatizer = WordNetLemmatizer()
    processed_titles = [preprocess_text(lemmatizer,x) for x in post_titles]
    tfidf_matrix = vectorizer.fit_transform([preprocess_text(lemmatizer,question)])
    # Calculate cosine similarity between the question and each post title
    similarities = cosine_similarity(tfidf_matrix,vectorizer.transform(processed_titles))
    most_similar_indices = similarities.argsort()[0][::-1]
    # Get the most similar question_links corresponding to the most similar post titles
    most_similar_question_links = [question_links[i] for i in most_similar_indices if 
                                   similarities[0, i] >= similarity_threshold]
    return most_similar_question_links


def get_reddit_answers(question,top):
    answers = []
    timeStamps = []
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome()
    url = f'{BASE_URL}/search/?q={question.replace(" ", "+")}'
    reddit_responses_request = requests.get(url)

    if reddit_responses_request.status_code == 200:
        reddit_responses_crawler = BeautifulSoup(reddit_responses_request.text, PARSER)
        question_elements = reddit_responses_crawler.findAll("a", class_="absolute inset-0")
        post_elements = reddit_responses_crawler.findAll("span", {'data-testid': 'post-title-text'})
        post_titles = [p.text for p in post_elements]
        question_links = [q.attrs["href"] for q in question_elements]
        # similarity check between posts and question
        similarity_threshold = 0.75
        if len(post_titles) == 0: return [],[]
        most_relevant_links =  most_similar_posts(question,post_titles,question_links,similarity_threshold)

        for link in most_relevant_links:
            # navigate to the answer page
            post_url = f"{BASE_URL}{link}"
            driver.get(post_url)

            scrolling = True
            while scrolling:
                html = BeautifulSoup(driver.page_source, "html.parser")
                result = html.find_all("div", {"id": "-post-rtjson-content"})
                timeEntries = html.find_all("time")
                previousScrollHeight = driver.execute_script("return document.body.scrollHeight")
                i = 0
                for item in result:
                    answers.append(item.text)
                    try:
                        timeStamps.append(timeEntries[i]['datetime'])
                    except:
                        timeStamps.append("")
                    i += 1

                # Execute JavaScript to scroll to the end of the page
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scrollTo(width, height)

                time.sleep(2)

                currentScrollHeight = driver.execute_script("return document.body.scrollHeight")
                if currentScrollHeight == previousScrollHeight:
                    scrolling = False
    driver.quit()
    # remove duplicate answers 
    top_answers = []
    top_timeStamps = []
    for i in range(len(answers)):
        if answers[i] not in top_answers:
            top_answers.append(answers[i])
            top_timeStamps.append(timeStamps[i])
    return top_answers[:top],top_timeStamps[:top]

def data_collection():
    relevant_categories = ["Misconceptions",
                           "Conspiracies",
                           "Stereotypes",
                           "Confusion: People",
                           "Superstitions",
                           "Misquotations",
                           "Mandela Effect",
                           "Misinformation",
                           "Confusion: Places",
                           "Misconceptions: Topical"]
    data = pd.read_json(r"./misbelief-challenge/truthful_qa.json")
    lemmatizer = WordNetLemmatizer()
    data_store = {}
    total_count = len(data)
    batch_size = 20
    for start in range(800,820,batch_size):
        print(f"Current batch number: {start} - {start+batch_size}")
        for i in range(start,start+batch_size): # store answers batch wise into json files
            category_type = data['validation'][i]['category']
            if category_type in relevant_categories: # filter out irrelevant categories
                question = data['validation'][i]['question']

                answers,timeStamps = get_reddit_answers(question,top=30) 
                # preprocess text before storing
                answers = [preprocess_text(lemmatizer,sentence) for sentence in answers]
                data['validation'][i]['answers'] = answers
                data['validation'][i]['timeStamps'] = timeStamps
                #add data['validation'][i] to data_store with question_i as the index 
                data_store[f'question_{i}'] = data['validation'][i]
            
        with open(f"./misbelief-challenge/answers/answers_{start}-{start+batch_size}.json", "w") as f:
            json.dump(data_store, f, indent=4)



if  __name__ == "__main__":
    data_collection()
   