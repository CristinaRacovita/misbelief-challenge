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
import json

BASE_URL = "https://www.reddit.com"
QUESTION = "What happens to you if you eat watermelon seeds?"
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
        question = reddit_responses_crawler.findAll("a", class_="absolute inset-0")
        # taking just the top 3 as usually the links become irrelevant after that
        question_links = [q.attrs["href"] for q in question][:3]
        for link in question_links:
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
                    # print(item.text)
                    # print(timestamps[i].text)
                    # print('--' * 10)
                    i += 1

                # Execute JavaScript to scroll to the end of the page
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scrollTo(width, height)

                time.sleep(2)

                currentScrollHeight = driver.execute_script("return document.body.scrollHeight")
                if currentScrollHeight == previousScrollHeight:
                    scrolling = False
    driver.quit()

    return answers[:top],timeStamps[:top]

def data_collection():
    data = pd.read_json(r"./misbelief-challenge/truthful_qa.json") 
    lemmatizer = WordNetLemmatizer()
    data_store = {}
    total_count = len(data)
    batch_size = 20
    for start in range(260,280,batch_size):
        print(f"Current batch number: {start} - {start+batch_size}")
        for i in range(start,start+batch_size): # store answers batch wise into json files
            question = data['validation'][i]['question']

            answers,timeStamps = get_reddit_answers(question,top=50) 
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
   