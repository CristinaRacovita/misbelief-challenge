import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import time
import numpy as np
import requests
from datasets import load_dataset
import re

BASE_URL = "https://www.reddit.com"
#QUESTION = "What happens to you if you eat watermelon seeds?"
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

def get_reddit_answers(question):
    answers = []
    timeStamps = []
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome()
    url = f'{BASE_URL}/search/?q={question.replace(" ", "+")}'
    reddit_responses_request = requests.get(url)

    if reddit_responses_request.status_code == 200:
        reddit_responses_crawler = BeautifulSoup(reddit_responses_request.text, PARSER)
        question = reddit_responses_crawler.find("a", class_="absolute inset-0")
        question_link = question.attrs["href"]

        post_url = f"{BASE_URL}{question_link}"
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
                timeStamps.append(timeEntries[i]['datetime'])
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
    answers = np.array(answers).reshape(len(answers),1)
    timeStamps = np.array(timeStamps).reshape(len(timeStamps),1)
    answers_df = pd.DataFrame(answers, columns=["answers"])
    answers_df = answers_df['answers'].apply(remove_extra_spaces_and_emojis)
    timestamps_df = pd.DataFrame(timeStamps, columns=["timeStamps"])
    timestamps_df = timestamps_df.apply(pd.to_datetime)
    combined_df = pd.concat([timestamps_df,answers_df], axis=1)
    return combined_df


if  __name__ == "__main__":
    data = pd.read_json(r"./misbelief-challenge/truthful_qa.json") 
    for i in range(5):
        question = data['validation'][i]['question']
        answers_df = get_reddit_answers(question)
        #what to do with all the answers to the questions?
        answers_df.to_csv(f'./misbelief-challenge/answers/question_{i}.csv')

