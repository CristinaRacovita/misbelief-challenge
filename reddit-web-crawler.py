import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://www.reddit.com"
QUESTION = "What happens to you if you eat watermelon seeds?"
PARSER = "html.parser"

def get_reddit_answers(question):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    url = f'{BASE_URL}/search/?q={question.replace(" ", "+")}'
    reddit_responses_request = requests.get(url)

    if reddit_responses_request.status_code == 200:
        reddit_responses_crawler = BeautifulSoup(reddit_responses_request.text, PARSER)
        question = reddit_responses_crawler.find("a", class_="absolute inset-0")
        question_link = question.attrs["href"]

        post_url = f"{BASE_URL}{question_link}"
        driver.get(post_url)
        html = BeautifulSoup(driver.page_source, "html.parser")
        result = html.find_all("div", {"id": "-post-rtjson-content"})
        timestamps = html.find_all("time")
        i = 0
        for item in result:
            print(item.text)
            print(timestamps[i].text)
            print('--'*10)
            i+=1
        driver.quit()


get_reddit_answers(QUESTION)
