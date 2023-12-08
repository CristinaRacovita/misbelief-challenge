from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.quora.com"
QUESTION = "What happens to you if you eat watermelon seeds?"
PARSER = "html.parser"
QUORA_QUESTION = 'What happens to the seeds I swallow while eating watermelon?'

def get_quora_questions(question):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    url = f'{BASE_URL}/search/?q={question.replace(" ", "+")}'

    driver.get(url)
    html = BeautifulSoup(driver.page_source, "html.parser")
    result = html.find_all("div", {"class": "q-flex qu-flexDirection--row"})
    for item in result:
        print(item.text)

def get_quora_answers(question):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    question_url = f'{BASE_URL}/{question.replace(" ","-")}'
    driver.get(question_url)
    html2 = BeautifulSoup(driver.page_source, "html.parser")
    comments = html2.find_all("div", {"class": "q-box spacing_log_answer_content puppeteer_test_answer_content"})
    for comment in comments:
        print(comment.text)
        print('--'*10)

get_quora_answers(QUORA_QUESTION)
