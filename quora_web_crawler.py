from bs4 import BeautifulSoup
from model.quora_answer import QuoraAnswer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time

BASE_URL = "https://www.quora.com"
# QUESTION = "What happens to you if you eat watermelon seeds?"
# PARSER = "html.parser"
# QUORA_QUESTION = 'What happens to the seeds I swallow while eating watermelon?'
# USER_URL = 'https://www.quora.com/profile/Kait-McNeill'
options = webdriver.ChromeOptions()
options.add_argument("headless")

def scroll_down(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2) 

def get_quora_questions(question):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    url = f'{BASE_URL}/search/?q={question.replace(" ", "+")}'

    driver.get(url)
    html = BeautifulSoup(driver.page_source, "html.parser")
    result = html.find_all("div", {"class": "q-flex qu-flexDirection--row"})
    posts = []
    for item in result:
        if len(posts) > 3:
            break
        posts.append(item.text)
    driver.quit()
    return posts

def get_quora_answers_and_users(question):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    question_url = f'{BASE_URL}/{question.replace(" ","-")}'
    driver.get(question_url)
    for _ in range(2):
        scroll_down(driver)

    html = BeautifulSoup(driver.page_source, "html.parser")
    comments = html.find_all("div", {"class": "q-box spacing_log_answer_content puppeteer_test_answer_content"})
    users = html.find_all("a", {"class":"q-box Link___StyledBox-t2xg9c-0 dFkjrQ puppeteer_test_link qu-color--gray_dark qu-cursor--pointer qu-hover--textDecoration--underline"})
    timestamps = html.find_all("a", {"class":"q-box Link___StyledBox-t2xg9c-0 dFkjrQ answer_timestamp qu-cursor--pointer qu-hover--textDecoration--underline"})
    i = 0
    answers = []
    for comment in comments:
        if len(answers) > 3:
            break
        answers.append(QuoraAnswer(comment.text, users[i]['href'], timestamps[i].text))
        i+=1

    driver.quit()
    return answers

def get_user_location(profile_url):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(profile_url)
    html = BeautifulSoup(driver.page_source, "html.parser")
    locations = html.find_all("div", {"class": "q-text qu-truncateLines--2"})
    locs = []
    for location in locations:
        if 'Lived in' in location.text or 'Lives in' in location.text:
            locs.append(location.text)
            # print(location.text)
    driver.quit()
    return locs

# get_quora_answers_and_users(QUORA_QUESTION)
# get_user_location(USER_URL)
