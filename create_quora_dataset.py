import json
import pandas as pd
from quora_web_crawler import (
    get_quora_questions,
    get_quora_answers_and_users,
    get_user_location,
)

from processing.category_processing import filter_categories


def create_data_collection(max_number):
    data = pd.read_json(r"./truthful_qa.json")
    data_store = {}
    batch_size = 20
    if len(data["validation"]) - 1 < max_number:
        max_number = len(data["validation"]) - 1
    for start in range(800, max_number, batch_size):
        print(f"Current batch number: {start} - {start+batch_size}")
        maxim = start + batch_size
        if maxim > max_number:
            maxim = max_number

        for i in range(
            start, maxim
        ):  # store answers batch wise into json files
            if data["validation"][i]["category"] not in ["Misconceptions", "Conspiracies", "Stereotypes", "Confusion: People", "Superstitions", "Misquotations", "Confusion: Places", "Misinformation", "Mandela Effect", "Misconceptions: Topical"]:
                continue
            question = data["validation"][i]["question"]
            quora_posts = get_quora_questions(question)
            answers = []
            timestamps = []
            locations = []

            for post in quora_posts:
                quora_answer = get_quora_answers_and_users(post)
                for qans in quora_answer:
                    location = get_user_location(qans.user_link)
                    if location == []:
                        locations.append('Unknown')
                    else:
                        locations.extend(location)
                    answers.append(qans.answer)
                    timestamps.append(qans.timestamp)
            if not answers:
                i-=1
                continue
            data["validation"][i]["answers"] = answers
            data["validation"][i]["timeStamps"] = timestamps
            data["validation"][i]["locations"] = locations

            data_store[f"question_{i}"] = data["validation"][i]

        with open(f"./answers_{start}-{start+batch_size}.json", "w") as f:
            if data_store:
                json.dump(data_store, f, indent=4)
                data_store = {}


create_data_collection(1000)
