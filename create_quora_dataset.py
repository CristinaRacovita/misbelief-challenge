import json
import pandas as pd
from quora_web_crawler import (
    get_quora_questions,
    get_quora_answers_and_users,
    get_user_location,
)


def create_data_collection(max_number):
    data = pd.read_json(r"./truthful_qa.json")
    data_store = {}
    batch_size = 20
    for start in range(0, max_number, batch_size):
        print(f"Current batch number: {start} - {start+batch_size}")
        for i in range(
            start, start + batch_size
        ):  # store answers batch wise into json files
            question = data["validation"][i]["question"]
            quora_posts = get_quora_questions(question)
            answers = []
            timestamps = []
            locations = []

            for post in quora_posts:
                quora_answer = get_quora_answers_and_users(post)
                for qans in quora_answer:
                    location = get_user_location(qans.user_link)
                    locations.append(location)
                    answers.append(qans.answer)
                    timestamps.append(qans.timestamp)

            data["validation"][i]["answers"] = answers
            data["validation"][i]["timeStamps"] = timestamps
            data["validation"][i]["locations"] = locations

            data_store[f"question_{i}"] = data["validation"][i]

        with open(f"./answers_{start}-{start+batch_size}.json", "w") as f:
            json.dump(data_store, f, indent=4)


create_data_collection(2)
