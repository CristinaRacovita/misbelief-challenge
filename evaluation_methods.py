import pandas as pd

def evaluate_approaches():
    app_2 = pd.read_json(r"misbelief-challenge\evaluation\evaluation_approach_2.json")
    app_3 = pd.read_json(r"misbelief-challenge\evaluation\evaluation_approach_3.json")
    app_4 = pd.read_json(r"misbelief-challenge\evaluation\evaluation_approach_4.json")
    app_5 = pd.read_json(r"misbelief-challenge\evaluation\evaluation_approach_5.json")
    df = pd.concat([app_2,app_3,app_4,app_5],axis=1)
    df.columns = ["Word2Vec","INSTRUCTOR","FlairEmbeddings","TF-IDF"]
    # evaluate the mean correctness incorrectness and overall accuracy for each approach
    correctness_accuracy = df.apply(lambda x: x.apply(lambda y: y['correctness_accuracy']).mean(), axis=0)
    incorrectness_accuracy = df.apply(lambda x: x.apply(lambda y: y['incorrectness_accuracy']).mean(), axis=0)
    overall_accuracy = df.apply(lambda x: x.apply(lambda y: y['overall_accuracy']).mean(), axis=0)
    return correctness_accuracy, incorrectness_accuracy, overall_accuracy

if __name__ == "__main__":
    evaluate_approaches()
