from os import path
import glob
import pandas as pd
import os


train_path = "../data/DadoEvalTrain/Train/FineTask"

folders = ["1901-1905", "1906-1910", "1911-1915", "1916-1920", "1921-1925", "1926-1930", "1931-1935", "1936-1940",
           "1941-1945", "1946-1950", "1951-1955"]


if __name__ == "__main__":
    list_letters = []

    for class_id, class_name in enumerate(folders):
        train_class_path = path.join(train_path, class_name)
        for filename in glob.glob(path.join(train_class_path, "*.txt")):
            with open(filename, 'r') as f:
                text = f.read()
                text = text.strip()
                text = text.replace('\n', ' ')
                list_letters.append([text, class_id])

    print(len(list_letters))

    df = pd.DataFrame(list_letters, columns=['text', 'class'])

    df.to_csv(path.join("../data", "fine_train.csv"), index=False)
