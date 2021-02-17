from os import path
import glob
import pandas as pd
import os


train_path = "../data/DadoEvalTrain/Train/CoarseTask"

if __name__ == "__main__":
    list_letters = []

    for class_id in range(1, 6):
        train_class_path = path.join(train_path, "Class" + str(class_id))
        for filename in glob.glob(path.join(train_class_path, "*.txt")):
            with open(filename, 'r') as f:
                text = f.read()
                text = text.strip()
                text = text.replace('\n', ' ')
                list_letters.append([text, class_id - 1])

    print(len(list_letters))

    df = pd.DataFrame(list_letters, columns=['text', 'class'])

    df.to_csv(path.join("../data", "coarse_train.csv"), index=False)
