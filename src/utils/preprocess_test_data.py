from os import path
import glob
import pandas as pd
import os


gold_test_file = "../data/Gold-test/GoldStandard_same-genre.tsv"
test_path = "../data/Test_Release/same-genre"

if __name__ == "__main__":

    gold_file = pd.read_csv(gold_test_file, sep='\t', names=["file", "year", "class", "period"])
    gold_file = gold_file.replace({'class': {'Class1': 0, 'Class2': 1, 'Class3': 2, 'Class4': 3, 'Class5': 4}})
    print(len(gold_file))

    list_letters = []
    counter = 0
    for filepath in glob.glob(path.join(test_path, "*.txt")):
        with open(filepath, 'r') as f:
            filename = path.split(filepath)[1]
            if gold_file["file"].str.contains(filename).any():
                row = gold_file.loc[gold_file["file"] == filename]
                text = f.read()
                text = text.strip()
                text = text.replace('\n', ' ')
                list_letters.append([text, row.iloc[0]["class"]])
    print(len(list_letters))

    df = pd.DataFrame(list_letters, columns=['text', 'class'])

    df.to_csv(path.join("../data", "coarse_test.csv"), index=False)
