from os import path
import glob
import pandas as pd
import os


gold_test_file = "../data/Gold-test/GoldStandard_same-genre.tsv"
test_path = "../data/Test_Release/same-genre"


if __name__ == "__main__":

    gold_file = pd.read_csv(gold_test_file, sep='\t', names=["file", "year", "class", "period"])
    gold_file = gold_file.replace({'period': {"1901-1905": 0, "1906-1910": 1, "1911-1915": 2, "1916-1920": 3,
                                              "1921-1925": 4, "1926-1930": 5, "1931-1935": 6, "1936-1940": 7,
                                              "1941-1945": 8, "1946-1950": 9, "1951-1955": 10}})
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
                list_letters.append([text, row.iloc[0]["period"]])
    print(len(list_letters))

    df = pd.DataFrame(list_letters, columns=['text', 'class'])

    df.to_csv(path.join("../data", "fine_test.csv"), index=False)
