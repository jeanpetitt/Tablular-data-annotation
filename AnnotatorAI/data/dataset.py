import json
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
import csv


def transform_table_to_cell_vector(table_path):
    list_table = os.listdir(table_path)

    output = []
    for file in list_table:
        if file.endswith(".csv"):
            filename = file.split(".")[0]
            table = pd.read_csv(f"{table_path}/{file}", header=None)
            for cols in table.columns:
                for i, line in table.iterrows():
                    if isinstance(line[cols], str):
                        output.append(line[cols])
                    elif type(line[cols]) == type(np.nan):
                        output.append("")
                        print("np")
    df = pd.DataFrame(output, columns=["label_entity"])

    # df.to_csv(f"{table_path}/vector_table.csv")
    return df


class CustomDataset:
    def __init__(self):
        self.dataset = []

    """ 
        dataset load from huggingFace
    """

    def load_dataset_hub(self, path="yvelos/semantic_annotation", split="train"):
        self.dataset = load_dataset(path, split=split)
        return self.dataset

    """ 
        Dataset Load Locally
    """

    def load_dataset_locally(self, table_path, table_annotation):
        table2vec = transform_table_to_cell_vector(table_path=table_path)
        annotation_matrix = pd.read_csv(table_annotation)

        vec_matrix = pd.concat([table2vec, annotation_matrix], axis=1)

        for data in vec_matrix.values:
            self.dataset.append(
                {
                    "label": data[0],
                    "entity": data[4]
                }
            )
        return self.dataset

    def load_csv_dataset(self, path_csv):
        file = pd.read_csv(path_csv)

        for data in file.values:
            self.dataset.append(
                {
                    "label": data[0],
                    "description": data[1],
                    "entity": data[2]
                }
            )
        return self.dataset

    def push_data_to_hub(self, path_file):
        load_dotenv()
        if path_file.endswith("csv"):
            dataset = load_dataset('csv', data_files=f"{path_file}")
            train_size = 0.8
            train_data = dataset['train'].train_test_split(
                train_size=train_size)['train']
            test_data = dataset['train'].train_test_split(
                train_size=train_size)['test']
            print(test_data[0])
            with open("AnnotatorAI/data/test_semtab.jsonl", 'w') as f:

                for data in test_data:
                    json.dump(data, f)
            with open("AnnotatorAI/data/train_semtab.jsonl", 'w') as f:

                for data in train_data:
                    json.dump(data, f)

            print(dataset)
        elif path_file.endswith("json"):
            pass
        else:
            raise ValueError(
                "this format is not valid, try to use csv, json, jsonl and parquet files")


# data = CustomDataset()
# datasets = data.load_dataset_hub(
#     "yvelos/semtab_2023_ground_thruth", split=['test'])

# print(datasets)

# datas = data.push_data_to_hub(
#     path_file="cea_dataset.csv")

# print(datas)
