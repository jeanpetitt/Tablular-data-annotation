import pandas as pd
import numpy as np


def transform_table_to_cell_vector(table_path):

    file = pd.read_csv(table_path, header=None)
    output = []
    for cols in file.columns:
        for i, line in file.iterrows():
            if isinstance(line[cols], str):
                output.append(line[cols])
            elif type(line[cols]) == type(np.nan):
                output.append("")
                print("np")
    df = pd.DataFrame(output, columns=["question"])

    df.to_csv("8468806_0_4382447409703007384.csv")


transform_table_to_cell_vector("datasets/8468806_0_4382447409703007384.csv")
