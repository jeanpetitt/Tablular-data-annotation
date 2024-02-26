from peft import PeftConfig, PeftModel
import torch
import time
from transformers import MistralForCausalLM, AutoTokenizer
import pandas as pd
import csv
import os
import numpy as np


def adapter_config(model_id):
    config_adapt = PeftConfig.from_pretrained(model_id)
    return config_adapt


def load_peft_model(model_peft, model_id, device_map):
    model = MistralForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        model_peft,
        is_trainable=True
    )
    return model


def add_adapter(config, model_name):
    pass


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def compute_max_token(prompt_length, max_new_token, _CONTEXT_LENGTH=2048):
    max_returned_tokens = max_new_token + prompt_length
    assert max_returned_tokens <= _CONTEXT_LENGTH, (
        max_returned_tokens,
        _CONTEXT_LENGTH
    )


def annotatorIA_Mistral(model, tokenizer, prompt, max_new_token=100, seed=None):

    string = f"""
        ### USe this Agent to make semantic annotation of you data
        # Entity Label
        {prompt}
        # Entity URI
      """
    if seed is not None:
        torch.manual_seed(seed)
    encoded = tokenizer(string, return_tensors="pt", truncation=True)
    prompt_length = encoded["input_ids"][0].size(0)
    # print(prompt_length)
    compute_max_token(prompt_length=prompt_length, max_new_token=max_new_token)
    pad_token_id = tokenizer.pad_token_id
    input_ids = encoded['input_ids'].cuda()
    attention_mask = encoded['attention_mask'].cuda()

    start_inf = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=0.5,
            top_k=200,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    enf_inf = time.perf_counter() - start_inf
    output_ids = outputs.sequences
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    print(
        f"Time for inference: {enf_inf:.02f} sec total, {tokens_generated /enf_inf :.02f} tokens/sec"
    )
    print(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    generated_intruction = tokenizer.decode(
        output_ids[0], skip_special_tokens=True)
    lines = generated_intruction.split("\n")
    i = 0
    URI = ''
    for line in lines:
        if line.endswith("Entity URI"):
            URI = lines[i+1].split(" ")[-1]
            print(f"Prompt:\n{prompt}\nGenerated URI: ", URI)
            break
        else:
            URI = ''
            i += 1

    return URI


def annotateCea(path_folder, model, tokenizer, seed):

    files_cea = 'cea.csv'
    dataset = os.listdir(path_folder)
    dataset.sort(reverse=False)
    header_cea = ["tab_id", "col_id", "row_id", "entity"]
    with open(files_cea, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header_cea)
        for filed in dataset:
            if filed.endswith(".csv"):
                print(filed)
                _file = pd.read_csv(f"{path_folder}/{filed}", header=None)
                # get total row and colums of each table file csv
                total_rows = len(_file.axes[0])
                total_cols = len(_file.axes[1])
                list_uri = []
                cells_firs_line = [cell for cell in _file.loc[0]]
                print(cells_firs_line)
                for cols in _file.columns:
                    for i, line in _file.iterrows():
                        if isinstance(line[cols], str):
                            # get annotation of the cell
                            result = annotatorIA_Mistral(
                                model=model, tokenizer=tokenizer, seed=seed, prompt=line[cols])
                            list_uri.append(result)
                        # verify if cell is empty by default in dataframe all empty cell called nan
                        elif type(line[cols]) == type(np.nan):
                            list_uri.append("NIL")
                print(len(list_uri))
                # get name of cleaned file
                filename = filed.split(".")[0]
                print("fichier:", filename, "nbre de ligne: ",
                      total_rows, " nombre de col: ", total_cols)
                filetotalrowcol = total_rows * total_cols
                row = 0
                col = 0
                uri_n = 0
                # creer la structure d'un fichier cea
                while row < filetotalrowcol:
                    # for cell in total_cols:
                    if row < total_rows:
                        if list_uri[uri_n] == "NIL":
                            writer.writerow(
                                [filename, col, row, list_uri[uri_n]])
                            row += 1
                            uri_n += 1
                        else:
                            writer.writerow(
                                [filename, col, row, list_uri[uri_n]])
                            row += 1
                            uri_n += 1
                    else:
                        row = 0
                        filetotalrowcol -= total_rows
                        col += 1
                        # end structure cea.csv
            else:
                print("it is not csv file")

    csv_file.close()

    # read output cea csv file
    print("============cea=============")
    _cea = pd.read_csv(files_cea)
    data_cea = _cea.loc[0:]
    print(data_cea)
