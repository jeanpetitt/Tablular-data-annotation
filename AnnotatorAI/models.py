from dataclasses import dataclass
from typing import Optional
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer


@dataclass
class BaseLLMTune:
    def loginHub(self, token=None):
        if token is not None:
            login(token=token)
        else:
            load_dotenv()
            login(token=os.environ['TOKEN'])

        return login

    def push_model_to_hub(self, model, rep_id):
        model.push_model_to_hub(rep_id)
        return model

    def get_peft_config(self):
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=['k_proj', 'gate_proj', 'v_proj',
                            'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )
        return peft_config

    def load_model(
        self,
        model_id: Optional[str] = None,
        load_in_4bit: Optional[bool] = False,
        load_in_8bit: Optional[bool] = False,
        bnb_4bit_use_double_quant: Optional[bool] = False,
        bnb_4bit_quant_type: Optional[str] = "nf4",
        bnb_4bits_compute_dtype: Optional[str] = None,
        use_cache: Optional[bool] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        torch_dtype: Optional[str] = None,
        use_peft: Optional[bool] = True,
    ):
        if model_id is None:
            print("You must provide a path of the pretrained model")

        if load_in_4bit == True and load_in_8bit == True:
            print("you can use load the model in 4bit and 8bits in same time")

        if bnb_4bit_quant_type or bnb_4bit_use_double_quant or torch_dtype or bnb_4bits_compute_dtype is not None:
            device_map = {"": 0}

        if device_map == "auto":
            device_map = "cuda"

        if bnb_4bit_use_double_quant is not None and load_in_4bit is None or False:
            print("you must set load_in_4bit at True.")

        if bnb_4bits_compute_dtype is not None:
            bnb_4bits_compute_dtype = getattr(torch, bnb_4bits_compute_dtype)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            torch_dtype=torch_dtype,
            use_cache=use_cache,
            low_cpu_mem_usage=low_cpu_mem_usage,
            return_dict=return_dict
        )

        if use_peft == True:
            # get peft model
            peft_config = self.get_peft_config()
            model = get_peft_model(model, peft_config)
            trainable_params = model.print_trainable_parameters()
            print("Trainable Parameters: ", trainable_params)

            model = prepare_model_for_kbit_training(model)
        else:
            model = prepare_model_for_kbit_training(model)
        return model

    def load_tokenizer(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def compute_metrics(self):
        pass

    def format_instruction(self, sample):
        prompt = f"""
        <s>
            ### USe this Agent to make semantic annotation of you data
            <INST>
            # Entity Label
            {sample['label']}

            # Entity Description
            {sample['description'] if 'description' in sample else ''}
            </INST>
        </s>
            # Entity URI
            { sample['entity']}
        """
        return prompt

    def trainer(self, formatting_func, model, tokenizer, train_dataset, train_args, eval_dataset=None,):
        Trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            max_seq_length=1024,
            tokenizer=tokenizer,
            packing=True,
            args=train_args
        )
        return Trainer


@dataclass
class MistraLTune(BaseLLMTune):
    def __init__(self):
        return super().__init__()


@dataclass
class Llama2Tune(BaseLLMTune):
    def __init__(self):
        return super().__init__()

    def load_tokenizer(model_id):
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


@dataclass
class GPT2Tune:
    def __init__(self):
        return super().__init__()


@dataclass
class T5Tune:
    def __init__(self):
        return super().__init__()
