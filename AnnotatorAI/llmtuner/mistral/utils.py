from dataclasses import dataclass
from typing import Optional
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer


@dataclass
class MistraLTune:
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
            r=64,  # 8
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            # target_modules=['k_proj', 'gate_proj', 'v_proj',
            #                 'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )
        return peft_config

    def bnbConfig(self):
        bnb_4bits_compute_dtype = "float16"
        bnb_4bits_compute_dtype = getattr(torch, bnb_4bits_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_4bits_compute_dtype

        )
        return bnb_config

    def load_model(
        self,
        model_id: Optional[str] = None,
        device_map: Optional[str] = None,
        use_peft: Optional[bool] = True,
    ):
        if model_id is None:
            raise ValueError(
                "you should pass a model path to continue the process")

        if device_map == "auto":
            device_map = "cuda"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=self.bnbConfig,
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True,
            return_dict=True,
        )

        if use_peft == True:
            peft_config = self.get_peft_config()
            model = get_peft_model(model, peft_config)
            # display the trainable parameters
            model.print_trainable_parameters()

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
        prompt = f"""<s>### Instruction:\n Use this agent to make semantic annotation of your 
        datas \n###Human:\n Generate wikidata URI of {sample['label']} \n ###Assistant: \n the wikidata uri of 
        {sample['label']} is { sample['entity']}</s>
        """
        return prompt

    def trainer(self, formatting_func, model, tokenizer, train_dataset, train_args, eval_dataset=None):
        Trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            max_seq_length=2048,
            tokenizer=tokenizer,
            packing=True,
            args=train_args,
        )
        return Trainer


# model = BaseLLMTune()
# model.loginHub()
