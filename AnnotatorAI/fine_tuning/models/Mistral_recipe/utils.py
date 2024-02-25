
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import torch
import os
from peft import LoraConfig, AutoPeftModelForCausalLM


def loginHub(token=None):
    if token is not None:
        login(token=token)
    else:
        load_dotenv()
        login(token=os.environ['TOKEN'])

    return login


def train_HyperParmeters():
    args = TrainingArguments(
        output_dir="result",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,
        save_strategy="steps",
        learning_rate=2e-05,
        weight_decay=0.001,
        fp16=False,
        # bf16=bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_steps=-1,
        group_by_length=False,
        lr_scheduler_type="cosine",
        # disable_tqdm=disable_tqdm,
        report_to="tensorboard",
        seed=42,
        logging_dir=f"./logs"
    )
    return args


def peft_confifguration():
    """ 
        LoRA Parameter
    """
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return peft_config


def load_model(model_id, device_map="auto"):
    """ 
        BistAndBytes parameter
    """
    use_4bits = True
    bnb_4bits_compute_dtype = "float16"
    bnb_4bits_quan_type = "nf4"  # we can use nf4 of fp4
    # activation nested quantization for 4-bits base model (double quantization)
    use_double_quant_nested = False

    compute_dtype = getattr(torch, bnb_4bits_compute_dtype)
    # BitAndBytesConfg int-4 configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bits,
        bnb_4bit_use_double_quant=use_double_quant_nested,
        bnb_4bit_quant_type=bnb_4bits_quan_type,
        bnb_4bits_compute_dtype=compute_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16
    )
    return model


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.name_or_path
    return tokenizer


def load_trainer(model, dataset, peft_config, tokenizer, format_instruction, args):
    Trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=args

    )
    return Trainer


def load_peft_model(model_id):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        is_trainable=True
    )
    return model
