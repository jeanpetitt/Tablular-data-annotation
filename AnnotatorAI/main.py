import torch
from models import MistraLTune, GPT2Tune, Llama2Tune, T5Tune
import time
import gc
from data.dataset import CustomDataset
from arguments.model_arg import argsparser, TrainingArgument

mistral = MistraLTune()
gpt2 = GPT2Tune()
llama2 = Llama2Tune()
t5 = T5Tune()
data = CustomDataset()


datasets = data.load_dataset_hub("yvelos/semtab_2023_ground_thruth")


def augmented_data(datas):
    new_dataset = []
    for i in range(3):
        for data in datas:
            new_dataset.append(data)
    return new_dataset


dataset = augmented_data(datasets)


def main():
    sys_args = argsparser()
    training_args = TrainingArgument(
        learning_rate=sys_args.lr,
        output_dir=sys_args.output_dir,
        num_train_epochs=sys_args.epochs,
        per_device_train_batch_size=sys_args.per_device_train_batch_size,
        max_steps=sys_args.max_steps,
        optim=sys_args.optim
    )
    training_args = training_args.load_train_args()

    if sys_args.model_type == 'gpt2'.upper():
        gpt2.loginHub(token=sys_args.hf_token)
        model = gpt2.load_model(
            model_id=sys_args.model_id,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bits_compute_dtype='float16',
            bnb_4bit_quant_type="nf4",
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True,
            return_dict=True,
            use_peft=True
        )
        tokenizer = gpt2.load_tokenizer(sys_args.model_id)
        formatting_func = gpt2.format_instruction
        trainer = gpt2.trainer(
            model=model,
            train_dataset=dataset,
            # eval_dataset=datasets['test'],
            train_args=training_args,
            formatting_func=formatting_func,
            tokenizer=tokenizer
        )
    else:
        mistral.loginHub(token=sys_args.hf_token)
        model = mistral.load_model(
            model_id=sys_args.model_id,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bits_compute_dtype='float16',
            bnb_4bit_quant_type="nf4",
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True,
            return_dict=True,
            use_peft=True
        )
        tokenizer = mistral.load_tokenizer(model_id=sys_args.model_id)
        formatting_func = mistral.format_instruction
        trainer = mistral.trainer(
            model=model,
            train_dataset=dataset,
            # eval_dataset=datasets['test'],
            train_args=training_args,
            formatting_func=formatting_func,
            tokenizer=tokenizer
        )
    start_time = time.perf_counter()

    print("Start training: ", start_time)
    trainer.train()
    end_time = f"{(time.perf_counter() - start_time) /60 :.2f}"
    print("Total training time: ", end_time, " min")

    del model
    del trainer
    gc.collect()


if __name__ == '__main__':
    main()
