import torch
from models import BaseLLMTune
import time
import gc
from data.dataset import CustomDataset
from arguments.model_arg import argsparser, TrainingArgument
mistral = BaseLLMTune()

data = CustomDataset()

datas = data.load_csv_dataset(
    "Tablular-data-annotation/AnnotatorAI/fine_tuning/models/Mistral_recipe/data/semtab_dataset.csv")

train_dataset_1 = datas[:1500]
train_dataset_2 = datas[1501:2000]
train_dataset_3 = datas[3001:5000]
eval_dataset_1 = datas[2001:3000]


def main():
    sys_args = argsparser()
    mistral.loginHub(token=sys_args.hf_token)
    training_args = TrainingArgument(
        learning_rate=2e-5,
        output_dir=sys_args.output_dir,
        num_train_epochs=sys_args.epochs
    )
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
        train_dataset=train_dataset_1,
        eval_dataset=eval_dataset_1,
        train_args=training_args,
        formatting_func=formatting_func,
        tokenizer=tokenizer
    )
    start_time = time.perf_counter()

    print("Start training: ", start_time)
    trainer.train()
    end_time = f"{(time.perf_counter() - start_time) /60 :.2f}"
    print("Total training time: ", end_time, " min")


if __name__ == '__main__':
    main()
