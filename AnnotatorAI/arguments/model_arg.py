from dataclasses import dataclass
from typing import Optional
from transformers import TrainingArguments as TrainArgs
import argparse


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        required=True, help="Name of the base model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str,
                        required=True, help="destination of the fine tuned model", default="result")
    parser.add_argument("--epochs", type=int,
                        required=True, help="destination of the fine tuned model", default=2)

    parser.add_argument("--dataset", type=str,
                        default="yvelos/semantic_annotation", help="dataset used to train model")
    parser.add_argument("--hf_rep", type=str,
                        default="yvelos/Annotator_2_Mi", help="huggingFace repository for the fine tuned model")
    parser.add_argument("--hf_token", type=str, help="huggingFace Token")
    args = parser.parse_args()
    return args


@dataclass
class TrainingArgument:
    learning_rate: Optional[str] = None
    num_train_epochs: Optional[int] = None
    output_dir: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = 1
    gradient_checkpointing: Optional[bool] = True
    per_device_train_batch_size: Optional[int] = 1
    save_steps: Optional[int] = 0
    logging_steps: Optional[int] = 10
    weight_decay: Optional[float] = 0.05
    evaluation_strategy: Optional[str] = "epoch"
    save_strategy: Optional[str] = "epoch"
    load_best_model_at_end: Optional[bool] = True
    lr_scheduler_type: Optional[str] = "cosine"
    optim: Optional[str] = "paged_adamw_32bit"
    report_to: Optional[str] = "tensorboard"
    seed: Optional[str] = 42
    max_steps: Optional[int] = -1
    logging_dir: Optional[str] = None
    group_by_length: Optional[str] = False
    warmup_ratio: Optional[float] = 0.03
    max_grad_norm: Optional[float] = 0.3

    def load_train_args(self):
        args = TrainArgs(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optim,
            report_to=self.report_to,
            seed=self.seed,
            max_steps=self.max_steps,
            logging_dir=self.logging_dir,
            group_by_length=self.group_by_length,
            max_grad_norm=self.max_grad_norm
        )
        return args
