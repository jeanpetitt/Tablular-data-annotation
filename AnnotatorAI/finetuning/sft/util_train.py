from trl import SFTTrainer
from typing import Optional
from dataclasses import dataclass
from transformers import TrainingArguments
from typing import Union, Optional, Tuple, Callable, Dict
import torch


@dataclass
class Trainer(SFTTrainer):
    def train(self):
        pass


trainer = Trainer()
