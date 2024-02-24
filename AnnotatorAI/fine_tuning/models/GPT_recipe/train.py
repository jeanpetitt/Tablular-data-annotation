from utils import loginHub, peft_confifguration, train_HyperParmeters, load_model, load_tokenizer, load_trainer, load_peft_model
import time
import gc
import argparse


def format_instruction(sample):
    prompt = f"""
        ### USe this Agent to make semantic annotation of you data
        # Entity Label
        {sample['label']}

        # Entity Description
        {sample['description']}

        # Entity URI
        { sample['entity']}
      """
    return prompt


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        required=True, help="Name of the base model", default="openai-community/gpt2")
    parser.add_argument("--dataset", type=str,
                        default="yvelos/semantic_annotation", help="dataset used to train model")
    parser.add_argument("--hf_rep", type=str,
                        default="yvelos/Annotator_2_GPT", help="huggingFace repository for the fine tuned model")
    parser.add_argument("--hf_token", type=str, help="huggingFace Token")
    args = parser.parse_args()
    return args


def train(model_id, dataset):
    """ Load LoRa configuration"""
    peft_config = peft_confifguration()

    """ Load Training argumenst"""
    args = train_HyperParmeters(),

    """ Load Model and tokenizer"""
    model = load_model(model_id)
    tokenizer = load_tokenizer(model_id)

    """ Load Trainer"""
    trainer = load_trainer(
        model=model,
        dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        format_instruction=format_instruction,
        args=args

    )

    start_time = time.perf_counter()
    print("Start training: ", start_time)
    trainer.train()
    end_time = f"{(time.perf_counter() - start_time) /60 :.2f}"
    print("Total training time: ", end_time, " min")

    trainer.save_model()

    del model
    del trainer
    print(gc.collect())

    return tokenizer


if __name__ == '__main__':
    args = argsparser()
    """ Login into the hub"""
    loginHub(args.hf_token)

    tokenizer = train(args.model_id, args.dataset)
    """ push model on the hub"""
    model = load_peft_model("./result")
    model.push_to_hub(model_id=args.hf_rep)
    tokenizer.push_to_hub(model_id=args.hf_rep)
