import gc
import os
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datetime import datetime
import random
import json

# Config
input_data_dir1 = "./data/train.jsonl" # unzip the `REC_data.zip` citations + explanation data and put the file here
input_data_dir2 = "Skywork/Skywork-Reward-Preference-80K-v0.1"
input_data_dir3 = "nvidia/HelpSteer2"
input_data_dir4 = "NCSOFT/offsetbias"
input_data_dir5 = "Vezora/Code-Preference-Pairs"

base_model_name = "mistralai/Mistral-Nemo-Instruct-2407"
output_dir = "./runs_1e-4-256-iter2-paper"
padding_side_req = "right" #by default mistral uses left, but trl has overflow issues with left padding.
max_seq_length = 6144
peft_config = LoraConfig(
    r=256,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

BASE_PROMPT = '''<s>[INST] You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.

Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.
Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are *equally likely* to be the better.

# Instruction:
{instruction}
# Output (a):
{output_1}
# Output (b):
{output_2}
# Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)": [/INST] {response}</s>'''
BASE_PROMPT_REC = '''<s>[INST] {prompt} [/INST] {output}</s>'''


response_template = '[/INST]'
instruction_template = '<s>[INST]'


def apply_formatting_skywork(example, tokenizer):
    prompts = [
        BASE_PROMPT.format(instruction=example["chosen"][0]['content'], output_1=example["chosen"][-1]['content'], output_2=example["rejected"][-1]['content'], response="Output (a)"),
        BASE_PROMPT.format(instruction=example["chosen"][0]['content'], output_1=example["rejected"][-1]['content'], output_2=example["chosen"][-1]['content'], response="Output (b)")
    ]
    # Randomly select one of the prompts
    selected_prompt = random.choice(prompts)
    return {"text": selected_prompt}

def apply_formatting_REC(example, tokenizer):
    prompt = example["prompt"]
    output = example["completion"]

    formatted_prompt = BASE_PROMPT_REC.format(
        prompt=prompt,
        output=output
    )
    return {"text": formatted_prompt}

def apply_formatting_helpsteer(example1, example2, tokenizer):
    instruction = example1["prompt"]
    response1 = example1["response"]
    response2 = example2["response"]

    response1_score = 0.65 * example1["helpfulness"] + 0.8 * example1["correctness"] + 0.45 * example1["coherence"]
    response2_score = 0.65 * example2["helpfulness"] + 0.8 * example2["correctness"] + 0.45 * example2["coherence"]
    if response1_score > response2_score:
        response = "Output (a)"
    else:
        response = "Output (b)"
    final_prompt = BASE_PROMPT.format(instruction=instruction, output_1=response1, output_2=response2, response=response)

    return {"text": final_prompt}

def apply_formatting_offsetbias(example, tokenizer):
    instruction = example["instruction"]
    output1 = example["output_1"]
    output2 = example["output_2"]

    if example["label"] == 1:
        response = "Output (a)"
    else:
        response = "Output (b)"
    final_prompt = BASE_PROMPT.format(instruction=instruction, output_1=output1, output_2=output2, response=response)

    return {"text": final_prompt}

def apply_formatting_code(example, tokenizer):
    instruction = example["input"]
    output1 = example["accepted"]
    output2 = example["rejected"]
    response1 = "Output (a)"
    response2 = "Output (b)"

    prompts = [
        BASE_PROMPT.format(instruction=instruction, output_1=output1, output_2=output2, response=response1),
        BASE_PROMPT.format(instruction=instruction, output_1=output2, output_2=output1, response=response2)
    ]
    # Randomly select one of the prompts
    selected_prompt = random.choice(prompts)
    return {"text": selected_prompt}

def filter_by_length(example, tokenizer):
    len_text = len(tokenizer(example["text"], padding=False, truncation=False)["input_ids"])
    return True if len_text <= max_seq_length else False

def load_and_format_datasets(input_data_dir1, input_data_dir2, input_data_dir3, input_data_dir4, input_data_dir5, tokenizer):
    # Initialize an empty list to hold the data
    train_ds1 = load_dataset('json', data_files=input_data_dir1, split='train').shuffle(seed=82)

    original_columns=train_ds1.column_names
    #format dataset
    train_dataset1 = train_ds1.map(
        function=lambda example: apply_formatting_REC(example, tokenizer),
        remove_columns=original_columns)
    train_dataset1 = train_dataset1.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds1: {len(train_dataset1)}""")

    train_ds2 = load_dataset(input_data_dir2)["train"]
    original_columns=train_ds2.column_names
    #format dataset
    train_dataset2 = train_ds2.map(
        function=lambda example: apply_formatting_skywork(example, tokenizer),
        remove_columns=original_columns)
    train_dataset2 = train_dataset2.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds2: {len(train_dataset2)}""")

    train_ds3 = load_dataset(input_data_dir3)["train"]
    row_list = []
    #format dataset
    for i in range(0, len(train_ds3), 2):
        row_list.append(apply_formatting_helpsteer(train_ds3[i], train_ds3[i+1], tokenizer))
    train_dataset3 = Dataset.from_list(row_list) 
    train_dataset3 = train_dataset3.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds3: {len(train_dataset3)}""")

    train_ds4 = load_dataset(input_data_dir4)["train"]
    original_columns=train_ds4.column_names
    #format dataset
    train_dataset4 = train_ds4.map(
        function=lambda example: apply_formatting_offsetbias(example, tokenizer),
        remove_columns=original_columns)
    train_dataset4 = train_dataset4.filter(lambda example: filter_by_length(example, tokenizer))
    print(f"""train_ds4: {len(train_dataset4)}""")
    
    train_ds5 = load_dataset(input_data_dir5)["train"]
    original_columns=train_ds5.column_names
    #format dataset
    train_dataset5 = train_ds5.map(
        function=lambda example: apply_formatting_code(example, tokenizer),
        remove_columns=original_columns)
    train_dataset5 = train_dataset5.filter(lambda example: filter_by_length(example, tokenizer))
    train_dataset5 = train_dataset5.select(range(10000))
    print(f"""train_ds5: {len(train_dataset5)}""")
    
    # Combine the datasets and shuffle
    train_dataset = concatenate_datasets([train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5]).shuffle(seed=42)

    return train_dataset

def initialize_tokenizer_and_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        #quantization_config=bnb_config,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16
    )
    base_model.config.use_cache = False #use_cache=False if gradient_checkpointing else True
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side=padding_side_req,
        add_eos_token=False,
        add_bos_token=False,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    return tokenizer, base_model

def main():
    tokenizer, base_model = initialize_tokenizer_and_model()
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer)
    train_dataset = load_and_format_datasets(input_data_dir1, input_data_dir2, input_data_dir3, input_data_dir4, input_data_dir5, tokenizer)
    #print("Example 0:\nPrompt after preprocessing:", train_dataset[0]["text"])
    #print("****"*20)
    #print("Example 1:\nPrompt after preprocessing:", train_dataset[1]["text"])

    training_args = SFTConfig(
        output_dir=output_dir,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4, #2.5e-4,  # about 10x smaller than the Mistral pretraining learning rate
        logging_steps=200,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True,
        save_strategy="steps",
        save_steps=200,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        run_name=f"logs-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    assert tokenizer.padding_side == padding_side_req

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator)

    trainer.train()
    trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")
    # Flush memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    #trainer.save_model(output_dir)

if __name__ == "__main__":
    main()