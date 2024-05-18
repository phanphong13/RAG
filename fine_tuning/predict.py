import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from trl import SFTTrainer
from datasets import load_dataset, DatasetDict, load_from_disk
from torch import cuda, bfloat16
import transformers

import torch
import torch.nn as nn

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/home/test/RAG/fine_tuning/results/neural"

# model = AutoModelForCausalLM.from_pretrained(model_dir)

# # Load pre-trained tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = PeftConfig.from_pretrained(model_dir)

print(config)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# model = PeftModel.from_pretrained(model, model_dir)
model = get_peft_model(model, config)



mem = model.get_memory_footprint()
print("Memory footprint: {} ".format(mem))


# tst = """<s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>

# Use the following pieces of information to answer the users question.
# If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

# Context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
# Question: When did Beyonce start becoming popular? [/INST] Only returns the helpful answer below and nothing else.
# Helpful answer: /s>"""


system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def create_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)

def get_prompt(item, all_answers=False):
    context = item["context_origin"]
    question = item["question"]
    # answers = item["answers"]["text"]
    # answers = item["answer"]
     # Check if the question is answerable
    # if item['answers']['text']:
    #     answers = f"{item['answers']['text'][0]}"  # Taking the first answer
    # else:
    #     answers = "No answer.\n\n"

    return {
        "text": create_prompt(
            f"""\
Use the following pieces of information to answer the users question.
If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

Context: {context}
Question: {question}""",
            [],
            system_prompt,
        )
        + f""" \
Only returns the helpful answer below and nothing else.
Helpful answer: /s>"""}


input_file = "/home/test/RAG/vectorstores/neural_brige/neural.json" 
output_file = "/home/test/RAG/vectorstores/neural_brige/fine_tuning.json" 
import json

with open(input_file, "r", encoding="utf-8") as json_file:
    dataD = json.load(json_file)
# p = get_prompt(dataD[0])

# print(p)
# print(type(p["text"]))

# print(model)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
# result = pipe(p["text"])
# print(type(result[0]['generated_text']))
for item in dataD:
    question = item['question']
    p = get_prompt(item)

    print(question)
    result = pipe(p["text"])
    split_strings = result[0]['generated_text'].split("\nHelpful answer:")

    # Lấy phần tử cuối cùng
    prediction = split_strings[-1]
    print(question)
    # item["ground_truth"] = item["ground_truth"]
    item["answer"] = prediction

with open(output_file, 'w') as output:
    json.dump(dataD, output, indent=4)