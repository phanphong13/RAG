# #!pip install torch accelerate bitsandbytes datasets transformers peft trl scipy

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, DatasetDict, load_from_disk
from torch import cuda, bfloat16
import transformers

import torch
import torch.nn as nn

# data = load_from_disk("/home/test/RAG/fine_tuning/data/neural")["train"]
# print(data[0]["text"])

model_id = "meta-llama/Llama-2-7b-chat-hf"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=True
)


### Ko cos
model.config.use_cache = False
model.config.pretraining_tp = 1

### ko co

# model.eval()
# print(f"Model loaded on {device}")


# mem = model.get_memory_footprint()
# print("Memory footprint: {} ".format(mem))

# should be (7B) 7,000,000,000*4(Int4) / 8(8 bits per byte) = 3,500,000,000 = 3.5GB
# actual (7B)  3,829,940,224 (not all weights become int 4)
# actual (13B)  7,083,970,560  (not all weights become int 4)


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

# Load the dataset from Hugging Face
from datasets import load_dataset

# dataset = load_dataset("kaist-ai/CoT-Collection", split="train")
# dataset_cot = load_dataset("rajpurkar/squad_v2", split="train")
dataset_cot = load_dataset("neural-bridge/rag-dataset-12000", split="train")

print(f'Number of records: {len(dataset_cot)}')
print(f'Column names are: {dataset_cot.column_names}')


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
    context = item["context"]
    question = item["question"]
    # answers = item["answers"]["text"]
    answers = item["answer"]
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
\nOnly returns the helpful answer below and nothing else.\n
Helpful answer: {answers}/s>"""}

# p = create_prompt(dataset_cot[0])

# print(p)
# print(p["text"])

dataset = dataset_cot.map(get_prompt)

# print(dataset[0]["text"])

# dataset = dataset.map(
#         batched=True,
#         remove_columns=['id', 'title']
#     )

print(dataset[0]["text"])

#max length of the model
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

mx = get_max_length(model)

#tokenize dataset
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < mx)

len(dataset)

seed = 42
set_seed(seed)

dataset = dataset.shuffle(seed=seed)
print(dataset[0]["text"])
# dataset = load_from_disk("/home/test/RAG/fine_tuning/data/neural")["train"]
# print(dataset[0]["text"])


for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
print(modules)

#['v_proj', 'up_proj', 'down_proj', 'k_proj', 'o_proj', 'q_proj', 'gate_proj']

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,  #attention heads
    lora_alpha=64,  #alpha scaling
    target_modules=modules,  #gonna train all
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM", #for Decoder models like GPT Seq2Seq for Encoder-Decoder models like T5
)

##Get the PEFT Model using the downloaded model and the loRA config
model = get_peft_model(model, config)

print(model)

# Print Trainable parameters
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
)

output_dir = "/home/test/RAG/fine_tuning/results/neural"

tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200, #20,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

trainer.train()

trainer.model.save_pretrained(output_dir)
# trainer.tokenizer.save_pretrained(output_dir)