import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


#### Start llama

# bnb_config = BitsAndBytesConfig(load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=False)

# model_id = "lmsys/vicuna-7b-v1.5"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# pipe = pipeline("text-generation",
#                 model=model,
#                 tokenizer = tokenizer,
#                 max_new_tokens = 512,
#                 temperature = 0.1)

# llm = HuggingFacePipeline(pipeline=pipe)

# print(llm.invoke('What is the capital of England?'))

#### End llama2


#### Openai

# from langchain_openai import ChatOpenAI
# import os
# os.environ["OPENAI_API_KEY"] = ""
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#### End openai


#### mistral
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    max_new_tokens = 512,
    temperature = 0.1,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=pipe)

### End mistral

#### T5

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",
#                                               load_in_8bit=True,
#                                               device_map='auto',
#                                             #   torch_dtype=torch.float16,
#                                             #   low_cpu_mem_usage=True,
#                                               )

# pipe = pipeline(
#     "text2text-generation",
#     model=model, 
#     tokenizer=tokenizer, 
#     max_length=512,
#     temperature=0,
#     top_p=0.95,
#     repetition_penalty=1.15
# )

# llm = HuggingFacePipeline(pipeline=pipe)     


### End T5


### wizad

# tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

# model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
#                                               load_in_8bit=True,
#                                               device_map='auto',
#                                               torch_dtype=torch.float16,
#                                               low_cpu_mem_usage=True
#                                               )

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=1024,
#     temperature=0,
#     top_p=0.95,
#     repetition_penalty=1.15
# )

# llm = HuggingFacePipeline(pipeline=pipe)     


### end wizad

print(llm.invoke('What is the capital of England?'))

zero_shot_prompt='''Use the following pieces of information to answer the users question. 
If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

Context:{context}
Question:{question}

Only returns the helpful answer below and nothing else.
Helpful answer
'''

z_prompt = PromptTemplate(template = zero_shot_prompt, input_variables=["context", "question"])

# DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige"
# input_file = DB_FAISS_PATH + "/neural.json" 

DB_FAISS_PATH = "/home/test/RAG/vectorstores/squad"
input_file = DB_FAISS_PATH + "/squad.json" 


# modelPath = "sentence-transformers/all-MiniLM-l6-v2"
modelPath = "intfloat/e5-base-v2"


device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {'device':device}

encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

db = FAISS.load_local(DB_FAISS_PATH,embeddings, allow_dangerous_deserialization = True)

prompt = z_prompt
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=False,
                                 chain_type_kwargs={'prompt':prompt  })

import json
import os

output_file = DB_FAISS_PATH + "/llm/mistral.json"
print(input_file)
with open(input_file, "r", encoding="utf-8") as json_file:
    dataD = json.load(json_file)


for item in dataD:
    question = item['question']
    print(question)
    pre = qa.invoke({"query": question})
    split_strings = pre['result'].split("\nHelpful answer\n")

    # Lấy phần tử cuối cùng
    prediction = split_strings[-1]
    print(question)
    contexts =  [doc.page_content for doc in retriever.get_relevant_documents(question)]
    item["ground_truth"] = item["ground_truth"]
    item["context_retriever"] = contexts
    item["answer"] = prediction

with open(output_file, 'w') as output:
    json.dump(dataD, output, indent=4)