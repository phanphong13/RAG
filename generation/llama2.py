import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False)

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config,device_map={"":0})

pipe = pipeline("text-generation",
                model=model,
                tokenizer = tokenizer,
                max_new_tokens = 512,
                temperature = 0.1)

llm = HuggingFacePipeline(pipeline=pipe)

# print(llm.invoke('What is the capital of Vietnam?'))


zero_shot_prompt='''Use the following pieces of information to answer the users question. 

Context:{context}
Question:{question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
'''

few_shot_prompt='''You are an expert information retrieval assistant. Your task is to accurately and concisely extract the answer from the provided context. 
If the answer is not available within the context, respond with "I do not know". 
Use the following examples to understand how to perform your task:

### Example 1
Context: Neil Armstrong became the first man to walk on the Moon as the commander of the American mission Apollo 11 by NASA, which landed on July 20, 1969.
Question: Who was the first man to walk on the Moon?
Only returns the helpful answer below and nothing else.
Helpful answer: 
Neil Armstrong

### Example 2
Context: The Amazon rainforest, representing over half of the planet's remaining rainforests, is the largest and most biodiverse tract of tropical rainforest in the world.
Question: What is the Amazon known for?
Only returns the helpful answer below and nothing else.
Helpful answer: 
It is known for being the largest and most biodiverse tract of tropical rainforest in the world.

### Example 3
Context: Helsinki is the capital city of Finland, located on the shore of the Gulf of Finland.
Question: What is the capital of Sweden?
Only returns the helpful answer below and nothing else.
Helpful answer: I don't know.

### Example 4
Context: The office of the President of the United States was established in 1789. The first president was George Washington.
Question: Who was the first president of the United States?
Only returns the helpful answer below and nothing else.
Helpful answer: George Washington

### Your Task
Context: {context}
Question: {question}
Only returns the helpful answer below and nothing else.
Helpful answer: 
'''



z_prompt = PromptTemplate(template = zero_shot_prompt, input_variables=["context", "question"])
f_prompt = PromptTemplate(template = few_shot_prompt, input_variables=["context", "question"])


from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter


# from datasets import load_dataset

# data_squad_v2 = load_dataset("neural-bridge/rag-dataset-12000", split="test")
# data_squad_v2 = data_squad_v2['context']

# print(data_squad_v2)

# def remove_duplicates(list_str):
#   new_list = set(list_str)
#   return list(new_list)

# data_squad_v2 = remove_duplicates(data_squad_v2)

# data =list(map(lambda string: Document(page_content=string, metadata={}), data_squad_v2))

# print(len(data))

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# # 'data' holds the text you want to split, split the text into documents using the text splitter.
# docs = text_splitter.split_documents(data)

DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/splitter/spacy"

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# modelPath = "intfloat/e5-base-v2"


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

retriever = db.as_retriever(search_kwargs={"k": 3})
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CohereRerank

# compressor = CohereRerank(cohere_api_key='CgTSzXRXcP6GQtUbS74XyMVSBJ0Bbfcemx2Vzue5')
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
prompt = z_prompt

qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=False,
                                 chain_type_kwargs={'prompt':prompt  })

# question = "In what country is Normandy located?"
# # context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
# result = qa.invoke({"query": question})
# split_strings = result['result'].split("\nHelpful answer:")
# print(split_strings[-1])

#     # Lấy phần tử cuối cùng
#     prediction = split_strings[-1]

# a = [doc.page_content for doc in retriever.get_relevant_documents(question)]
# for item in a:
#     print("/n/n")
#     print(item)

# print(len(a))
import json
import os

# for title in data:
    # input_file = "squad_v2_clean/"+ title + ".json" 
input_file = "/home/test/RAG/vectorstores/neural_brige/neural.json" 
output_file = DB_FAISS_PATH + "/data.json"
print(input_file)
with open(input_file, "r", encoding="utf-8") as json_file:
    dataD = json.load(json_file)


for item in dataD:
    question = item['question']
    print(question)
    pre = qa.invoke({"query": question})
    split_strings = pre['result'].split("\nHelpful answer:")

    # Lấy phần tử cuối cùng
    prediction = split_strings[-1]
    print(question)
    contexts =  [doc.page_content for doc in retriever.get_relevant_documents(question)]
    item["ground_truth"] = item["ground_truth"]
    item["context_retriever"] = contexts
    item["answer"] = prediction

with open(output_file, 'w') as output:
    json.dump(dataD, output, indent=4)