from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA

import torch


DB_FAISS_PATH = "vectorstores1/db_faiss/"

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline

tokenizer = LlamaTokenizer.from_pretrained("TheBloke/stable-vicuna-13B-HF")

model = LlamaForCausalLM.from_pretrained("TheBloke/stable-vicuna-13B-HF",
                                            #   load_in_8bit=True,
                                            #   device_map='auto',
                                            #   torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True
                                              )

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

print(llm('What is the capital of Vietnam?'))

# # Define the path to the pre-trained model you want to use
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# # Create a dictionary with model configuration options, specifying to use the CPU for computations
# model_kwargs = {'device':'cpu'}

# # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
# encode_kwargs = {'normalize_embeddings': False}

# # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )

# db = FAISS.load_local(DB_FAISS_PATH,embeddings)

# question = "Who is Thomas Jefferson?"
# searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)
# retriever = db.as_retriever(search_kwargs={"k": 3})

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

# question = "Who is Thomas Jefferson?"
# result = qa.run({"query": question})
# print(result)