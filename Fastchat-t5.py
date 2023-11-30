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

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch



from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

print(llm('What is the capital of England?'))

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

db = FAISS.load_local(DB_FAISS_PATH,embeddings)

question = "What is cheesemaking?"
searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

question = "What is cheesemaking?"
result = qa.run({"query": question})
print(result)