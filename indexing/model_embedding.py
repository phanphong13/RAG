import torch
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.document import Document


from datasets import load_dataset

data_squad_v2 = load_dataset("neural-bridge/rag-dataset-12000", split="test")
data_squad_v2 = data_squad_v2['context']

print(data_squad_v2)

def remove_duplicates(list_str):
  new_list = set(list_str)
  return list(new_list)

data_squad_v2 = remove_duplicates(data_squad_v2)

data =list(map(lambda string: Document(page_content=string, metadata={}), data_squad_v2))

# data = data[:100]

print(len(data))

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)

# print(docs)

##### Mini

# DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/embedding/minilm"
# # DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/chunk_size/size512/over256"

# # Define the path to the pre-trained model you want to use
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# # Create a dictionary with model configuration options, specifying to use the CPU for computations
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_kwargs = {'device': device}

# # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
# encode_kwargs = {'normalize_embeddings': False}

# # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )

##### End mini


##### Cohere

# DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/embedding/cohere"
# # DB_FAISS_PATH = "vectorstores/vectorstores_squad/embedding/cohere"

# import os
# os.environ['COHERE_API_KEY'] = 'k0BwnuzL2icVhcnJY2FOvTBy5jx3bwpV6QxWHvR8'
# from langchain_cohere import CohereEmbeddings
# embeddings = CohereEmbeddings(model="embed-english-v3.0")


#### End cohere


#### e5

# DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/embedding/e5"


# # DB_FAISS_PATH = "vectorstores/vectorstores_squad/embedding/e5"

# modelPath = "intfloat/e5-base-v2"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_kwargs = {'device': device}

# encode_kwargs = {'normalize_embeddings': False}

# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     
#     model_kwargs=model_kwargs, 
#     encode_kwargs=encode_kwargs
# )

#### End e5


#### openai

# from langchain.embeddings.openai import OpenAIEmbeddings
# DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/embedding/openai"


# # DB_FAISS_PATH = "vectorstores/vectorstores_squad/embedding/openai"

# import os

# os.environ["OPENAI_API_KEY"] = ""

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

### end openai

db = FAISS.from_documents(docs, embeddings)
db.save_local(DB_FAISS_PATH)
