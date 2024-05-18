import torch
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.document import Document

DB_FAISS_PATH = "/home/test/RAG/vectorstores/squad"

from datasets import load_dataset

data_squad_v2 = load_dataset("squad_v2", split="validation")
data_squad_v2 = data_squad_v2['context']

print(data_squad_v2)

def remove_duplicates(list_str):
  new_list = set(list_str)
  return list(new_list)

data_squad_v2 = remove_duplicates(data_squad_v2)

data =list(map(lambda string: Document(page_content=string, metadata={}), data_squad_v2))

print(len(data))
# print(data)
# print(data)


# dataset_name = "squad_v2"
# page_content_column = "context"  # or any other column you're interested in

# # Create a loader instance
# loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# # Load the data
# val_data = loader.load()

# data1 = val_data[130319:]

# print(data1)

#####  RecursiveCharacterTextSplitter ****

DB_FAISS_PATH = "/home/test/RAG/vectorstores/squad/chunk_size/size512/over128"

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)

# print(docs)

# Define the path to the pre-trained model you want to use
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"
modelPath = "intfloat/e5-base-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {'device': device}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

##### End recursive ###


###### Semantic ######


# # Define the path to the pre-trained model you want to use
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# # modelPath = "intfloat/e5-base-v2"

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

# text_splitter = SemanticChunker(embeddings)
# docs = text_splitter.create_documents(data_squad_v2)

# print(docs)

##### End semantic


#### Spacy

# import spacy
# spacy.load("en_core_web_sm")

# from langchain_text_splitters import SpacyTextSplitter
# text_splitter = SpacyTextSplitter(chunk_size=1000, chunk_overlap = 150)

# docs = text_splitter.split_documents(data)

# print(docs)
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
##### End spacy



##### NLTK
# import nltk
# nltk.download('punkt')

# from langchain_text_splitters import NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=1024, chunk_overlap = 128)

# docs = text_splitter.split_documents(data)

# print(docs)

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

# END NLTK


db = FAISS.from_documents(docs, embeddings)
db.save_local(DB_FAISS_PATH)