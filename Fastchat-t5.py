from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA

import torch


DB_FAISS_PATH = "vectorstores/db_faiss/"

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

# question = "What is the original meaning of the word Norman?"
# searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

# question = "What is the original meaning of the word Norman?"
# result = qa.run({"query": question})
# print(type(result))
# import json

# output_data = {
#         "title": "Normans",
#         "context": "Test",
#         "qas": []
#     }

# output_data["qas"].append({
#             "question": question,
#             "answer": 'test',
#             "llm_prediction": result
#         })

# # Lưu output_data vào một tệp JSON mới với tên khác nhau
# output_filename = "Normans_data_with_prediction.json"
# with open(output_filename, "w", encoding="utf-8") as output_file:
#     json.dump(output_data, output_file, ensure_ascii=False, indent=2)


import json

# Đọc dữ liệu từ tệp JSON
json_filename = "squad2_data_by_title.json"
with open(json_filename, "r", encoding="utf-8") as json_file:
    data_by_title = json.load(json_file)

# Kiểm tra xem title "Normans" có tồn tại trong dữ liệu hay không
if "Normans" in data_by_title:
    # Lấy thông tin cho title "Normans"
    title_data = data_by_title["Normans"]

    # Chuẩn bị dữ liệu để lưu vào tệp JSON mới
    output_data = {
        "title": "Normans",
        "context": title_data['context'],
        "qas": []
    }

    
   
    # Ví dụ giả sử bạn có một dự đoán từ mô hình ngôn ngữ (LLM)
    # llm_prediction = "This is a sample LLM prediction."

    # Thêm câu hỏi, câu trả lời và dự đoán từ LLM vào output_data
    for item in title_data['qas']:
        question = item['question']
        prediction = qa.run({"query": question})
        output_data["qas"].append({
            "question": item['question'],
            "answer": item['answer'],
            "llm_prediction": prediction
        })

    # Lưu output_data vào một tệp JSON mới với tên khác nhau
    output_filename = "Normans_data_with_prediction.json"
    with open(output_filename, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    print(f"Data saved to {output_filename}")
else:
    print("Title 'Normans' not found in the data.")

