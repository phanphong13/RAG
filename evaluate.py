from datasets import Dataset 
from ragas.metrics import faithfulness
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from langchain_openai import ChatOpenAI

import os
import json
from dotenv import load_dotenv


# data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
data = {"question": [], "answer": [], "ground_truth": []}
DB_FAISS_PATH = "/home/test/RAG/vectorstores/neural_brige/"

input = DB_FAISS_PATH + "/fine_tuning.json"
with open(input, "r", encoding="utf-8") as json_file:
    data_pre = json.load(json_file)


print(data_pre)
for item in data_pre:
    ans = item['answer']
    # print(ans.split('/s>')[-1])
    # print(type(ans))
    data['question'].append(item['question']) 
    # data['answer'].append(item['answer']) 
    data['answer'].append(ans.split('/s>')[-1]) 
    # data['contexts'].append(item['context_retriever']) 
    data['ground_truth'].append(item['ground_truth']) 


dataset = Dataset.from_dict(data)

print(dataset)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

score = evaluate(dataset, llm = llm, metrics=[answer_similarity, answer_correctness]) # embedding
# score = evaluate(dataset, llm = llm, metrics=[faithfulness, answer_relevancy, context_relevancy]) # embedding
print(score)