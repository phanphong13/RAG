import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.load import dumps, loads
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain


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


original_input_user_query = 'Who is the music director of the Quebec Symphony Orchestra?'
num_queries = 4
# Define the prompt template for generating search queries based on the original input query
query_gen_prompt_str = """[INST] <<SYS>>
You are a helpful assistant that generates multiple search queries based on a single input query. Generate {num_queries} search queries, one on each line, related to the given input query
Only provide the output containing the new similar search queries. Exclude any prefix or suffix statements.
<</SYS>>
Input Query:
{query}
[/INST] 
"""

# original_input_user_query = "Who is the music director of the Quebec Symphony Orchestra?"

# Instantiate a PromptTemplate object using the prompt template string
query_gen_prompt = PromptTemplate.from_template(query_gen_prompt_str)

# # Generate the input prompt for the LLM based on the original query and the number of queries to generate
# input_prompt = query_gen_prompt.format(num_queries = num_queries, query = original_input_user_query)
# # Generate search queries using watsonx.ai LLM for the given input prompt
# pre = llm.generate([input_prompt])
# results = pre.generations[0][0].text.split("\n")

# queries = [original_input_user_query]
# results = results[8:]
# for i in results:
#     i = i.split(". ")[1]
#     queries.append(i)
DB_FAISS_PATH = "/home/test/RAG/vectorstores/squad/"

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

def get_similiar_docs(index_store , query , k=5):
    similar_docs = index_store.similarity_search_with_score(query, k=k)
    return similar_docs
# retriever = db.as_retriever(search_kwargs={"k": 5})

def get_all_similar_docs(index_store, queries):
    all_similar_docs = {}
    for query in queries:
        search_results = get_similiar_docs(index_store, query)
        all_similar_docs[query] = search_results
    return all_similar_docs


# all_similar_docs  = get_all_similar_docs(db, queries)


class HashableDocument:
    def __init__(self, document, score):
        self.document = document
        self.score = score

    def __hash__(self):
        # Hash based on the document's content and score
        return hash((self.document.page_content, self.score))

    def __eq__(self, other):
        # Check for equality based on the document's content
        return isinstance(other, HashableDocument) and \
               self.document.page_content == other.document.page_content

def reciprocal_rank_fusion(search_results_dict, k=60, top_n_nodes=3):
    fused_scores = {}
    # print("Initial individual search result ranks:")

    # for query_description, doc_scores_list in search_results_dict.items():
    #     print(f"For query '{query_description}': {doc_scores_list}")
    # print()

    for query_description, doc_scores_list in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores_list, key=lambda x: x[1], reverse=True)):
            hashable_doc = HashableDocument(doc, score)
            if hashable_doc.document.page_content not in fused_scores:
                fused_scores[hashable_doc.document.page_content] = 0
            previous_score = fused_scores[hashable_doc.document.page_content]
            fused_scores[hashable_doc.document.page_content] += 1 / (rank + k)
            # print(f"Updating score for {doc} from {previous_score} to {fused_scores[hashable_doc.document.page_content]} based on rank {rank} in query '{query_description}'")

    reranked_results = [Document(page_content=page_content)
                        for page_content, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    # print("\nFinal reranked results:", reranked_results)
    return reranked_results[:top_n_nodes]
    # return reranked_results


# final_docs = reciprocal_rank_fusion(all_similar_docs)

# output_context = [doc.page_content for doc in final_docs]

zero_shot_prompt='''Use the following pieces of information to answer the users question. 

Context:{context}
Question:{question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
'''
z_prompt = PromptTemplate(template = zero_shot_prompt, input_variables=["context", "question"])

final_rag_chain = (z_prompt
    | llm
    | StrOutputParser()
)

# res = final_rag_chain.invoke({"context":output_context,"question":original_input_user_query})
# print(type(res))

import json
input_file = "/home/test/RAG/vectorstores/squad/squad.json" 
output_file = DB_FAISS_PATH + "/data_fusion.json"
print(input_file)
with open(input_file, "r", encoding="utf-8") as json_file:
    dataD = json.load(json_file)


for item in dataD:
    original_input_user_query = item['question']
    print(original_input_user_query)
    # Generate the input prompt for the LLM based on the original query and the number of queries to generate
    input_prompt = query_gen_prompt.format(num_queries = num_queries, query = original_input_user_query)
    # Generate search queries using watsonx.ai LLM for the given input prompt
    pre = llm.generate([input_prompt])
    results = pre.generations[0][0].text.split("\n")

    queries = [original_input_user_query]
    results = results[8:]
    for i in results:
        # i = i.split(". ")[1]
        queries.append(i)
    all_similar_docs  = get_all_similar_docs(db, queries)
    final_docs = reciprocal_rank_fusion(all_similar_docs)

    output_context = [doc.page_content for doc in final_docs]

    res = final_rag_chain.invoke({"context":output_context,"question":original_input_user_query})
    split_strings = res.split("\nHelpful answer:")

    # Lấy phần tử cuối cùng
    prediction = split_strings[-1]
    # print(question)
    # contexts =  [doc.page_content for doc in retriever.get_relevant_documents(question)]
    item["ground_truth"] = item["ground_truth"]
    item["context_retriever"] = output_context
    item["answer"] = prediction

with open(output_file, 'w') as output:
    json.dump(dataD, output, indent=4)