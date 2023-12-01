# python -m venv venv
# source venv/bin/activate

# pip install google-generativeai
# pip install langchain
# pip install pypdf
# pip install jupyter


import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyA5_lv3R2zBI0mKxFjj16zH71lXwm3CFmQ"


from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import google.generativeai
import osjjjjjjjjjjjjjj

# google_api_key=os.getenv('GOOGLE_API_KEY')
google_api_key="AIzaSyA5_lv3R2zBI0mKxFjj16zH71lXwm3CFmQ"


llm = GooglePalm(google_api_key=google_api_key)
llm.temperature = 0.1

prompts = ['Explain the difference between effective and affective with examples']
llm_result = llm._generate(prompts)

print(llm_result.generations[0][0].text)