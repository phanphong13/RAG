import requests
import json
import torch
import os
from tqdm import tqdm

# Load the training dataset and take a look at it
# with open('train-v2.0.json', 'rb') as f:
#   squad = json.load(f)

# # print(squad['data'][0].keys())

# gr = -1

# def read_data(path):  
#   # load the json file
#   with open(path, 'rb') as f:
#     squad = json.load(f)

#   contexts = []
#   questions = []
#   answers = []

#   for group in squad['data']:
#     for passage in group['paragraphs']:
#       context = passage['context']
#       for qa in passage['qas']:
#         question = qa['question']
#         for answer in qa['answers']:
#           contexts.append(context)
#           questions.append(question)
#           answers.append(answer)

#   return contexts, questions, answers

# train_contexts, train_questions, train_answers = read_data('train-v2.0.json')
# valid_contexts, valid_questions, valid_answers = read_data('dev-v2.0.json')

# # # print a random question and answer
# # print(f'There are {len(train_questions)} questions')
# # print(train_questions[-10000])
# # print(train_answers[-10000])

# def add_end_idx(answers, contexts):
#   for answer, context in zip(answers, contexts):
#     gold_text = answer['text']
#     start_idx = answer['answer_start']
#     end_idx = start_idx + len(gold_text)

#     # sometimes squad answers are off by a character or two so we fix this
#     if context[start_idx:end_idx] == gold_text:
#       answer['answer_end'] = end_idx
#     elif context[start_idx-1:end_idx-1] == gold_text:
#       answer['answer_start'] = start_idx - 1
#       answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
#     elif context[start_idx-2:end_idx-2] == gold_text:
#       answer['answer_start'] = start_idx - 2
#       answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

# add_end_idx(train_answers, train_contexts)
# add_end_idx(valid_answers, valid_contexts)
  
# # # You can see that now we get the answer_end also
# # print(train_questions[-10000])
# # print(train_answers[-10000])


# from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
# valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)


# # print(train_encodings.keys())

# # no_of_encodings = len(train_encodings['input_ids'])
# # print(f'We have {no_of_encodings} context-question pairs')

# def add_token_positions(encodings, answers):
#   start_positions = []
#   end_positions = []
#   for i in range(len(answers)):
#     start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
#     end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

#     # if start position is None, the answer passage has been truncated
#     if start_positions[-1] is None:
#       start_positions[-1] = tokenizer.model_max_length
#     if end_positions[-1] is None:
#       end_positions[-1] = tokenizer.model_max_length

#   encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# add_token_positions(train_encodings, train_answers)
# add_token_positions(valid_encodings, valid_answers)

# # print(train_encodings['start_positions'][:10])

# class SQuAD_Dataset(torch.utils.data.Dataset):
#   def __init__(self, encodings):
#     self.encodings = encodings
#   def __getitem__(self, idx):
#     return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#   def __len__(self):
#     return len(self.encodings.input_ids)
  
# train_dataset = SQuAD_Dataset(train_encodings)
# valid_dataset = SQuAD_Dataset(valid_encodings)

# from torch.utils.data import DataLoader

# # Define the dataloaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=16)


# # print(train_loader)

# from transformers import BertForQuestionAnswering

# model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# # Check on the available device - use GPU
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f'Working on {device}')

# from transformers import AdamW

# N_EPOCHS = 5
# optim = AdamW(model.parameters(), lr=5e-5)

# model.to(device)
# model.train()

# for epoch in range(N_EPOCHS):
#   loop = tqdm(train_loader, leave=True)
#   for batch in loop:
#     optim.zero_grad()
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     start_positions = batch['start_positions'].to(device)
#     end_positions = batch['end_positions'].to(device)
#     outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
#     loss = outputs[0]
#     loss.backward()
#     optim.step()

#     loop.set_description(f'Epoch {epoch+1}')
#     loop.set_postfix(loss=loss.item())

# model_path = '/home/dungspringai/Documents/RAG/BERT-SQuAD'
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

from transformers import BertForQuestionAnswering, BertTokenizerFast

model_path = 'C:/Users/Admin/Documents/RAG/BERT-SQuAD'
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

model = model.to(device)

model.eval()

acc = []

# for batch in tqdm(valid_loader):
#   with torch.no_grad():
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     start_true = batch['start_positions'].to(device)
#     end_true = batch['end_positions'].to(device)
    
#     outputs = model(input_ids, attention_mask=attention_mask)

#     start_pred = torch.argmax(outputs['start_logits'], dim=1)
#     end_pred = torch.argmax(outputs['end_logits'], dim=1)

#     acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
#     acc.append(((end_pred == end_true).sum()/len(end_pred)).item())

# acc = sum(acc)/len(acc)

# print("\n\nT/P\tanswer_start\tanswer_end\n")
# for i in range(len(start_true)):
#   print(f"true\t{start_true[i]}\t{end_true[i]}\n"
#         f"pred\t{start_pred[i]}\t{end_pred[i]}\n")


def get_prediction(context, question):
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
  outputs = model(**inputs)
  
  answer_start = torch.argmax(outputs[0])  
  answer_end = torch.argmax(outputs[1]) + 1 
  
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
  
  return answer


import json

# Đọc dữ liệu từ tệp JSON
json_filename = "C:/Users/Admin/Documents/RAG/squad2_data_by_title.json"
with open(json_filename, "r", encoding="utf-8") as json_file:
    data_by_title = json.load(json_file)

# print(data_by_title)

for title, data in data_by_title.items():
  title_data = data_by_title[title]
  # print(title_data)

  context = title_data['context']
  # Chuẩn bị dữ liệu để lưu vào tệp JSON mới
  output_data = {
      "title": title,
      "context": context,
      "qas": []
  }

  

  # Ví dụ giả sử bạn có một dự đoán từ mô hình ngôn ngữ (LLM)
  # llm_prediction = "This is a sample LLM prediction."

  # Thêm câu hỏi, câu trả lời và dự đoán từ LLM vào output_data
  for item in title_data['qas']:
      question = item['question']
      prediction = get_prediction(context, question)
      output_data["qas"].append({
          "question": item['question'],
          "answer": item['answer'],
          "llm_prediction": prediction
      })

  # Lưu output_data vào một tệp JSON mới với tên khác nhau
  output_filename = title + "_data_bert.json"
  with open(output_filename, "w", encoding="utf-8") as output_file:
      json.dump(output_data, output_file, ensure_ascii=False, indent=2)

  print(f"Data saved to {output_filename}")