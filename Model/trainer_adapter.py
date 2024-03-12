# Set system path for Google Colab
import sys
sys.path.append('/content/Head1Ver1')

import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from dataprocess import Process_Data, CustomDataset, CustomCollateFunction
from Model import Language_model, ModelBody

if __name__ == '__main__':
	print("start trainer_adapter.py")
	# Open dataset folder
	with open("/content/conll04_train.json") as f:
		train_data = json.load(f)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Hyperparameter
	batch_size = 8
	lr = 5e-5
	num_eps = 5

	# Language Model	
	tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased") # BERT
	language_model = Language_model(freeze=True).to(device)

	# Head 
	model = ModelBody().to(device)

	# loss
	loss = nn.CrossEntropyLoss()

	optim = AdamW(model.parameters(), lr=lr)

	processed_data = Process_Data(train_data, tokenizer)
	for epoch in range(num_eps):
		for i, sample in enumerate(processed_data):
			max_sequence_len = 512
			sample_len = len(sample["tokens"]) + 3 # [CLS], [UNK], [SEP]

			input_ids = [101] + sample["tokens"] + [100] + [102] + [0 for _ in range(max_sequence_len - sample_len)]
			input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

			token_types_ids = [0 for _ in range(max_sequence_len)]
			token_types_ids = torch.tensor(token_types_ids).unsqueeze(0).to(device)

			attention_mask = [1 for _ in range(sample_len)] + [0 for _ in range(max_sequence_len - sample_len)]
			attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

			inputs = {"input_ids": input_ids, "token_type_ids": token_types_ids, "attention_mask": attention_mask}

			postive_token = []
			positive_token_label = []
			for entity in sample["entities"]:
				start = entity["start"] + 1 # + [CLS]
				end = entity["end"] + 1 # + [CLS]
				label_start = [0 for _ in range(end-1)] + [1] + [0 for _ in range(end, 512)] # Không phải 1 dải 1 nữa
				label_end = [0 for _ in range(start)] + [1] + [0 for _ in range(start+1, 512)] # Không phải 1 dải 1 nữa
				postive_token = postive_token + [start, end]
				positive_token_label = positive_token_label + [label_start, label_end]

			negative_token = []
			negative_token_label = [] 
			neg_label = [0 for _ in range(sample_len-2)] + [1] + [0 for _ in range(sample_len-1, 512)] # Trỏ đế [UNK]
			for index in range(sample_len - 2):
				if (index + 1) not in postive_token: # + [CLS]
					negative_token.append(index + 1)
					negative_token_label.append(neg_label)

			tokens_index = postive_token + negative_token
			labels = positive_token_label + negative_token_label

			dataset = CustomDataset(tokens_index=tokens_index, labels=labels)
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CustomCollateFunction)

			last_hidden_states = language_model.forward(inputs)

			for j, batch in enumerate(dataloader):
				optim.zero_grad()

				tokens_list = []
				for index in batch["list_index"]:
					zeros = [0 for _ in range(512)]
					zeros[index] = 1
					tokens_list.append(zeros)
				tokens_list = torch.tensor(tokens_list).float().to(device)
				
				tokens_stack = torch.matmul(tokens_list, last_hidden_states.view(512, 768)) # Shape: (batch_size, 768)
				labels_stack = batch["labels_stack"].to(device) # Shape: (batch_size, 512)

				logits = model.forward(last_hidden_states, tokens_stack)
				# print("logits shape: {logits.shape}")

				loss_value = loss(logits.float(), labels_stack.float())

				loss_value.backward()

				print(f"Epoch: {epoch}, Document: {i}, Batch: {j}, loss: {loss_value}")

				optim.step()   

	model.save_checkpoint()
