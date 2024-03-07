import torch
from transformers import AutoTokenizer, AutoModel

from Model import Language_model, ModelBody


def predict(text=None, list_token_processed=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = ModelBody().to(device)
	model.load_checkpoint()

	tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased") 
	Language_model = Language_model().to(device)

	if (list_token_processed=None):
		list_token = tokenizer(text, add_special_tokens=False)["input_ids"] + [100] # +[UNK]
	else:
		list_token = list_token_processed + [100]

	max_sequence_len = 512
	sample_len  = len(list_token) + 2 # [CLS], [SEP]

	inputs_ids = [101] + sample["tokens"] + [102] + [0 for _ in range(max_sequence_len - sample_len)]
	input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

	token_types_ids = [0 for _ in range(max_sequence_len)]
	token_types_ids = torch.tensor(token_types_ids).unsqueeze(0).to(device)

	attention_mask = [1 for _ in range(sample_len)] + [0 for _ in range(max_sequence_len - sample_len)]
	attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

	inputs = {"input_ids": input_ids, "token_type_ids": token_types_ids, "attention_mask": attention_mask}
	last_hidden_states = Language_model.forward(inputs)

	prediction_text = []
	prediction_start_end = []
	# Iterate through every token
	for i in range(sample_len - 2):

		current_token = last_hidden_states[0][i + 1].unsqueeze(0)

		prediction = model.forward(last_hidden_states, current_token)
		end_index = torch.argmax(prediction, dim=1)
		if (list_token[end_index] != 100):
			if (i > end_index):
				prediction_text.append([list_token[j] for j in range(end_index, i+1)])
				prediction_start_end.append(f"Start: {j}, End: {i + 1}")
			else:
				prediction_text.append([list_token[j] for j in range(i, end_index+1)])
				prediction_start_end.append(f"Start: {i}, End: {j + 1}")
	return prediction_text, prediction_start_end


if __name__ == '__main__':
	text = input('Input text: ')
	prediction_text, prediction_start_end = predict(text)
	for i, result in enumerate(prediction_text):
		print(f"Result {i}: {result}")
