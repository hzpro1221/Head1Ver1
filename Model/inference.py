import torch
from transformers import AutoTokenizer, AutoModel

from Model import Language_model, ModelBody


if __name__ == '__main__':
	text = input('Input text: ')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	model = ModelBody().to(device)
	model.load_checkpoint()

	tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased") 
	Language_model = Language_model().to(device)

	inputs = tokenizer(text, return_tensors=pt, padding='max_length', max_length=512).to(device)
	last_hidden_states = Language_model.forward(inputs)

	number_of_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])

	# Iterate through every token
	for i in range(number_of_tokens):

		current_token = last_hidden_states[0][i + 1].unsqueeze(0)

		prediction = model.forward(last_hidden_states, current_token)
		end_index = torch.argmax(prediction, dim=1)
		
		print(f"Token {i}: {end_index}")

