from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def Porcess_Data(Data, tokenizer):
	Porcessed_Data = []
	for sample in Data:
		list_token = []
		list_index = [] 
		for token in sample["tokens"]:
			id = tokenizer(token, add_special_tokens=False)["input_ids"]
			list_token.append(id)
			list_index += id
		list_enities = [] 
		for entity in sample["entities"]:
			start = 0
			end = 0
			padding_start = 0
			padding_end = 0

			if entity["start"] != 0:
				for index in range(entity["start"]):
					padding_start += len(list_token[index])
			for index in range(entity["start"], entity["end"]):
				padding_end += len(list_token[index]) 
			start += padding_start
			end = start + padding_end
			list_enities.append({"start": start, "end": end})
		Porcessed_Data.append({"tokens": list_index, "entities": list_enities})
	return Porcessed_Data
# Format: 
# {'tokens': [3780, 1036, 7607, 1005, 1057, 1012, 1055, 1012, 5426, 2930, 2824, 13109, 16932, 2692, 28332, 15136, 2683, 2549, 15278, 2557, 2128, 4135, 3501, 2897, 1999, 3009, 12875, 2692, 13938, 2102, 2410, 13114, 6365], 
# 'entities': [{'start': 4, 'end': 8}, {'start': 18, 'end': 19}, {'start': 19, 'end': 24}, {'start': 26, 'end': 30}, {'start': 30, 'end': 33}]}

class CustomDataset(Dataset):
	def __init__(self, tokens_index, labels):
		self.tokens_index = tokens_index
		self.labels = labels

	def __len__(self):
		return len(self.tokens_index)

	def __getitem__(self, idx):
		return {"index": self.tokens_index[idx], "label": self.labels[idx]}