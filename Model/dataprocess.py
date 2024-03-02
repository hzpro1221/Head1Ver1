from transformers import AutoTokenizer

def Porcess_Data(Data, tokenizer):
	Porcessed_Data = []
	for sample in Data:
		list_token = []
		list_index = [] # return
		for token in sample["tokens"]:
			id = tokenizer(token, add_special_tokens=False)["input_ids"]
			list_token.append(id)
			list_index += id
		list_enities = [] # return
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
