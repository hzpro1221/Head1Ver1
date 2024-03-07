import sys
sys.path.append('/content/Head1Ver1')

import json

from inference import predict

if __name__ == '__main__':
	# Open dataset folder
	with open("/content/conll04_dev.json") as f:
		validation_data = json.load(f)

	tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased") # BERT
	processed_data = Process_Data(validation_data, tokenizer)

	total = 0
	score = 0

	for sample in processed_data:
		list_entities = []
		for entity in sample["entities"]:
			list_entities.append(f"Start: {start}, End: {end}")
		total += len(sample["entities"])

		list_prediction = predict(text=None, list_token_processed=sample["tokens"])

		for entity in list_entities:
			if  (entity in list_prediction):
				score += 1

	print(f"Validation result: {(score*100)/total}%")