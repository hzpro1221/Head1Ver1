import sys
sys.path.append('/content/Head1Ver1')

import json
from transformers import AutoTokenizer, AutoModel

from inference import predict
from dataprocess import Process_Data


if __name__ == '__main__':
	# Open dataset folder
	with open("/content/conll04_dev.json") as f:
		validation_data = json.load(f)

	tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased") # BERT
	processed_data = Process_Data(validation_data, tokenizer)

	total = 0
	score = 0

	for i, sample in enumerate(processed_data):
		print(f"document {i}")		
		list_entities = []
		for entity in sample["entities"]:
			list_entities.append(f'Start: {entity["start"]}, End: {entity["end"]}')
		total += len(sample["entities"])

		prediction_text, prediction_start_end = predict(text="", list_token_processed=sample["tokens"])

		for entity in list_entities:
			if  (entity in prediction_start_end):
				score += 1
		if (i > 50):
			break

	print(f"Validation result: {(score*100)/total}%")