import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Linear_layer(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(20*768, 512)

	def forward(self, vectors):
		x = vectors.view(1, 20*768)
		x = self.linear(x)
		return x

class AvgPool_layer(nn.Module): 
	def __init__(self):
		super().__init__()
		self.avgPool = nn.AvgPool2d((2, 1))

	def forward(self, vectors): 
		x = self.avgPool(vectors)
		return x

class Convolution_layer(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(1, 20, 5)

	def forward(self, vectors):
		x = self.conv(vectors)
		return x

class Language_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = AutoModel.from_pretrained("bert-base-uncased")

		# Freeze parameter
		for param in self.model.parameters():
			param.requires_grad = False

	def forward(self, inputs):
		outputs = self.model(**inputs)
		last_hidden_states = outpus.last_hidden_state
		return last_hidden_states

class Condition_Feed_forward_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(768, 20)
		self.activation = nn.ReLU()
		self.layer2 = nn.Linear(20, 512*768)

	def forward(self, vectors):
		x = self.activation(self.layer1(vectors))
		x = self.layer2(x)
		return x

class Context_Feed_forward_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(512*768, 20)
		self.activation = nn.ReLU()
		self.layer2 = nn.Linear(20, 512*768)

	def forward(self, last_hidden_states):
		x = last_hidden_states.view(1, 512*768)
		x = self.activation(self.layer1(x))
		x = self.layer2(x)
		return x

class ModelBody(nn.Module):
	def __init__(self):
		super().__init__()
		self.Condition_Feed_forward_block = Condition_Feed_forward_block()
		self.Convolution_layer = Convolution_layer()
		self.AvgPool_layers = nn.ModuleList([AvgPool_layer() for _ in range(9)]) # 2^9 = 512
		self.Linear_layer = Linear_layer()

	def forward(self, added_last_hidden_states, current_token_pointed):
		x = self.Condition_Feed_forward_block.forward(current_token_pointed).view(1, 512, 768)
		x = x + added_last_hidden_state
		x = self.Convolution_layer.forward(x)
		for AvgPool_layer in AvgPool_layers:
			x = AvgPool_layer.forward(x)
		x = x.view(1, 20*768)
		x = self.Linear_layer.forward(x)
		return x

# Inference Module
# class Model(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.ModelBody = ModelBody()
# 		self.Language_model = Language_model()
# 		self.Context_Feed_forward_block = Context_Feed_forward_block()

# 	def forward(self, text):
# 		last_hidden_states = self.Language_model.forward(text)
# 		Context_vectors = self.Context_Feed_forward_block.forward(last_hidden_states)
# 		x = last_hidden_states + Context_vectors 
# 		for token in last_hidden_states:
# 			predicted = self.ModelBody.forward(x, token)