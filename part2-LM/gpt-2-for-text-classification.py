# This part require finetuing
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model_config = GPT2Config.from_pretrained('gpt2', num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained(model_config)

# Define input text
input_text = "This is an example input text for classification."
input_ids = tokenizer.encode(input_text, return_tensors='pt')
outputs = model(input_ids)

predicted_class = torch.argmax(outputs[0])

print(predicted_class)