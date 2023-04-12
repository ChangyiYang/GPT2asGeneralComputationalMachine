# This code demonstrates GPT-2's language task ability
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, pipeline, set_seed
# Please input the prompt you like
prompt = "Hello, I'm a language model," # TODO: replace with raiseError
generator = pipeline('text-generation', model='gpt2')
generated_result = generator(prompt, max_length=50, num_return_sequences=3)

for i, result in enumerate(generated_result):
    print("The {}th result begins:\n{}\nThe {}th result ends\n".format(i,result["generated_text"],i))

# Despite using pipeline directly provided by transformers, it is also good to take a closer look
prompt = "My dog is cute" # TODO: replace this with RaiseError 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# tokens is a dictionary of multiple GPT2 parameters 
tokens = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **tokens,
    max_length=50
)

# Convert the outputs into words
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)