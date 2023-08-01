import torch
import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"


# Load the model & tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt
query = "In 10 points, tell me why France is one of the best country."

# Create the inputs
inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(device)

# Steam the inputs
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

# Create the generate function kwargs
gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)

# Create a new thread to do non-blocking generation
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

# Stream the text
for text in streamer:
    print(text)
