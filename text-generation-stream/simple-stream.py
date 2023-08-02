import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt
query = "In 10 points, tell me why France is one of the best country."

# Create the inputs
inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(device)

# Steam the inputs to the stdout
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

# Generate the outputs
outputs = model.generate(
    **inputs, streamer=streamer, eos_token_id=tokenizer.eos_token_id, max_new_tokens=600
)

# Decode the outputs
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the outputs at the end
print(text[0])
