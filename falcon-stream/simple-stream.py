import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"

# Load the model & tokenizer
print(f"Loading {model_name}")
load_start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

load_end = time.time()
print("Loaded in ", load_end - load_start, " seconds")

# Prompt
query = "In 10 points, tell me why France is one of the best country."

print("Generating inputs...")
gen_start = time.time()

# Create the inputs
inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(device)

# Steam the inputs
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

print("Generating outputs...")
outputs = model.generate(
    **inputs, streamer=streamer, eos_token_id=tokenizer.eos_token_id, max_new_tokens=600
)

print("Decoding outputs")
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

gen_end = time.time()
print("Generated in ", gen_end - gen_start, " seconds")
print(text[0])
