import torch
from threading import Thread
from transformers import T5Tokenizer, T5ForConditionalGeneration, TextIteratorStreamer

# Check the gpu availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use a qa model
model_name = "google/flan-t5-base"

# Load the model and the tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

# Pre-gen a prompt
prompt = "Answer the following question: What's Romain's favorite color? Context: Romain's favorite color is green."

# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Create a streamer
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

# Create the args
gen_kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=500)

# Start the generation
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

# Iterate over the streamer
for text in streamer:

    if text is None or text == "":
        continue

    print(text)
