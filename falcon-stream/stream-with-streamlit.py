import torch
import time
import streamlit as st
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"

model = None
tokenizer = None


def load():
    global model, tokenizer

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)



def generate(query, container):
    global model, tokenizer

    # Create the inputs
    inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(device)

    # Steam the inputs
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

    # Create the generate function kwargs
    gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500)

    # Create a new thread to do non-blocking generation
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    text = ""

    # Stream the text
    for token in streamer:
        container.empty()
        text += token
        container.write(text)


def main():
    st.title("Streaming Demo")

    st.write("Ask a question below: ")
    query = st.text_input("Ask a question")

    container = st.empty()

    if query and query != "":
        load()

        generate(query, container)



if __name__ == "__main__":
    main()

