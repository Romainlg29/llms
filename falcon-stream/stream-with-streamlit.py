import torch
import time
import streamlit as st
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"


# Load the model
@st.cache_resource
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )


model = load_model()


# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(model_name)


tokenizer = load_tokenizer()


def generate(query, container):
    global model, tokenizer

    # Create the inputs
    inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(
        device
    )

    # Steam the inputs
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

    # Create the generate function kwargs
    gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500)

    # Create a new thread to do non-blocking generation
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    text = ""

    # Stream the text to streamlit
    for token in streamer:
        container.empty()
        text += token
        container.write(text)


def main():
    st.title("Streaming Demo")

    st.write(f"Using model: {model_name}")
    st.write("Ask any question below: ")

    # The user query
    query = st.text_input("Ask a question")

    # Container to stream the text to
    container = st.empty()

    if query and query != "":
        generate(query, container)


if __name__ == "__main__":
    main()
