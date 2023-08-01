import torch
import chromadb
from uuid import uuid4 as uuid
import streamlit as st
from threading import Thread
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"  # "cerebras/btlm-3b-8k-base"

# Create a database connection
@st.cache_resource
def db():
    return chromadb.Client()

database = db()

# Create a collection
@st.cache_resource
def get_collection():
    return database.create_collection("pdfs")

collection = get_collection()

# Create a model
@st.cache_resource
def get_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

model = get_model()

# Create a tokenizer
@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(model_name)

tokenizer = get_tokenizer()

# Generate the response
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
    st.title("PDFs QA")

    # Load a PDF
    pdf_file = st.file_uploader("Upload a PDF")

    # Add the PDF to the collection
    if pdf_file:
        pdf = PdfReader(pdf_file)

        for page in pdf.pages:

            collection.add(documents=[page.extract_text()], metadatas=[{"source": pdf_file.name}], ids=[str(uuid())])

    st.write("Ask a question below: ")
    query = st.text_input("Ask a question")

    container = st.empty()

    if query and query != "":
        generate(query, container)
        #collection.query(query, n_results=2)



if __name__ == "__main__":
    main()

