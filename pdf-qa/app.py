import torch
import chromadb
from uuid import uuid4 as uuid
import streamlit as st
from threading import Thread
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tiiuae/falcon-7b-instruct"

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

# Chunk the text
def get_chunks(seq, size, overlap):
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    for i in range(0, len(seq) - overlap, size - overlap):
        yield seq[i:i + size]


# Add a pdf to Chroma
@st.cache_resource
def add_pdf(pdf_file):
    pdf = PdfReader(pdf_file)

    for page in pdf.pages:

        # Create chunks of the pdf
        chunks = get_chunks(page.extract_text(), 1024, 64)

        for chunk in chunks:

            # Add the chunks to the db
            collection.add(documents=[chunk], metadatas=[{"source": pdf_file.name, "page": page.page_number}], ids=[str(uuid())])


# Generate the response
def generate(query, container):
    global model, tokenizer

    # Create the inputs
    inputs = tokenizer(query, return_tensors="pt", return_token_type_ids=False).to(device)

    # Steam the inputs
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

    # Create the generate function kwargs
    gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500, temperature=.05)

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
    global collection

    st.title("Pdf QA")

    # Load a PDF
    pdf_file = st.file_uploader("Upload a PDF")

    # Add the PDF to the collection
    if pdf_file:
        add_pdf(pdf_file)

    query = st.text_input("Ask a question")

    # Create an empty container for the response stream
    container = st.empty()

    if query and query != "":

        # Query the database for results
        result = collection.query(query_texts=query, n_results=2)

        st.write("Source:")

        if result is None:
            return

        for i in range(2):
            with st.expander(f"Source {i}"):

                st.write(f"Source: {result['metadatas'][0][i]['source']}, Page: {result['metadatas'][0][i]['page']}")
                st.write("Texte:")
                st.write(result["documents"][0][i])
                st.write(f"Distance: {result['distances'][0][i]}")

        prompt = f"You are a smart assistant designed to help high school teachers come up with reading comprehension questions. \nGiven a piece of text, you must come up with a question and answer pair that can be used to test a student's reading comprehension abilities. \nWhen coming up with this question/answer pair, you must respond in the following format: 'Question: <question> Answer: <answer>' \nQuestion: {query} \nDocuments: {result['documents']}"

        generate(prompt, container)



if __name__ == "__main__":
    main()

