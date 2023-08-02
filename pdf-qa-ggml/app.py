import torch
import chromadb
from uuid import uuid4 as uuid
import streamlit as st
from threading import Thread
from pypdf import PdfReader
from ctransformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = 'models/'
model_name = "ggml-model-gpt4all-falcon-q4_0.bin"

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
        model_path_or_repo_id=f"{model_dir}{model_name}", model_type="falcon"
    )

model = get_model()
model.config.stream = True


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
    global model

    # Create the inputs
    inputs = model.tokenize(query)

    text = ""
    for token in model.generate(inputs):
        container.empty()
        text += model.detokenize(token)
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

        st.write("Sources:")

        if result is None:
            return

        for i in range(2):
            with st.expander(f"Source {i}"):

                st.write(f"Source: {result['metadatas'][0][i]['source']}, Page: {result['metadatas'][0][i]['page']}")
                st.write("Text:")
                st.write(result["documents"][0][i])
                st.write(f"Distance: {result['distances'][0][i]}")


        prompt = f"From the context that you will be given, answer the following question: '{query}'. Context: {result['documents'][0]}"

        generate(prompt, container)



if __name__ == "__main__":
    main()

