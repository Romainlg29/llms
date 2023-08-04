import torch
import chromadb
from uuid import uuid4 as uuid
import streamlit as st
from pypdf import PdfReader
from ctransformers import AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = 'models/'
model_name = "ggml-model-gpt4all-falcon-q4_0.bin"

qa_model_name = "google/flan-t5-base"

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

# Create the generation model
@st.cache_resource
def get_gen_model():
    return AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=f"{model_dir}{model_name}", model_type="falcon"
    )

gen_model = get_gen_model()

# Create the qa model
@st.cache_resource
def get_qa_model():
    return T5ForConditionalGeneration.from_pretrained(qa_model_name, device_map="auto")

qa_model = get_qa_model()


# Create the qa tokenizer
@st.cache_resource
def get_qa_tokenizer():
    return T5Tokenizer.from_pretrained(qa_model_name, legacy=False)

qa_tokenizer = get_qa_tokenizer()


# Chunk the text
def get_chunks(seq, size, overlap):
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    for i in range(0, len(seq) - overlap, size - overlap):
        yield seq[i:i + size]


# Split the text into paragraphs
def to_paragraphs(text):
    return list(filter(lambda x : x != '', text.split('\n\n')))


# Add a pdf to Chroma
@st.cache_resource
def add_pdf(pdf_file):
    pdf = PdfReader(pdf_file)

    for page in pdf.pages:

        # Create chunks of the pdf on paragraph to keep a maximum of context
        paragraphs = to_paragraphs(page.extract_text())

        # if the paragraph is too long, split it into chunks
        chunks = []
        for paragraph in paragraphs:
            chunks.extend(list(get_chunks(paragraph, 1000, 100)))

        for chunk in chunks:

            # skip the small chunks with less than 10 characters
            # this is to avoid adding data with no context
            if len(chunk) < 10:
                continue

            # Add the chunks to the db
            collection.add(documents=[chunk], metadatas=[{"source": pdf_file.name, "page": page.page_number}], ids=[str(uuid())])


# Generate the response
@st.cache_data
def generate(query, documents):
    global qa_model, qa_tokenizer, gen_model

    # Create an empty container for the response stream
    container = st.empty()

    # Use the qa model to extract the answer from the documents
    qa_prompt = f"Answer the following question: '{query}' with the context: {documents}"
    qa_inputs = qa_tokenizer(qa_prompt, return_tensors="pt").input_ids.to(device)

    qa_generation = qa_model.generate(qa_inputs, max_new_tokens=300)
    qa_result = qa_tokenizer.decode(qa_generation[0], skip_special_tokens=True)

    # Use the generation model to generate the response
    gen_prompt = f"Generate an answer to the question: '{query}' by using the response: '{qa_result}'"
    text = ""

    print(gen_prompt)

    for token in gen_model(gen_prompt, max_new_tokens=600, stream=True):
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

    if st.button("Ask") and query != "":

        # Query the database for results
        result = collection.query(query_texts=query, n_results=2)

        if result is None:
            st.write("No results found")
            return

        # Generate the response
        generate(query, result['documents'])

        st.write("Sources:")
        for i in range(2):
            with st.expander(f"Source {i}"):

                st.write(f"Source: {result['metadatas'][0][i]['source']}, Page: {result['metadatas'][0][i]['page']}")
                st.write("Text:")
                st.write(result["documents"][0][i])



if __name__ == "__main__":
    main()

