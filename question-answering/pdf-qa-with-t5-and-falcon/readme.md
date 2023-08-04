# PDFs question answering

## How does it work?

In this usecase, we're using two models:

1. T5-base model for question answering
2. Falcon model for text generation

The T5-base model is used to extract the answer from the Chroma returns. Then, the T5's output is fed to the Falcon model to generate the final answer.

Chroma is a vector database that handle the embeddings and queries, we're storing our data in it.

## How to use it?

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Download the falcon model

Download the ```ggml-model-gpt4all-falcon-q4_0.bin```.
(Can be found at https://gpt4all.io)

Then, put it in the models folder.

### 2. Run streamlit

```bash
streamlit run app.py
```