import streamlit as st
import re
import torch
import httpx
import pymupdf
from selectolax.parser import HTMLParser
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize


# Setup
st.set_page_config(page_title="Chatbot", layout="wide"); print("Initializing models...") 
generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', device='mps')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized")


# Scrape article using httpx + selectolax
def scrape_article(url):
    print("Fetching URL...")
    response = httpx.get(url, timeout=30.0)
    html = HTMLParser(response.text)
    title = html.css_first('title').text(strip=True) if html.css_first('title') else 'No title'
    paragraphs = [node.text(strip=True) for node in html.css('p') if node.text()]
    return {
        "title": title,
        "content": "\n\n".join(paragraphs)
    }

# Read PDF content
def read_pdf(file):
    print("Reading PDF file...")
    text = ""
    pdf = pymupdf.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# Read TXT content
def read_txt(file):
    print("Reading TXT file...")
    return file.read().decode("utf-8")

# Chunking
def split_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Build FAISS index
def build_index(text):
    print("Building index & chunking...")
    chunks = split_text(text)
    embeddings = normalize(embedding_model.encode(chunks))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return chunks, index

# RAG Chat
def rag_chat(input_query, index, chunks):
    print("Performing RAG search...")
    query_embedding = normalize(embedding_model.encode([input_query]))
    D, I = index.search(np.array(query_embedding), k=3)
    retrieved_docs = [chunks[i] for i in I[0]]
    retrieved_context = "\n".join(retrieved_docs)

    trimmed = st.session_state.message_history[-MAX_TURNS*2:]
    messages = [{"role": "system", "content": f"Use the following context to answer the user's question:\n{retrieved_context}"}] + trimmed
    messages.append({"role": "user", "content": input_query})

    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    response = outputs[0]['generated_text'].split('<|assistant|>')[-1].strip()

    st.session_state.message_history.append({"role": "user", "content": input_query})
    st.session_state.message_history.append({"role": "assistant", "content": response})
    return response

# General Chat
def general_chat(input_query):
    print("Generating Response...")
    trimmed = st.session_state.message_history[-MAX_TURNS*2:]
    messages = [{"role": "system", "content": "You are a helpful assistant. Respond as a chatbot. Do not assume prior knowledge of the user or mention blog posts."}] + trimmed
    messages.append({"role": "user", "content": input_query})

    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.3, top_k=50, top_p=0.95)
    response = outputs[0]['generated_text'].split('<|assistant|>')[-1].strip()

    st.session_state.message_history.append({"role": "user", "content": input_query})
    st.session_state.message_history.append({"role": "assistant", "content": response})
    return response

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.ready = False
    st.session_state.url = ""
    st.session_state.title = ""
    st.session_state.message_history = []

MAX_TURNS = 3



# FRONT-END
st.title("Chatbot using TinyLlama & RAG")

with st.expander("Load a Document (URL / PDF / TXT)", expanded=False):
    url = st.text_input("Paste a URL", "")
    uploaded_file = st.file_uploader("Or upload a PDF/TXT file", type=["pdf", "txt"])

    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            content = ""
            title = ""
            if url:
                article = scrape_article(url)
                content = article["content"]
                title = article["title"]
            elif uploaded_file:
                title = uploaded_file.name
                if uploaded_file.name.endswith(".pdf"):
                    content = read_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    content = read_txt(uploaded_file)

            if content:
                chunks, index = build_index(content)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.ready = True
                st.session_state.title = title
                st.session_state.message_history = []
                st.success(f"Loaded: {title}")

# Chat input
user_input = st.text_input("Enter your message")

if user_input:
    if st.session_state.ready:
        response = rag_chat(user_input, st.session_state.index, st.session_state.chunks)
    else:
        response = general_chat(user_input)

# Show conversation
for msg in st.session_state.message_history:
    role = "Assistant" if msg["role"] == "assistant" else "User"
    st.markdown(f"**{role}:** {msg['content']}")


if st.button("Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    # st.experimental_rerun()
