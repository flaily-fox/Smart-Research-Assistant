# This file will handle everything related to text extraction, chunking, embeddings

# core/document_processor.py
import streamlit as st
import pypdf
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import model initialization (we'll define these in llm_functions or models.py)
# For now, let's pass them directly or import a specific init function
# For now, we'll assume generation_model and embedding_model are passed or globally available via app.py setup.

# --- Functions for Document Processing ---
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text() or ""
    except Exception as e:
        # Don't use st.error directly in core functions, return status/message
        # or raise specific exceptions to be handled by the caller (app.py)
        raise RuntimeError(f"Error reading PDF: {e}")
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Function to generate embeddings for a list of texts
# @st.cache_data removed here; caching should be handled by the caller (app.py)
def get_embeddings(texts, embedding_model_name):
    embeddings = []
    for text in texts:
        try:
            response = genai.embed_content(model=embedding_model_name, content=text, task_type="RETRIEVAL_DOCUMENT")
            embeddings.append(response['embedding'])
        except Exception as e:
            # We'll log this in a real app; for now, warn and skip
            print(f"Warning: Could not generate embedding for a text chunk (likely too long). Skipping this chunk. Error: {e}")
            embeddings.append(None) # Add None for problematic chunks
    return [e for e in embeddings if e is not None] # Filter out None values

# Function to find relevant chunks based on a query
def find_relevant_chunks(query_text, text_chunks, chunk_embeddings, embedding_model_name, top_k=3):
    if not chunk_embeddings:
        return [], [], []

    query_embedding_response = genai.embed_content(model=embedding_model_name, content=query_text, task_type="RETRIEVAL_QUERY")
    query_embedding = np.array(query_embedding_response['embedding']).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, np.array(chunk_embeddings))[0]

    sorted_indices = similarities.argsort()[::-1]

# Select top_k unique chunks based on sorted indices
    selected_indices = []
    for idx in sorted_indices:
        if text_chunks[idx] not in [text_chunks[s_idx] for s_idx in selected_indices]:
            selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break

    relevant_chunks = [text_chunks[i] for i in selected_indices]
    
    justifications = []
    highlight_snippets = []
    for i, chunk_idx in enumerate(selected_indices):
        chunk_content = text_chunks[chunk_idx]
        snippet = chunk_content[:150] + "..." if len(chunk_content) > 150 else chunk_content
        justifications.append(f"From document (snippet: \"{snippet}\")")
        highlight_snippets.append(chunk_content)

    return relevant_chunks, justifications, highlight_snippets