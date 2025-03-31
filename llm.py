import streamlit as st
import fitz  # PyMuPDF for PDF handling
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import tempfile
import os
from collections import defaultdict

# Set up API key for Google Gemini AI
API_KEY = "AIzaSyCmVgr0Nc3tT8tUWHjjWIpGA3dG6ofFHik"
genai.configure(api_key=API_KEY)

# Define a maximum character length for the context
MAX_CONTEXT_LENGTH = 2000

# Adjust the context dynamically based on its length
def truncate_context(context, max_length=MAX_CONTEXT_LENGTH):
    if len(context) > max_length:
        context = context[:max_length] + "..."
    return context

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Must match embedding model output size
faiss_index = faiss.IndexFlatL2(dimension)
doc_chunks, doc_sources = [], []

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to add data to FAISS index
def add_to_faiss(chunk, embedding, source):
    global faiss_index, doc_chunks, doc_sources
    doc_chunks.append(chunk)
    doc_sources.append(source)
    faiss_index.add(np.array([embedding], dtype="float32"))

# Search FAISS for most relevant document
def search_faiss(query_embedding, top_k=5):
    if len(doc_chunks) == 0:
        return [], None

    distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), top_k)
    doc_results = defaultdict(list)

    for i in indices[0]:
        if i < len(doc_chunks):
            doc_results[doc_sources[i]].append(doc_chunks[i])

    if doc_results:
        most_relevant_doc = max(doc_results, key=lambda k: len(doc_results[k]))
        return doc_results[most_relevant_doc], most_relevant_doc

    return [], None

# Streamlit UI
st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #000000; }
        .stTitle { color: #ffffff; text-align: center; font-size: 36px; font-weight: bold; }
        .answer-box { background-color: #333333; padding: 15px; border-radius: 10px; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='stTitle'>📄 AI-powered PDF Q&A</h1>", unsafe_allow_html=True)

# Instructions
st.markdown("<p class='stSubHeader'>Upload your PDFs and ask questions about the content.</p>", unsafe_allow_html=True)

# Upload PDFs Section
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Show progress bar during file processing
if uploaded_files:
    progress_bar = st.progress(0)  # Initialize the progress bar
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_pdf.name

        pdf_name = uploaded_file.name
        chunks = extract_text_from_pdf(temp_pdf_path)
        chunk_embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks if chunk.strip()]

        for chunk, embedding in zip(chunks, chunk_embeddings):
            add_to_faiss(chunk, embedding, pdf_name)
        
        # Remove the temporary file after processing
        os.remove(temp_pdf_path)
        
        # Update progress bar
        progress_bar.progress((idx + 1) / total_files)

    st.success(f"{len(uploaded_files)} PDFs uploaded and processed successfully!")

# Handle user input
user_question = st.text_input("Ask a question about the documents", placeholder="Type your question here...")

if user_question and API_KEY:
    # Create a placeholder for the "Thinking..." message
    thinking_message = st.empty()
    thinking_message.markdown("<h3 style='color: white;'>Thinking... Please wait while we generate the answer.</h3>", unsafe_allow_html=True)

    # Get the question embedding and search FAISS for the relevant documents
    question_embedding = embed_model.encode(user_question).tolist()
    results, source_doc = search_faiss(question_embedding, top_k=5)

    # Load Gemini AI Model
    model = genai.GenerativeModel("gemini-1.5-flash")

    if results:
        relevant_text = "\n".join(results)
        truncated_relevant_text = truncate_context(relevant_text)

        # Generate response with both PDF and general knowledge
        prompt = f"""
        You are an AI assistant. Answer the user's question based on the given PDF context along with your general explanation.
        *Context from {source_doc}:*  
        {truncated_relevant_text}

        *User Question:*  
        {user_question}

        """

        response = model.generate_content(prompt)

        # Clear the "Thinking..." message after generating the response
        thinking_message.empty()

        st.markdown(f"### Answer (From {source_doc} and General AI Knowledge):")
        st.markdown(f"<div class='answer-box'>{response.text}</div>", unsafe_allow_html=True)

    else:
        # No relevant document found, provide a general answer
        prompt = f"""
        The user has asked: {user_question}  
        No relevant context was found in the uploaded documents.  
        Please provide a general answer based on AI knowledge.
        """

        response = model.generate_content(prompt)

        # Clear the "Thinking..." message after generating the response
        thinking_message.empty()

        st.markdown("### Answer (General AI Knowledge):")
        st.markdown(f"<div class='answer-box'>{response.text}</div>", unsafe_allow_html=True)