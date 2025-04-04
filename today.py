import streamlit as st
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import uuid
import base64
from io import BytesIO
import hashlib

# Initialize the app with beautiful UI
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 12px;
    }
    .stButton button {
        background-color: #4a6fa5;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        border: none;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #3a5a80;
        color: white;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 80%;
    }
    .user-message {
        background-color: rgba(250, 250, 250, 0.3);
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .bot-message {
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .source-container {
        margin-top: 10px;
        border-top: 1px solid #eee;
        padding-top: 10px;
    }
    .source-item {
        margin: 5px 0;
        font-size: 0.9em;
    }
    .chat-container {
        margin-bottom: 150px;
    }
    .download-btn {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="document_chunks")

# Create directories if they don't exist
os.makedirs("uploaded_pdfs", exist_ok=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Document processing functions
def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    
    # Save the PDF file locally
    file_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return text, file_path

def scrape_website(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        return ' '.join(soup.stripped_strings), url
    except Exception as e:
        st.error(f"Failed to scrape website: {str(e)}")
        return None, None

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_content(content, source_name, source_type, file_path=None):
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )["embedding"]
        
        metadata = {
            "source": str(source_name),
            "type": str(source_type)
        }
        
        if source_type == "pdf" and file_path is not None:
            metadata["file_path"] = str(file_path)
        elif source_type == "web":
            metadata["url"] = str(source_name)
        
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        collection.add(
            ids=[f"{source_type}_{uuid.uuid4().hex}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata]
        )


def get_relevant_content(query, n_results=3):
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )
    
    if results and results['documents']:
        combined_content = []
        sources = []
        seen_sources = set()
        
        # Sort results by distance (most relevant first)
        sorted_results = sorted(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]),
                              key=lambda x: x[2])
        
        for doc, metadata, distance in sorted_results:
            combined_content.append(doc)
            source_key = metadata.get("file_path", metadata.get("url", ""))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                safe_metadata = {
                    "source": metadata.get("source", "Unknown"),
                    "type": metadata.get("type", "unknown"),
                    "file_path": metadata.get("file_path", None),
                    "url": metadata.get("url", None),
                    "distance": float(distance)  # Store relevance score
                }
                sources.append(safe_metadata)
        
        return "\n\n".join(combined_content), sources
    
    return None, None

# Sidebar for document upload and chat history
with st.sidebar:
    st.header("📂 Document Management")
    
    tab1, tab2 = st.tabs(["PDF Upload", "Website Scraping"])
    
    with tab1:
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                for uploaded_file in uploaded_files:
                    text, file_path = extract_pdf_text(uploaded_file)
                    if text:
                        store_content(text, uploaded_file.name, "pdf", file_path)
                        st.success(f"Processed: {uploaded_file.name}")
    
    with tab2:
        url = st.text_input("Enter website URL")
        if st.button("Scrape Website"):
            if url:
                with st.spinner("Scraping website content..."):
                    text, source_url = scrape_website(url)
                    if text:
                        store_content(text, url, "web")
                        st.success(f"Scraped: {url}")
    
    st.divider()
    st.header("💬 Chat History")
    
    for i, chat in enumerate(st.session_state.chat_history):
        if chat['role'] == 'user':
            def make_callback(index):
                def callback():
                    st.session_state.current_question = st.session_state.chat_history[index]['content']
                return callback
            
            st.button(
                chat['content'][:50] + ("..." if len(chat['content']) > 50 else ""),
                key=f"history_{i}",
                on_click=make_callback(i),
                use_container_width=True
            )

# Main chat interface
st.header("🤖 DocuChat AI")

# Display chat history in main area
chat_container = st.container()
with chat_container:
    for chat_idx, chat in enumerate(st.session_state.chat_history):
        if chat['role'] == 'user':
            st.markdown(
                f"""
                <div class="chat-message user-message">
                    <div><strong>You:</strong> {chat['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message bot-message">
                    <div><strong>AI:</strong> {chat['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display download buttons for PDF sources used in this specific answer
            if chat.get('sources'):
                # Get only the sources used for this specific answer
                answer_sources = chat.get('sources', [])
                
                # Remove duplicates while preserving order
                unique_sources = []
                seen_sources = set()
                for source in answer_sources:
                    source_key = source.get("url") if source["type"] == "web" else source.get("file_path", "")
                    if source_key and source_key not in seen_sources:
                        seen_sources.add(source_key)
                        unique_sources.append(source)
                
                if unique_sources:
                    st.markdown('<div class="source-container">', unsafe_allow_html=True)
                    st.markdown("**Sources used in this answer:**")  # Updated label
                    
                    # Sort by relevance if available (lower distance = more relevant)
                    if all('distance' in source for source in unique_sources):
                        unique_sources.sort(key=lambda x: x['distance'])
                    
                    for source_idx, source in enumerate(unique_sources):
                        if source["type"] == "pdf" and source.get("file_path"):
                            with open(source["file_path"], "rb") as file:
                                # Generate unique key
                                key_string = f"{chat_idx}_{source_idx}_{source['file_path']}"
                                hashed_key = hashlib.sha256(key_string.encode()).hexdigest()
                                
                                # Add relevance score if available
                                label = f"📄 Download {source['source']}"
                                if 'distance' in source:
                                    relevance = 1/(source['distance'] + 1e-6)  # Avoid division by zero
                                    label += f" (relevance: {relevance:.2f})"
                                
                                st.download_button(
                                    label=label,
                                    data=file,
                                    file_name=source['source'],
                                    mime="application/pdf",
                                    key=f"download_{hashed_key}",
                                    use_container_width=True
                                )
                        elif source["type"] == "web" and source.get("url"):
                            # Add relevance score if available
                            link_text = f"🌐 {source['source']}"
                            if 'distance' in source:
                                relevance = 1/(source['distance'] + 1e-6)
                                link_text += f" (relevance: {relevance:.2f})"
                            
                            st.markdown(
                                f'<div class="source-item"><a href="{source["url"]}" target="_blank">{link_text}</a></div>',
                                unsafe_allow_html=True
                            )
                    st.markdown('</div>', unsafe_allow_html=True)

# ... (rest of the code remains the same)

# Fixed input at bottom
input_container = st.container()
with input_container:
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input(
                "Type your question here...",
                key="input",
                label_visibility="collapsed",
                placeholder="Ask anything about your documents..."
            )
        with col2:
            submitted = st.form_submit_button("➤", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if submitted and question:
    with st.spinner("Generating answer..."):
        # Add user question to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Get answer
        relevant_content, sources = get_relevant_content(question)
        
        if not relevant_content:
            answer = "I couldn't find relevant information in your documents to answer this question."
            sources = []
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Use the following content to answer the question. If you can't find the answer, say relevant content not available.
            
            Content sources: {[source['source'] for source in sources]}
            Content:
            {relevant_content}
            
            Question: {question}
            
            Provide a detailed answer and mention which document(s) the information came from.
            Answer:
            """
            
            response = model.generate_content(prompt)
            answer = response.text
        
        # Add AI response to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer,
            'sources': sources
        })
        
        # Auto-scroll to bottom
        auto_scroll_js = """
        <script>
            function scrollToBottom() {
                window.scrollTo(0, document.body.scrollHeight);
            }
            window.onload = scrollToBottom;
            window.parent.document.addEventListener('DOMSubtreeModified', function() {
                setTimeout(scrollToBottom, 100);
            });
        </script>
        """
        st.components.v1.html(auto_scroll_js)
        
        # Rerun to update the display
        st.rerun()
