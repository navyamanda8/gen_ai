import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import json
from datetime import datetime
import base64
from playwright.sync_api import sync_playwright
import time

st.set_page_config(page_title="PDF & Web Q&A", page_icon="📄", layout="wide")

load_dotenv()

# Load API Keys
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GENAI_API_KEY:
    raise ValueError("API key for Gemini AI is missing. Please set GENAI_API_KEY in your .env file.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")

# Initialize clients
genai.configure(api_key=GENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define Constants
SAVE_FOLDER = "saved_pdfs"
SCRAPED_DATA_FOLDER = "scraped_websites"
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(SCRAPED_DATA_FOLDER, exist_ok=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize session state
if "current_website" not in st.session_state:
    st.session_state.current_website = None
if "scraped_content" not in st.session_state:
    st.session_state.scraped_content = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Utility Functions
def clean_text(text):
    """Cleans and normalizes text to remove unwanted spaces, special characters, and extra lines."""
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces/newlines
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'"-]', ' ', text)  # Keep essential punctuation
    text = re.sub(r'\n+', '\n', text).strip()  # Remove excessive newlines
    return text

def save_scraped_data(url, content):
    """Save cleaned scraped content to a local JSON file"""
    domain = urlparse(url).netloc
    filename = f"{domain}_{uuid.uuid4().hex[:8]}.json"
    filepath = os.path.join(SCRAPED_DATA_FOLDER, filename)
    
    data = {
        "url": url,
        "content": content,
        "timestamp": str(datetime.now())
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath

def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_in_supabase(chunk, embedding, source, content_type="pdf"):
    try:
        data = {
            "id": str(uuid.uuid4()),
            "source": source,
            "text": chunk,
            "embedding": embedding,
            "content_type": content_type
        }
        response = supabase.table("pdf_documents").insert(data).execute()
        if hasattr(response, 'error') and response.error:
            st.error(f"Error storing document: {response.error}")
    except Exception as e:
        st.error(f"Error storing document in Supabase: {str(e)}")

def semantic_search(query_embedding, top_k=10):
    """Perform semantic search in Supabase"""
    try:
        # Convert the query embedding to a list
        query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Execute the search
        result = supabase.rpc('search_documents', {
            'query_embedding': query_embedding,
            'match_count': top_k
        }).execute()
        
        if hasattr(result, 'error') and result.error:
            st.error(f"Search error: {result.error}")
            return []
        
        return result.data if result.data else []
        
    except Exception as e:
        st.error(f"Error searching in Supabase: {str(e)}")
        return []

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scrape_with_playwright(url):
    """Scrape dynamic content using Playwright"""
    with sync_playwright() as p:
        # Launch browser (headless by default)
        browser = p.chromium.launch()
        page = browser.new_page()
        
        try:
            # Navigate to page and wait for content to load
            page.goto(url, timeout=60000)
            
            # Wait for the page to fully load (adjust as needed)
            page.wait_for_load_state("networkidle", timeout=60000)
            
            # Scroll to bottom to trigger lazy-loaded content
            page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 100;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            
                            if(totalHeight >= scrollHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                    });
                }
            """)
            
            # Wait a bit after scrolling
            time.sleep(2)
            
            # Get the page content
            html_content = page.content()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "noscript", "iframe", "footer", "header", "nav", "aside", "svg"]):
                tag.decompose()
            
            # Extract meaningful content
            main_content = ""
            content_tags = ["main", "article", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
            
            for tag in content_tags:
                elements = soup.find_all(tag)
                for element in elements:
                    if len(element.text.strip()) > 50:  # Ignore very short elements
                        main_content += element.get_text(separator="\n", strip=True) + "\n"
            
            if not main_content:
                main_content = soup.get_text(separator="\n", strip=True)  # Fallback to full text extraction
            
            # Clean the extracted text
            cleaned_content = clean_text(main_content)
            
            # Update session state
            st.session_state.current_website = url
            st.session_state.scraped_content = cleaned_content
            
            return cleaned_content, None
            
        except Exception as e:
            return None, f"Playwright Error: {str(e)}"
        finally:
            browser.close()

def scrape_website(url):
    """Try both traditional scraping and Playwright as fallback"""
    # First try traditional scraping (faster for static sites)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if this looks like a dynamic page (contains common JS framework markers)
        if "react-root" in response.text or "ng-app" in response.text or "vue-app" in response.text:
            raise Exception("Page appears to be dynamic, switching to Playwright")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "noscript", "iframe", "footer", "header", "nav", "aside"]):
            tag.decompose()

        # Extract meaningful content
        main_content = ""
        content_tags = ["main", "article", "section", "div", "p", "h1", "h2"]
        
        for tag in content_tags:
            elements = soup.find_all(tag)
            for element in elements:
                if len(element.text.strip()) > 50:  # Ignore very short elements
                    main_content += element.get_text(separator="\n", strip=True) + "\n"
        
        if not main_content:
            main_content = soup.get_text(separator="\n", strip=True)  # Fallback to full text extraction

        # Clean the extracted text
        cleaned_content = clean_text(main_content)
        
        # Update session state
        st.session_state.current_website = url
        st.session_state.scraped_content = cleaned_content
        
        return cleaned_content, None
        
    except Exception as initial_error:
        # If traditional scraping fails, try with Playwright
        st.warning(f"Traditional scraping failed, trying with Playwright: {str(initial_error)}")
        return scrape_with_playwright(url)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📄 AskDoc with Context</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📜 Current Context")
if st.session_state.current_website:
    st.sidebar.markdown(f"**Website:**\n{st.session_state.current_website}")
if st.session_state.uploaded_files:
    st.sidebar.markdown("**Uploaded PDFs:**")
    for file in st.session_state.uploaded_files:
        st.sidebar.markdown(f"- {file.name}")
st.sidebar.header("💬 Chat History")

for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":  # Show only user questions
        with st.sidebar.expander(f"Q{index+1}: {message['content'][:30]}..."):  # Display first 30 chars as title
            st.write(message["content"][:150] + "...")  # Show preview on click

# File Upload Section
st.header("📤 Upload Documents or Enter Website URL")
tab1, tab2 = st.tabs(["PDF Upload", "Website Scraping"])

with tab1:
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

    if uploaded_files and st.button("Process PDFs"):
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            save_path = os.path.join(SAVE_FOLDER, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            chunks = extract_text_from_pdf(save_path)
            
            for chunk in chunks:
                if chunk.strip():
                    embedding = embed_model.encode(chunk).tolist()
                    store_in_supabase(chunk, embedding, uploaded_file.name, "pdf")

            progress_bar.progress((idx + 1) / total_files)

        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} PDFs processed successfully!")

with tab2:
    website_url = st.text_input("Enter website URL to scrape", key="website_url")
    
    if st.button("Scrape Website", key="scrape_button"):
        if website_url:
            with st.spinner("Scraping and cleaning website content..."):
                scraped_content, error = scrape_website(website_url)
                
                if error:
                    st.error(f"Error scraping website: {error}")
                else:
                    # Save cleaned content to local file
                    filepath = save_scraped_data(website_url, scraped_content)
                    
                    # Process the scraped content (chunk and embed)
                    chunks = [scraped_content[i:i+500] for i in range(0, len(scraped_content), 500)]
                    
                    for chunk in chunks:
                        if chunk.strip():
                            embedding = embed_model.encode(chunk).tolist()
                            store_in_supabase(chunk, embedding, f"Website: {website_url}", "website")
                    
                    st.success(f"Successfully scraped {website_url}!")
        else:
            st.warning("Please enter a valid website URL")

# Chat Interface
st.markdown("## 💬 Ask a Question")

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source"):
            # Check if source is a PDF
            if isinstance(message["source"], str) and message["source"].endswith(".pdf"):
                pdf_path = os.path.join(SAVE_FOLDER, message["source"])
                if os.path.exists(pdf_path):
                    # Create a clickable link that opens in new tab
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode()}" '
                        f'download="{message["source"]}" target="_blank">📌 Source: {message["source"]}</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.caption(f"📌 Source: {message['source']} (file not found)")
            else:
                st.caption(f"📌 Source: {message['source']}")

# Handle User Input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if any files have been uploaded or websites scraped
    if not st.session_state.uploaded_files and not st.session_state.current_website:
        response_data = {
            "role": "assistant",
            "content": "Please upload PDF files or process a website URL first before asking questions."
        }
    else:
        # Generate AI Response only if documents exist
        model = genai.GenerativeModel("gemini-1.5-flash")
        query_embedding = embed_model.encode(prompt)
        
        # Search in Supabase
        search_results = semantic_search(query_embedding)
        
        if search_results:
            context = "\n".join([r['text'] for r in search_results])
            source = search_results[0]['source']
            
            ai_prompt = f"""
            **Instruction**: Analyze the given context carefully and provide an accurate response.
            
            **Context**:
            {context}
            
            **Question**: {prompt}
            
            **Rules**:

            - Provide answers strictly based on the context.
            - If the context provides a direct answer, explain it in detail and reference the relevant document section.
            - If partial information is available, summarize it while mentioning that full details are unavailable.
            - If no relevant information is found, clearly state: "This information isn't available in the provided documents."
            - Do not make assumptions or provide answers beyond what is explicitly stated in the documents.
            """
            try:
                response = model.generate_content(ai_prompt)
                answer = response.text
                
                # Check if the answer indicates no relevant information was found
                no_info_phrases = [
                    "isn't covered in the provided documents",
                    "isn't available in the provided documents",
                    "not mentioned in the documents",
                    "no information found",
                    "not covered",
                    "not available"
                ]
                
                show_source = not any(phrase.lower() in answer.lower() for phrase in no_info_phrases)
                
                response_data = {
                    "role": "assistant",
                    "content": answer,
                    "source": source if show_source else None
                }
            except Exception as e:
                response_data = {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                }
        else:
            response_data = {
                "role": "assistant",
                "content": "No matching information found in the provided documents."
            }

    st.session_state.messages.append(response_data)
    with st.chat_message("assistant"):
        st.markdown(response_data["content"])
        if response_data.get("source"):
            # Check if source is a PDF
            if isinstance(response_data["source"], str) and response_data["source"].endswith(".pdf"):
                pdf_path = os.path.join(SAVE_FOLDER, response_data["source"])
                if os.path.exists(pdf_path):
                    # Create a clickable link that opens in new tab
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode()}" '
                        f'download="{response_data["source"]}" target="_blank">📌 Source: {response_data["source"]}</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.caption(f"📌 Source: {response_data['source']} (file not found)")
            else:
                st.caption(f"📌 Source: {response_data['source']}")

    st.rerun()












# import streamlit as st
# import fitz  # PyMuPDF
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# from supabase import create_client, Client
# import uuid
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse
# import re
# import json
# from datetime import datetime
# import base64

# st.set_page_config(page_title="PDF & Web Q&A", page_icon="📄", layout="wide")

# load_dotenv()

# # Load API Keys
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# if not GENAI_API_KEY:
#     raise ValueError("API key for Gemini AI is missing. Please set GENAI_API_KEY in your .env file.")
# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise ValueError("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")

# # Initialize clients
# genai.configure(api_key=GENAI_API_KEY)
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Define Constants
# SAVE_FOLDER = "saved_pdfs"
# SCRAPED_DATA_FOLDER = "scraped_websites"
# os.makedirs(SAVE_FOLDER, exist_ok=True)
# os.makedirs(SCRAPED_DATA_FOLDER, exist_ok=True)
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize session state
# if "current_website" not in st.session_state:
#     st.session_state.current_website = None
# if "scraped_content" not in st.session_state:
#     st.session_state.scraped_content = None
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Utility Functions
# def clean_text(text):
#     """Cleans and normalizes text to remove unwanted spaces, special characters, and extra lines."""
#     text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces/newlines
#     text = re.sub(r'[^a-zA-Z0-9.,!?;:\'"-]', ' ', text)  # Keep essential punctuation
#     text = re.sub(r'\n+', '\n', text).strip()  # Remove excessive newlines
#     return text


# def save_scraped_data(url, content):
#     """Save cleaned scraped content to a local JSON file"""
#     domain = urlparse(url).netloc
#     filename = f"{domain}_{uuid.uuid4().hex[:8]}.json"
#     filepath = os.path.join(SCRAPED_DATA_FOLDER, filename)
    
#     data = {
#         "url": url,
#         "content": content,
#         "timestamp": str(datetime.now())
#     }
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
    
#     return filepath

# def extract_text_from_pdf(pdf_path, chunk_size=500):
#     doc = fitz.open(pdf_path)
#     text = "".join([page.get_text("text") for page in doc])
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# def store_in_supabase(chunk, embedding, source , content_type="pdf" ):
#     try:
#         data = {
#             "id": str(uuid.uuid4()),
#             "source": source,
#             "text": chunk,
#             "embedding": embedding,

#         }
#         response = supabase.table("pdf_documents").insert(data).execute()
#         if hasattr(response, 'error') and response.error:
#             st.error(f"Error storing document: {response.error}")
#     except Exception as e:
#         st.error(f"Error storing document in Supabase: {str(e)}")

# def semantic_search(query_embedding, top_k=10):
#     """Perform semantic search in Supabase"""
#     try:
#         # Convert the query embedding to a list
#         query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
#         # Execute the search
#         result = supabase.rpc('search_documents', {
#             'query_embedding': query_embedding,
#             'match_count': top_k
#         }).execute()
        
#         if hasattr(result, 'error') and result.error:
#             st.error(f"Search error: {result.error}")
#             return []
        
#         return result.data if result.data else []
        
#     except Exception as e:
#         st.error(f"Error searching in Supabase: {str(e)}")
#         return []

# def is_valid_url(url):
#     try:
#         result = urlparse(url)
#         return all([result.scheme, result.netloc])
#     except ValueError:
#         return False

# def scrape_website(url):
#     try:
#         if not is_valid_url(url):
#             return None, "Invalid URL format"
        
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
        
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.text, 'html.parser')

#         # Remove unwanted elements
#         for tag in soup(["script", "style", "noscript", "iframe", "footer", "header", "nav", "aside"]):
#             tag.decompose()

#         # Extract meaningful content
#         main_content = ""
#         content_tags = ["main", "article", "section", "div", "p",'h1','h2']
        
#         for tag in content_tags:
#             elements = soup.find_all(tag)
#             for element in elements:
#                 if len(element.text.strip()) > 50:  # Ignore very short elements
#                     main_content += element.get_text(separator="\n", strip=True) + "\n"
        
#         if not main_content:
#             main_content = soup.get_text(separator="\n", strip=True)  # Fallback to full text extraction

#         # Clean the extracted text
#         cleaned_content = clean_text(main_content)
        
#         # Update session state
#         st.session_state.current_website = url
#         st.session_state.scraped_content = cleaned_content
        
#         return cleaned_content, None

#     except requests.exceptions.RequestException as req_err:
#         return None, f"Request Error: {req_err}"
#     except Exception as e:
#         return None, f"Scraping Error: {str(e)}"


# # Streamlit UI
# st.markdown("<h1 style='text-align: center;'>📄 AskDoc with Context</h1>", unsafe_allow_html=True)

# # Sidebar
# st.sidebar.header("📜 Current Context")
# if st.session_state.current_website:
#     st.sidebar.markdown(f"**Website:**\n{st.session_state.current_website}")
# if st.session_state.uploaded_files:
#     st.sidebar.markdown("**Uploaded PDFs:**")
#     for file in st.session_state.uploaded_files:
#         st.sidebar.markdown(f"- {file.name}")
# st.sidebar.header("💬 Chat History")

# for index, message in enumerate(st.session_state.messages):
#     if message["role"] == "user":  # Show only user questions
#         with st.sidebar.expander(f"Q{index+1}: {message['content'][:30]}..."):  # Display first 30 chars as title
#             st.write(message["content"][:150] + "...")  # Show preview on click

# # File Upload Section
# st.header("📤 Upload Documents or Enter Website URL")
# tab1, tab2 = st.tabs(["PDF Upload", "Website Scraping"])

# with tab1:
#     uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

#     if uploaded_files and st.button("Process PDFs"):
#         progress_bar = st.progress(0)
#         total_files = len(uploaded_files)

#         for idx, uploaded_file in enumerate(uploaded_files):
#             save_path = os.path.join(SAVE_FOLDER, uploaded_file.name)

#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             chunks = extract_text_from_pdf(save_path)
            
#             for chunk in chunks:
#                 if chunk.strip():
#                     embedding = embed_model.encode(chunk).tolist()
#                     store_in_supabase(chunk, embedding, uploaded_file.name, "pdf")

#             progress_bar.progress((idx + 1) / total_files)

#         st.session_state.uploaded_files = uploaded_files
#         st.success(f"{len(uploaded_files)} PDFs processed successfully!")

# with tab2:
#     website_url = st.text_input("Enter website URL to scrape", key="website_url")
    
#     if st.button("Scrape Website", key="scrape_button"):
#         if website_url:
#             with st.spinner("Scraping and cleaning website content..."):
#                 scraped_content, error = scrape_website(website_url)
                
#                 if error:
#                     st.error(f"Error scraping website: {error}")
#                 else:
#                     # Save cleaned content to local file
#                     filepath = save_scraped_data(website_url, scraped_content)
                    
#                     # Process the scraped content (chunk and embed)
#                     chunks = [scraped_content[i:i+500] for i in range(0, len(scraped_content), 500)]
                    
#                     for chunk in chunks:
#                         if chunk.strip():
#                             embedding = embed_model.encode(chunk).tolist()
#                             store_in_supabase(chunk, embedding, f"Website: {website_url}", "website")
                    
#                     st.success(f"Successfully scraped {website_url}!")
#         else:
#             st.warning("Please enter a valid website URL")

# # Chat Interface
# st.markdown("## 💬 Ask a Question")

# # Display Previous Messages
# # Display Previous Messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message.get("source"):
#             # Check if source is a PDF
#             if isinstance(message["source"], str) and message["source"].endswith(".pdf"):
#                 pdf_path = os.path.join(SAVE_FOLDER, message["source"])
#                 if os.path.exists(pdf_path):
#                     # Create a clickable link that opens in new tab
#                     st.markdown(
#                         f'<a href="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode()}" '
#                         f'download="{message["source"]}" target="_blank">📌 Source: {message["source"]}</a>',
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.caption(f"📌 Source: {message['source']} (file not found)")
#             else:
#                 st.caption(f"📌 Source: {message['source']}")

# # ... (keep all your existing code until the response handling section)

# # Handle User Input
# if prompt := st.chat_input("Type your question here..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Check if any files have been uploaded or websites scraped
#     if not st.session_state.uploaded_files and not st.session_state.current_website:
#         response_data = {
#             "role": "assistant",
#             "content": "Please upload PDF files or process a website URL first before asking questions."
#         }
#     else:
#         # Generate AI Response only if documents exist
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         query_embedding = embed_model.encode(prompt)
        
#         # Search in Supabase
#         search_results = semantic_search(query_embedding)
        
#         if search_results:
#             context = "\n".join([r['text'] for r in search_results])
#             source = search_results[0]['source']
            
#             ai_prompt = f"""
#             **Instruction**: Analyze the given context based on the provided document or web content and carefully provide an response.
            
#             **Context**:
#             {context}
            
#             **Question**: {prompt}
            
#             **Rules**:

#             - Provide answers strictly based on the context.
#             - If the context provides a direct answer, explain it in detail and reference the relevant document section.
#             - If partial information is available, summarize it while mentioning that full details are unavailable.
#             - If no relevant information is found, clearly state: "This information isn't available in the provided documents."
#             - Do not make assumptions or provide answers beyond what is explicitly stated in the documents.
#             """
#             try:
#                 response = model.generate_content(ai_prompt)
#                 answer = response.text
                
#                 # Check if the answer indicates no relevant information was found
#                 no_info_phrases = [
#                     "isn't covered in the provided documents",
#                     "isn't available in the provided documents",
#                     "not mentioned in the documents",
#                     "no information found",
#                     "not covered",
#                     "not available"
#                 ]
                
#                 show_source = not any(phrase.lower() in answer.lower() for phrase in no_info_phrases)
                
#                 response_data = {
#                     "role": "assistant",
#                     "content": answer,
#                     "source": source if show_source else None
#                 }
#             except Exception as e:
#                 response_data = {
#                     "role": "assistant",
#                     "content": f"Error: {str(e)}"
#                 }
#         else:
#             response_data = {
#                 "role": "assistant",
#                 "content": "No matching information found in the provided documents."
#             }

#     st.session_state.messages.append(response_data)
#     with st.chat_message("assistant"):
#         st.markdown(response_data["content"])
#         if response_data.get("source"):
#             # Check if source is a PDF
#             if isinstance(response_data["source"], str) and response_data["source"].endswith(".pdf"):
#                 pdf_path = os.path.join(SAVE_FOLDER, response_data["source"])
#                 if os.path.exists(pdf_path):
#                     # Create a clickable link that opens in new tab
#                     st.markdown(
#                         f'<a href="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode()}" '
#                         f'download="{response_data["source"]}" target="_blank">📌 Source: {response_data["source"]}</a>',
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.caption(f"📌 Source: {response_data['source']} (file not found)")
#             else:
#                 st.caption(f"📌 Source: {response_data['source']}")

#     st.rerun()
