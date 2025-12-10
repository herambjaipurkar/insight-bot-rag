import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# PAGE CONFIG
st.set_page_config(page_title="InsightBot with Gemini", layout="wide")

# HEADER
st.title("🤖 InsightBot: Chat with PDFs")
st.markdown("Powered by Google's **free Gemini API**.")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")

# FUNCTIONS
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_vector_store(text_chunks, api_key):
    try:
        # Using the modern embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.stop()

def get_response(user_question, api_key):
    # Load the database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    You are a helpful AI assistant. Answer the question based ONLY on the provided context below.
    
    Context:
    {context_text}
    
    Question: 
    {user_question}
    
    Answer:
    """
    
    # UPDATED: Changed 'gemini-pro' to 'gemini-1.5-flash'
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = model.invoke(prompt)
    return response.content

# MAIN APP
if api_key:
    pdf_docs = st.file_uploader("1. Upload PDF & Click Process", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process PDF"):
        if pdf_docs:
            with st.spinner("Analyzing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("Could not read text from PDF. It might be an image scan.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    chunks = text_splitter.split_text(raw_text)
                    
                    if chunks:
                        get_vector_store(chunks, api_key)
                        st.success("Done! PDF Processed.")
                    else:
                        st.error("PDF was empty.")
        else:
            st.warning("Please upload a file.")

    user_question = st.text_input("2. Ask a question about the PDF:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                answer = get_response(user_question, api_key)
                st.write("### Reply:")
                st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Enter your API Key in the sidebar to start.")
