import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="InsightBot with Gemini", layout="wide")

# Header
st.title("🤖 InsightBot: Chat with PDFs using Google Gemini")
st.markdown("Upload a PDF document and ask questions based on its content. Powered by Google's **free Gemini API**.")

# Sidebar for API Key and Instructions
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("AIzaSyAoM7icEMfbDNwSn0K-Ryvjf1FvA48769M", type="password")
    st.markdown("""
    **How to get a free API Key:**
    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    2. Create a key.
    3. Paste it above.
    """)
    st.divider()
    st.markdown("Built with [Streamlit](https://streamlit.io) and [LangChain](https://python.langchain.com/).")

# Helper Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    # Using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    # Define the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main App Logic
if api_key:
    # File Uploader
    pdf_docs = st.file_uploader("Upload your PDF Files and Click 'Process'", accept_multiple_files=True, type=['pdf'])

    if st.button("Process PDFs"):
        with st.spinner("Processing... This may take a moment."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, api_key)
            st.success("Done! You can now ask questions.")

    # Chat Interface
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, api_key)
else:
    st.warning("👈 Please enter your Google Gemini API Key in the sidebar to start.")
