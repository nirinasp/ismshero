import os

import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

st.set_page_config(page_title="ISMS Document Import")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be authenticated to access this page")
    st.stop()

st.title("ISMS Document Import")
st.write("Register ISMS PDF Documents")

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(supabase_url, supabase_key)

def save_file(file):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

def get_documents_from_pdf_files(uploaded_files):
    documents = []

    for file in uploaded_files:
        file_path = save_file(file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    return split_docs


def create_vector_store(splits):
    vector_store = SupabaseVectorStore.from_documents(
        splits,
        OpenAIEmbeddings(model='text-embedding-3-large'),
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )

    return vector_store

def register_documents(uploaded_files):
    documents = get_documents_from_pdf_files(uploaded_files)
    create_vector_store(documents)


uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        try:
            register_documents(uploaded_files)
            st.success("Documents imported successfully!")
        except Exception as e:
            st.error(f"An error occured when importing the documents: {e}")
            st.stop()
    
