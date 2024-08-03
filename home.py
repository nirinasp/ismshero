import os
import json

import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain, LLMChain, SimpleSequentialChain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

st.set_page_config(
    page_title="ISMS Hero",
    page_icon="👋",
)

st.write("# Welcome to ISMS Hero! 👋")

st.markdown(
    """
    I am a security assistant of Spoon Consulting Ltd.
    My goal is to respond to questions about the company ISMS awre stuff and eventually provide the references of the provided information
    """
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.warning("You must be logged in order to ask for requests")
    st.stop()

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(supabase_url, supabase_key)

def get_vector_store():
    return SupabaseVectorStore(
        embedding=OpenAIEmbeddings(model='text-embedding-ada-002'),
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

def create_chain(vector_store):
    model = ChatOpenAI(temperature=0.4)

    system_prompt_str = """
    As a security manager for Spoon Consulting Ltd., a consulting company in digital transformation,
    your primary role is to locate and reference documents within the company's Information Security Management System.
    Your task involves identifying relevant documents and providing concise explanations about how specific sections of these documents relate to the query at hand.
    You should focus on delivering precise, relevant information while ensuring that your responses remain within the scope of the company's information security guidelines.
    If you are unable to locate the requested information, tell the user that you do not know anything about the asked question.

    Documents:

    {context}
    """
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_str),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    documents_create_chain = create_stuff_documents_chain(
        llm=model,
        prompt=system_prompt
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )


    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        documents_create_chain
    )


    reference_prompt_str = """
    You are an assistant that helps retrieve documents metadata based on provided answers.
    Return the documents metadata as a JSON Array (begin with the square bracket directly) where each item object has the following keys: source: [document metadata source], page: [document metadata page], content: [document content]
    If the provided answers state that it didn't find any relevant info, then return an empty JSON Array (begin with the square bracket directly)
    """
    reference_prompt = ChatPromptTemplate.from_messages([
        ("system", reference_prompt_str),
        ("human", "Documents:\n\n{context}\n\nAnswer: {answer}")
    ])

    reference_chain = LLMChain(
        llm=model,
        prompt=reference_prompt
    )

    return retrieval_chain | reference_chain


def render_chat_history():
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

vector_store = get_vector_store()

chain = create_chain(vector_store)

if "chat_history" not in st.session_state.keys():
    st.session_state.chat_history = [{"role": "assistant", "content": "How can I help you ?"}]

    render_chat_history()

if user_input := st.chat_input("Write your ISMS aware questions here"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    render_chat_history()

    if st.session_state.chat_history[-1]['role'] != 'assistant':
        with st.chat_message('assistant'):
            with st.spinner("Please wait ..."):
                response = chain.invoke({
                    "input": user_input,
                    'chat_history': st.session_state.chat_history
                })

                st.markdown(response['answer'])

                try:
              
                    document_source_text = response['text'][response['text'].find("["):response['text'].rfind("]") + 1]
                
                    documents_source = json.loads(document_source_text)

                    for idx, doc in enumerate(documents_source,1):
                        filename = os.path.basename(doc['source'])
                        page_num = doc['page']
                        ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                        with st.popover(ref_title):
                            st.caption(doc['content'])
                except:
                    pass

                st.session_state.chat_history.append(
                    {
                    "role": "assistant",
                    "content": response['answer']
                    }
                )
