import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from apikey import apikey

# Ensure the API key is set in the environment variables
os.environ['OPENAI_API_KEY'] = apikey
def clear_history():
    st.session_state.history = []

def check_file_extension(file_name):
    return file_name.split('.')[-1].lower()

def loader_type(file_name):
    extension = check_file_extension(file_name)
    if extension == 'txt':
        return TextLoader
    else:
        return PyPDFLoader

st.title("Chat with a Document")
upload_file = st.file_uploader("Upload a document", type=['txt', 'pdf'])
add_file = st.button("Add Document", on_click=clear_history)

if upload_file and add_file:
    try:
        file_path = os.path.join('../', upload_file.name)
        with open(file_path, 'wb') as f:
            f.write(upload_file.getvalue())

        loader_class = loader_type(upload_file.name)
        loader = loader_class(file_path)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(document)

        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)

        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        retriever = vector_store.as_retriever()
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)

        st.success("Document added successfully!")
        st.session_state['crc'] = crc

    except Exception as e:
        st.error(f"Failed to process document: {e}")

question = st.text_input("Ask a question about the document")

if question:
    if 'crc' in st.session_state:
        crc = st.session_state['crc']
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        response = crc.invoke({'question': question, 'chat_history': st.session_state.history})
        st.session_state.history.append((question, response['answer']))
        st.write(response['answer'])
