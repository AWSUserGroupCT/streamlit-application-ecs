import os
import streamlit as st
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import boto3
# Ensure the API key is set in the environment variables


def get_ssm_parameter(name):
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(Name=name, WithDecryption=True)
    return response['Parameter']['Value']
def clear_history():
    st.session_state.history = []

def check_file_extension(file_name):
    return file_name.split('.')[-1].lower()

def loader_type(file_name):
    extension = check_file_extension(file_name)
    if extension == 'txt':
        return TextLoader
    elif extension == 'pdf':
        return PyPDFLoader
    elif extension == 'pptx':
        return UnstructuredPowerPointLoader


st.title("Chat with a Document")
model_selection = st.selectbox(
   "Which model would you like to use?",
   ("Llama 2 Chat 70B", "Llama 2 Chat 13B", "Jurassic-2 Ultra", "Jurassic-2 Mid", "GPT-3.5 Turbo", "gpt-4o"),
   index=None,
   placeholder="Select model method...",
   on_change=clear_history()
)

models_dict = {
    "Llama 2 Chat 13B": "meta.llama2-13b-chat-v1",
    "Llama 2 Chat 70B": "meta.llama2-70b-chat-v1",
    "Jurassic-2 Ultra": "ai21.j2-ultra-v1",
    "Jurassic-2 Mid": "ai21.j2-mid-v1",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "gpt-4o": "gpt-4o",
}
upload_file = st.file_uploader("Upload a document", type=['txt', 'pdf', 'pptx'])
add_file = st.button("Add Document", on_click=clear_history)

if upload_file and add_file:
    try:
        file_path = os.path.join('./', upload_file.name)
        with open(file_path, 'wb') as f:
            f.write(upload_file.getvalue())

        loader_class = loader_type(upload_file.name)
        loader = loader_class(file_path)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(document)
        # create_index('https://gvgrg4ppdrcrbhc5fggk.us-east-1.aoss.amazonaws.com', 'embeddings')
        # bulk_upload_embeddings('https://gvgrg4ppdrcrbhc5fggk.us-east-1.aoss.amazonaws.com', 'embeddings', embeddings)
        # if model select contains gpt then use OpenAIEmbeddings else use BedrockEmbeddings
        llm = None
        embeddings = None
        if ("gpt" or "GPT") in model_selection:
            os.environ['OPENAI_API_KEY'] = get_ssm_parameter('/streamlit/openaikey')
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name=models_dict[model_selection], temperature=0)
        else:
            embeddings = BedrockEmbeddings(region_name="us-east-1")
            llm = Bedrock(model_id=models_dict[model_selection])

        vector_store = Chroma.from_documents(chunks, embeddings)

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
