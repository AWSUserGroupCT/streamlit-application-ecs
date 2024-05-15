import json
import requests
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings.bedrock import BedrockEmbedding
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import config

OPENSEARCH_URL = 'https://hwbe0yj7qodsyrwdc23f.us-east-1.aoss.amazonaws.com'
INDEX_NAME = 'embeddings_index'
VECTOR_DIMENSION = 128


def create_index(index_name, client):
    index_settings = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION
                }
            }
        }
    }
    response = client.indices.create(index=index_name, body=index_settings)
    return response


def bulk_upload_embeddings(url, index_name, embeddings):
    headers = {'Content-Type': 'application/x-ndjson'}
    data = ""
    for id, embedding in embeddings.items():
        action = {"index": {"_index": index_name, "_id": id}}
        data += json.dumps(action) + "\n"
        # Check if the embedding is already a list and handle accordingly
        if isinstance(embedding, list):
            data += json.dumps({"embedding": embedding}) + "\n"
        else:
            # In case the embedding might not be a list, adapt this part as necessary
            data += json.dumps({"embedding": embedding.tolist()}) + "\n"
    response = requests.post(f"{url}/_bulk", headers=headers, data=data, verify=False)
    return response.json()


# get region from AWS_REGION envionment variable if not set default to us-east-1
region = os.getenv('AWS_REGION', 'us-east-1')
service = 'aoss'  # still es?
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
print(awsauth.session_token)
openSearch_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

# Initialize embeddings processor
embeddings_processor = BedrockEmbedding(region_name="us-east-1")

# Create the index in OpenSearch
# create_i = create_index(INDEX_NAME, openSearch_client)
# print(create_i)

# Load PDF documents and process text
embeddings_to_upload = {}
path_to_dir = "./resumes/"
pdf_files = glob.glob(os.path.join(path_to_dir, "*.pdf"))
for _file in pdf_files:
    loader = PyPDFLoader(_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        text_content = chunk.page_content  # Assuming chunk has a 'text' attribute
        vector = embeddings_processor.get_text_embedding(text_content)
        doc_id = os.path.splitext(os.path.basename(_file))[0] + f"_chunk_{i}"
        embeddings_to_upload[doc_id] = vector
        # add vectors to opensearch



# Upload embeddings
bulk_upload = bulk_upload_embeddings(OPENSEARCH_URL, INDEX_NAME, embeddings_to_upload)
print(bulk_upload)
