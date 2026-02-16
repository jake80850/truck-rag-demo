import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Use local embeddings (no API key needed)
Settings.embed_model = FastEmbedEmbedding()
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)


QDRANT_URL = "http://localhost:6333"
COLLECTION = "truck_docs"
DATA_DIR = "data"

@st.cache_resource
def build_index():
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)

    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)
    return index

st.title("🚚 Telo Truck Yudi AI Demo")

index = build_index()
chat_engine = index.as_chat_engine()

if prompt := st.chat_input("Ask about the truck..."):
    st.write("You:", prompt)
    response = chat_engine.chat(prompt)
    st.write("AI:", response)
