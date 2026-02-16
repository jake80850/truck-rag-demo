import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama


from pathlib import Path

def load_css():
    css_path = Path(__file__).parent / "style.css"
    # debug: shows up in sidebar so you can confirm it found the file
    st.sidebar.caption(f"CSS: {css_path} | exists={css_path.exists()}")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.sidebar.error("style.css not found (check file location)")

load_css()
from pathlib import Path
import streamlit as st

svg = (Path(__file__).parent / "assets/telo_logo.svg").read_text(encoding="utf-8")

st.markdown(
    f'<div class="telo-logo-container">{svg}</div>',
    unsafe_allow_html=True
)



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

st.title("Telo Truck Yudi AI Demo")
st.caption("Meet the MT1 — ask anything about diagnostics, specs, or support notes.")


index = build_index()
chat_engine = index.as_chat_engine()


# --- Chat UI (normal) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# input
prompt = st.chat_input("Ask about the truck...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # get answer
    resp = chat_engine.chat(prompt)

    # IMPORTANT: show only the text
    answer_text = getattr(resp, "response", str(resp))

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    with st.chat_message("assistant"):
        st.markdown(answer_text)


