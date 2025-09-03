app_code = """
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests

import streamlit as st
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

INDEX_NAME = 'lex-fridman-podcast'
EMBED_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'mistral'

@st.cache_resource
def load_resources():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, embedder

index, embedder = load_resources()

def retrieve_context(query, top_k=3):
    q_emb = embedder.encode([query])[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return '\\n'.join([m['metadata']['text'] for m in results['matches']])

def query_ollama(model, prompt):
    url = 'http://localhost:11434/api/generate'
    payload = {'model': model, 'prompt': prompt, 'stream': False}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json().get('response', '')

def generate_answer(query):
    context = retrieve_context(query)
    prompt = f'''
    You are a helpful assistant answering questions about the Lex Fridman Podcast.
    Use ONLY the following context to answer the question.
    If the answer isn't in the context, say "I don't know."

    Context:
    {context}

    Question: {query}
    '''
    return query_ollama(LLM_MODEL, prompt)

st.set_page_config(page_title="Lex Fridman Podcast Chatbot", page_icon="🎙️")
st.title("🎙️ Lex Fridman Podcast Chatbot")

user_query = st.text_input("Enter your question:")
if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = generate_answer(user_query)
        st.success("Answer:")
        st.write(answer)
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created in your current folder!")

