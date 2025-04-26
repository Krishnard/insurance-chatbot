import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
import os

# Streamlit UI setup
st.set_page_config(page_title="Insurance Chatbot", layout="wide")
st.title("🛡️ AI-Powered Insurance Policy Chatbot")

# API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # Load documents
    with st.spinner("Indexing insurance policy document..."):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)

    st.subheader("Ask your insurance-related questions below 👇")

    # Initialize session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Your question:", placeholder="e.g., What does life insurance cover?")

    if user_input:
        response = query_engine.query(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", str(response)))

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
else:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
