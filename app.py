import streamlit as st
from chatbot import load_pdf, create_vectorstore, create_qa_chain

st.set_page_config(page_title="Insurance Chatbot 💬", page_icon="🤖")
st.title("AI-Powered Insurance Policy Chatbot 🤖")
st.write("Ask me anything about our insurance policies!")

# Load data and model
with st.spinner("Loading knowledge base..."):
    raw_text = load_pdf("data/insurance_policies.pdf")
    vectorstore = create_vectorstore(raw_text)
    qa_chain = create_qa_chain(vectorstore)

# Conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# User input
prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        result = qa_chain({"query": prompt})
        response = result["result"]
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
