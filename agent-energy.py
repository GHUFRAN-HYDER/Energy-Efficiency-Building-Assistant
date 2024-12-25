import streamlit as st
import os
from dotenv import load_dotenv
from swarmauri.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.messages.concrete.SystemMessage import SystemMessage
from swarmauri.messages.concrete.HumanMessage import HumanMessage
from swarmauri.llms.concrete.GroqModel import GroqModel

# Load environment variables
load_dotenv()

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    system_context = "You are an assistant that provides guidance to the user for energy efficiency in buildings/houses."
    st.session_state.conversation = MaxSystemContextConversation(
        system_context=SystemMessage(content=system_context),
        max_size=5
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_allowed_models(llm):
    failing_llms = [
        "llama3-70b-8192",
        "llama3.2-90b-text-preview",
        "mixtral-8x7b-32768",
        "llava-v1.5-7b-4096-preview",
        "llama-guard-3-8b",
    ]
    return [model for model in llm.allowed_models if model not in failing_llms]

# Initialize the LLM
@st.cache_resource
def initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please set the GROQ_API_KEY environment variable")
        st.stop()
    
    llm = GroqModel(api_key=api_key)
    allowed_models = get_allowed_models(llm)
    llm.name = "llama-3.1-70b-versatile"  # Using the model we selected in the notebook
    return llm

# Page config
st.set_page_config(
    page_title="Energy Efficiency Assistant",
    page_icon="🏠",
    layout="wide"
)

# Main UI
st.title("🏠 Energy Efficiency Building Assistant")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This assistant helps you with:
    - Building insulation
    - Energy-efficient lighting
    - HVAC systems
    - Renewable energy
    - Smart home technology
    - Energy-efficient appliances
    """)

# Initialize LLM
llm = initialize_llm()

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about energy efficiency..."):
    # Display user message
    st.chat_message("user").write(prompt)
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Process with LLM
    human_message = HumanMessage(content=prompt)
    st.session_state.conversation.add_message(human_message)
    
    # Get response
    with st.spinner("Thinking..."):
        llm.predict(conversation=st.session_state.conversation)
        response = st.session_state.conversation.get_last().content
    
    # Display assistant response
    st.chat_message("assistant").write(response)
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
