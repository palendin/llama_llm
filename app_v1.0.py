import streamlit as st
import os # Still import os because LangChain's Replicate might explicitly look for it in os.environ
from langchain_community.llms import Replicate
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate # Ensure this is imported for PromptTemplate

# Set Streamlit page configuration
st.set_page_config(page_title="🦙💬 Simple Llama 2 Chatbot", layout="wide")

# --- Streamlit UI ---
st.title("🦙💬 Simple Llama 2 Chatbot")
st.caption("Powered by Replicate and Streamlit")

# --- Replicate API Key Setup ---
# Access the API token directly from st.secrets
# Streamlit will automatically load this from .streamlit/secrets.toml locally,
# or from the Streamlit Cloud secrets configuration when deployed.
try:
    replicate_api_token = st.secrets["REPLICATE_API_TOKEN"]
    # LangChain's Replicate integration often expects the token in os.environ
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
except KeyError:
    st.error("REPLICATE_API_TOKEN not found in Streamlit secrets. "
             "Please add it to your `.streamlit/secrets.toml` file (for local dev) "
             "or via Streamlit Cloud's secrets management (for deployment).")
    st.stop()


# --- Model Selection and Parameters ---
with st.sidebar:
    st.header("Llama 2 Model Parameters")
    selected_model = st.selectbox('Choose a Llama2 model:', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    temperature = st.slider('Temperature:', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.slider('Top_p:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length:', min_value=32, max_value=2048, value=512, step=8)

# Map selected model to Replicate's model ID
if selected_model == 'Llama2-7B':
    llama_model_id = "meta/llama-2-7b-chat:8e6975e5ed6174911a65d623270289f93de89292d6e93af00dafe37060d29769"
elif selected_model == 'Llama2-13B':
    llama_model_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89229e6ce6af079dc23fee269283101847436dd65ee0a"
else:
    st.error("Invalid model selected.")
    st.stop()


# Initialize Replicate LLM
llm = Replicate(
    model=llama_model_id,
    model_kwargs={"temperature": temperature, "top_p": top_p, "max_new_tokens": max_length},
)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- User Input and LLM Response ---
if prompt := st.chat_input("Ask me anything about Llama 2..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = ""
            # The .stream() method provides token-by-token output
            for s in llm.stream(prompt):
                full_response += s
                st.write(full_response) # Update continuously as tokens arrive
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
