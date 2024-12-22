
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
# from IPython.display import Markdown, display
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import streamlit as st
import os


st.header("֎🇦🇮| Iam AI bot, ready to serve you..!🤖")
pdf_file = st.file_uploader('Choose your .pdf file', type="pdf")
# path of this script 
directory = "upload_files"
os.makedirs(directory,exist_ok=True)
# get fileName from user 
pdf_path = os.path.join(directory, pdf_file.name)
print(pdf_path)
# Creates a new file 

with open(pdf_path, mode='wb') as w:
        w.write(pdf_file.getvalue())
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question"}
    ]

documents = LlamaParse(api_key='llx-qM5LdxXKWLmzSg4YVfQU5Egmot3p0AHmPzdqlcY1LmmpZAiA',
                       result_type="markdown").load_data(pdf_path)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")



from llama_index.llms.groq import Groq

llm = Groq(
    model="llama-3.2-1b-preview",  # Change the model name
    api_key="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", # Add the API key here
)
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model

from llama_index.core import VectorStoreIndex
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True)
chat_engine = vector_index.as_chat_engine(chat_mode="condense_question", verbose=True)

def generate_response(text):
    prompt=text
    res = chat_engine.query(prompt)
    # display(Markdown(res.response).data)
    return res.response


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history