import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables for GOOGLE_API_KEY
load_dotenv()

# Create the HTML string with the text and the clickable logo
html_string = """
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px;">Ask me questions about my peer-reviewed publications!</span>
    <a href="https://scholar.google.com/citations?user=L1xZ36AAAAAJ&hl=en" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Google_Scholar_logo.svg" alt="Google Scholar" width="30"/>
    </a>
</div>
"""


st.set_page_config(page_title="Physics Research Assistant", page_icon="🔭")
st.title("SA's Research Assistant")
st.markdown(html_string, unsafe_allow_html=True)

# Load and Merge Vector Databases
@st.cache_resource
def load_vector_store():
    base_index_dir = "./data/faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    paper_folders = [f.path for f in os.scandir(base_index_dir) if f.is_dir()]
    
    if not paper_folders:
        st.error("No FAISS indexes found. Please run the ingestion notebook first.")
        return None

    # Load the first paper's index to act as the master database
    master_db = FAISS.load_local(paper_folders[0], embeddings, allow_dangerous_deserialization=True)
    
    # Merge the rest of the vector index into the master database
    for folder in paper_folders[1:]:
        local_db = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
        master_db.merge_from(local_db)
        
    return master_db

vector_store = load_vector_store()

# Setup the LLM and Retrieval Chain
if vector_store:
    api_key = os.getenv("GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        
    if not api_key:
        st.error("API Key not found! Please set GOOGLE_API_KEY in .streamlit/secrets.toml")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, 
        google_api_key=api_key
    )
    
    # Define a System Prompt
    system_prompt = (
        "You are an intelligent research assistant for a Data Scientist with a Physics PhD."
        "Use the provided context from their published papers to answer the user's questions. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # =====================================
    # Streamlit Chat UI
    # =====================================
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt_text := st.chat_input("Ask about Sagar's research methodology, findings, or data..."):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching publications..."):
                response = rag_chain.invoke({"input": prompt_text})
                answer = response["answer"]
                
                # Extract sources to show where the answer came from
                sources = set([doc.metadata.get('source', 'Unknown') for doc in response["context"]])
                source_text = "\n\n**Sources:**\n" + "\n".join([f"- {os.path.basename(s)}" for s in sources])
                
                full_response = answer + source_text
                st.markdown(full_response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})