## My Research Assistant (RAG System)
[![Streamlit App](https://sagara92-sas-research-assistant-app-q1kiqg.streamlit.app/)](https://sagara92-sas-research-assistant-app-q1kiqg.streamlit.app/)

Welcome to an end-to-end Retrieval-Augmented Generation (RAG) web app built to interactively answer questions related to my peer-reviewed publications. This project is an AI research assistant, which retrieves relevant responses from dense academic papers, researching Supermassive Black Holes, Active Galactic Nuclei (AGNs), Blazars, and QPO analysis of blazars.

This `Research Assistant` is trained on the published papers from my [Google Scholar](https://scholar.google.com/citations?user=L1xZ36AAAAAJ&hl=en) profile.

### Data Ingestion:
- To ensure cost-efficiency and modularity, the PDF files of the publications are parsed using LangChain's document loaders.
- The dense scientific text, equations, and tables are processed using a `RecursiveCharacterTextSplitter` with a chunk size of 1000 and an overlap of 200 tokens to preserve contextual meaning.
- Text chunks are transformed into high-dimensional vectors using HuggingFace's open-source all-MiniLM-L6-v2 model.
- Vectors are stored using a separate FAISS (Facebook AI Similarity Search) for each paper. 

### App Functionality
- The frontend is a lightweight, interactive chat interface built with Streamlit and hosted on Streamlit Community Cloud.
- The application scans the data directory, loads the isolated FAISS index for each publication, and merges them into a single master index in memory.
- When a query is submitted, the system embeds the question using the same HuggingFace model and retrieves the top 5 most relevant text chunks from the merged FAISS index.
- The retrieved context and the query are passed to Google's Gemini 2.5 Flash model and the LLM synthesizes an accurate response.
- The app parses the metadata of the retrieved chunks and appends the specific filename of the source publications to the final output, ensuring transparency.

### Tech Stack:
- Python, LangChain (langchain-classic, langchain-huggingface), sentence-transformers (all-MiniLM-L6-v2), FAISS, Google Gemini 2.5 Flash, Streamlit, Streamlit Community Cloud