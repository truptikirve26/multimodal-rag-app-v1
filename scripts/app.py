import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


#where extracted pdf is saved
EXTRACTED_CONTENT_DIR = "./extracted_pdf_content"

#dir for chromadb
# CHROMA_DB_DIR = "./chroma_db"

CHROMA_DB_DIR = "./scripts/chroma_db"
# Embedding model name
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


# Function to load resources
@st.cache_resource
def load_embedding_model(model_name: str):
    """
    Loads the HuggingFaceBgeEmbeddings model.
    Uses st.cache_resource to ensure it's loaded only once
    """
    print(f"Loading Embedding Model: {model_name}...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}

    )
    print("Embedding Model Loaded.")
    return embeddings

@st.cache_resource
def load_chroma_db(db_dir: str, _embeddings_model):
    """
    Loads the persistent ChromaDB vector store.
    Uses st.cache_resource to ensure it's loaded only once
    """
    print(f"Loading ChromaDB from {db_dir}...")

    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=_embeddings_model
    )
    print("ChromaDB loaded.")
    return vector_store


def main():
    st.set_page_config(page_title="Multimodal RAG Demo", layout="wide")
    st.title("Multimodal RAG with PDF Content")

    # Load resources
    embeddings_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    vector_store = load_chroma_db(CHROMA_DB_DIR, embeddings_model)

    st.write("---")

    st.header("🔍 Retrieval Test (Temporary)")
    st.write("This test will perform a generic search to verify ChromaDB is accessible and populated.")
    if st.button("Perform Generic Retrieval Test"):
        try:
            # Use a very generic query that should match something in any document
            test_query = "the"
            st.info(f"Searching for documents containing: '{test_query}'")
            test_results = vector_store.similarity_search_with_score(test_query, k=3)

            if test_results:
                st.success(f"Successfully retrieved {len(test_results)} test documents:")
                for i, (doc, score) in enumerate(test_results):
                    st.write(f"**Test Result {i + 1}** (Score: {score:.4f})")
                    st.write(
                        f"  **Source:** {doc.metadata.get('source', 'N/A')}, **Page:** {doc.metadata.get('page', 'N/A')}")
                    st.write(f"  **Content (first 150 chars):** {doc.page_content[:150]}...")
            else:
                st.warning("Generic test query returned no documents. ChromaDB might be empty or not properly loaded.")
        except Exception as e:
            st.error(f"Error during generic test retrieval: {e}")
    st.write("---")


    # User Query Input
    st.header("Ask a question about the PDF content?")
    query = st.text_input("Enter your query here:", placeholder="e.g., What is the self-attention mechanism?", key="user_query")

    if st.button("Search"):
        if query:
            st.info("Searching for relevant information...")
            try:
                # Perform similarity search
                results = vector_store.similarity_search_with_score(query, k=5)

                if results:
                    st.success(f"Found {len(results)} relevant chunks:")
                    for i, (doc,score) in enumerate(results):
                        st.subheader(f"Result {i+1} (Relevance Score: {score:.4f})")
                        st.write(f"**Source:** {doc.metadata.get('source','N/A')}")
                        st.write(f"**Page:** {doc.metadata.get('page','N/A')}")
                        st.write(f"**Type:** {doc.metadata.get('type', 'N/A')}")

                        if doc.metadata.get('type') == 'table':
                            st.markdown("---")
                            st.markdown("### Table Content (HTML)")
                            st.components.v1.html(doc.page_content, height=300, scrolling=True)
                            st.markdown("---")
                            st.markdown("### Table Content (Plain Text Representation)")
                            st.expander("Click to see plain text version of table").write(doc.page_content)
                        else:
                            st.markdown("### Content Preview")
                            st.expander("Click to see full content").write(doc.page_content)
                        st.markdown("---")
                else:
                    st.warning("No relevant information found for your query.")

            except Exception as e:
                st.error(f"An error occurred during search: {e}")
                st.info("please ensure your ChromaDB is correctly populated and accessible.")

        else:

            st.warning("Please enter a query to search.")

    st.write("---")
    st.markdown("### Note: This demo currently only performs retrieval. An LLM integration for answer generation will be added in the next steps. ")

if __name__=="__main__":
    main()





