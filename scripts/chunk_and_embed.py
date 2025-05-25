import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import chromadb

def load_pdfs(pdf_paths):
    print("Loading PDFs...")
    documents = []
    for path in pdf_paths:
        print(f"Loading {path} ...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def embed_and_store(chunks):
    print("Embedding and storing chunks...")

    # --- THESE LINES ARE CRUCIAL AND WERE MISSING/ABBREVIATED ---
    # Initialize Chroma client (reads config from env vars or creates persistent)
    client = chromadb.PersistentClient(path="./chroma_db_data") # Ensure this path is consistent
    # Create or get collection
    collection = client.get_or_create_collection(name="multimodal_rag_collection")
    # --- END CRUCIAL LINES ---

    # Extract texts for embedding
    texts = [chunk.page_content for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]

    # Initialize HuggingFaceInstructEmbeddings with an open-source model (no API keys)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    # Get embeddings for all texts
    embeddings_list = embeddings.embed_documents(texts)

    # Prepare metadatas from chunks
    # LangChain's Document objects have a .metadata attribute
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata.copy() # Make a copy to avoid modifying original
        # Add a 'text' field if not already present, pointing to the content itself
        meta['text'] = chunk.page_content # Redundant if you use 'documents=texts', but good for explicit storage
        # Ensure page number is easily accessible for retrieval
        if 'page' in meta:
            meta['page_number'] = meta['page']
        elif 'page_label' in meta: # Sometimes it's page_label
             meta['page_number'] = meta['page_label']
        # If no page_number, default to 0 or 'unknown'
        if 'page_number' not in meta:
            meta['page_number'] = 'unknown'

        # Clean metadata for ChromaDB (remove non-serializable objects or large ones if not needed)
        cleaned_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool, list, dict)):
                cleaned_meta[k] = v
            # You might want to handle other types or specific keys differently
            elif k == 'source' and isinstance(v, str): # LxangChain can put source path here
                cleaned_meta[k] = v


        metadatas.append(cleaned_meta)


    # Add to collection
    collection.add(
        documents=texts, # This holds the actual text content
        embeddings=embeddings_list,
        ids=ids,
        metadatas=metadatas # Crucial for filtering and showing context
    )

    print(f"Stored {len(texts)} chunks in ChromaDB.")

def main():
    # List your PDF files here
    pdf_paths = ["/Users/truptikirve/GenAI_Projects/multimodal-rag-app-v1/scripts/raw_data/attention_is_all_you_need.pdf"]

    # Step 1: Load PDFs
    documents = load_pdfs(pdf_paths)

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Embed chunks and store in ChromaDB
    embed_and_store(chunks)

if __name__ == "__main__":
    main()