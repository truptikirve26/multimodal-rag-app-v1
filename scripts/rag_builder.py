import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

#where extracted pdf is saved
EXTRACTED_CONTENT_DIR = "./extracted_pdf_content"

#dir for chromadb
CHROMA_DB_DIR = "./chroma_db"

# Embedding model name
EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"

def load_extracted_content(content_dir: str):
    """
    Loads text and table content from the extracted_pdf_content directory.
    """
    documents = []
    figures_dir = os.path.join(content_dir, "figures")

    for filename in os.listdir(content_dir):
        file_path = os.path.join(content_dir, filename)

        if filename.endswith(".txt"):
            # handle text files
            with open(file_path, "r",encoding="utf-8") as f:
                content = f.read()

            # extract metadata from filename (e.g., page_number, index)
            # example filename: text_page_1_idx_1.txt
            parts = filename.replace(".txt","").split('_')
            metadata = {
                "source": filename,
                "type": "text",
                "page": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else "N/A",
                "index": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else "N/A"
            }

            documents.append(Document(page_content=content, metadata=metadata))

        elif filename.endswith(".html"):
            #handle html table files
            with open(file_path,"r",encoding="utf-8") as f:
                html_content = f.read()

            # extract metadata from filename
            parts = filename.replace(".html","").split("_")
            metadata = {
                "source": filename,
                "type": tuple,
                "page": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else "N/A",
                "index": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else "N/A",
                "html_content": html_content # store html in metadata for retrieval

            }

            documents.append(Document(page_content=html_content, metadata=metadata))

        #images are already saved by unstructured's partition pdf with metadata.image_path

    print(f"Loaded {len(documents)} text and table documents from {content_dir}")

    return documents

def create_and_persist_chroma_db(documents: list[Document], db_dir: str, embedding_model_name: str):
    """
    Creates or loads a ChromaDB vector store and adds documents to it
    """
    # initialize the embedding model
    print(f"Initialing the embedding model: {embedding_model_name}....")
    #InstructEmbeddings will download the model the first time it's used
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},  # Specify 'cpu' if no GPU, or 'cuda' for GPU
        encode_kwargs={'normalize_embeddings': True}  # Recommended for BGE
    )
    print("Embedding model initialized.")

    # Create or load the ChromaDB vector store
    #if the dir exists, it loads or else it creates.
    print(f"Creating/Loading ChromaDB at {db_dir}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir
    )

    vectorstore.persist()
    print(f"ChromaDB created/updated with {len(documents)} documents.")
    print(f"ChromaDB collection count: {vectorstore._collection.count()}")

    return vectorstore

if __name__=="__main__":

    print("Starting RAG Builder")

    # Load extracted text and table content
    all_documents = load_extracted_content(EXTRACTED_CONTENT_DIR)

    # filter out complex metadata that ChromaDB doesn't support
    print(f"Filtering complex metadata from {len(all_documents)} documents...")
    filtered_documents = filter_complex_metadata(all_documents)

    print("Metadata filtering complete")

    # create and persist ChromaDB
    chroma_instance = create_and_persist_chroma_db(
        documents=all_documents,
        db_dir=CHROMA_DB_DIR,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

print("\n RAG Builder finished. ChromaDB is ready for use.")


# Test Retrieval

print("\n Testing Retrieval...")
query = "What is the self-attention mechanism ?"
print(f"Query: {query}")
results = chroma_instance.similarity_search_with_score(query, k=3)
for i, (doc,score) in enumerate(results):
    print(f"\nResult {i + 1} (Score: {score:.4f}):")
    print(f"  Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
    print(f"  Type: {doc.metadata.get('type')}")
    print(f"  Content (first 200 chars):\n{doc.page_content[:200]}...")