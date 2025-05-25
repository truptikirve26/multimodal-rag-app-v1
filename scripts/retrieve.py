import os
import chromadb
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# Paths
# The ChromaDB persistent directory, consistent with your chunk_and_embed.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# chroma_db_path = os.path.join(project_root, "chroma_db_data")
chroma_db_path ="/Users/truptikirve/GenAI_Projects/multimodal-rag-app-v1/scripts/chroma_db_data"

# Settings
# Use the same embedding model you used for storing
EMBEDDING_MODEL = "hkunlp/instructor-base"
TOP_K = 5

def initialize_chromadb_retriever():
    print(f"Initializing ChromaDB client at: {chroma_db_path}...")
    client = chromadb.PersistentClient(path=chroma_db_path)

    print(f"Getting ChromaDB collection: multimodal_rag_collection...")
    collection = client.get_or_create_collection(name="multimodal_rag_collection")

    # --- ADD THESE LINES TO CHECK COLLECTION STATUS ---
    try:
        count = collection.count()
        print(f"ChromaDB collection '{collection.name}' has {count} items.")
        if count == 0:
            print("WARNING: Collection is empty. Please run 'chunk_and_embed.py' again to populate it.")
        else:
            # You can also peek at some items to see their structure
            peek_results = collection.peek(limit=2) # Get first 2 items
            print("\n--- Sample ChromaDB Items (first 2) ---")
            if peek_results['documents']:
                print(f"Documents: {peek_results['documents']}")
            if peek_results['metadatas']:
                print(f"Metadatas: {peek_results['metadatas']}")
            print("---------------------------------------")

    except Exception as e:
        print(f"Error checking collection count: {e}")
    # --- END ADDED LINES ---

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL)

    return collection, embedder

def retrieve(query: str, collection, embedder, top_k: int = TOP_K):
    """
    Embeds the query and retrieves relevant text chunks from ChromaDB.

    Args:
        query: The user's query string.
        collection: The ChromaDB collection object.
        embedder: The HuggingFaceInstructEmbeddings instance.
        top_k: The number of top results to retrieve.

    Returns:
        A list of dictionaries, each containing retrieved text, its metadata (page), and score.
    """
    print("Encoding query...")
    # Embed the query using the same embedder
    query_embedding = embedder.embed_query(query) # Use embed_query for a single string

    print(f"Query embedding length: {len(query_embedding)}")
    print("Performing search in ChromaDB...")

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding], # Chroma expects a list of query embeddings
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'] # Request document content, metadata, and distances
    )

    retrieved_items = []
    # results['documents'] will be a list of lists (one inner list per query).
    # Since we have only one query, we take results['documents'][0].
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            retrieved_items.append({
                "score": float(distance),  # ChromaDB returns distance, lower is better (closer)
                "page": metadata.get("page", "N/A"), # Safely get page number
                "text": doc_content # This is the stored document content (text summary)
            })
    return retrieved_items

def main():
    # Initialize ChromaDB collection and embedding model
    collection, embedder = initialize_chromadb_retriever()

    # Sample query loop
    while True:
        print("\nWaiting for user query input...")
        query = input("Enter your query (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break

        print(f"\nTop {TOP_K} results for '{query}':\n")
        results = retrieve(query, collection, embedder)

        if results:
            for i, result in enumerate(results, 1):
                print(f"[{i}] Type: Text, Page {result['page']} (Distance: {result['score']:.4f})")
                print(result['text'])
                print("-" * 80)
        else:
            print("No results found.")

if __name__ == "__main__":
    main()