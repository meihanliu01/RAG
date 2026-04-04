import os
import sys
import json
import faiss
import numpy as np
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Safety settings for Mac M4 / Apple Silicon
os.environ["OBJC_DISABLE_INITIALIZE_FOR_SAFETY"] = "YES"

print(">>> 1. Script started, importing libraries...", flush=True)

def main():
    print(">>> 2. Entering main function...", flush=True)
    
    print(">>> 3. Loading SQuAD dataset (validation split)...", flush=True)
    dataset = load_dataset("squad", split="validation")
    
    # Get unique contexts to avoid redundant embedding of the same passage
    unique_contexts = list(set(dataset['context']))
    print(f">>> Successfully extracted {len(unique_contexts)} unique context passages.", flush=True)
    print(">>> 4. Splitting documents into chunks...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(unique_contexts)
    chunk_texts = [doc.page_content for doc in chunks]
    print(f">>> Total text chunks created: {len(chunk_texts)}", flush=True)

    print(">>> 5. Preparing to load Embedding Model (BGE)...", flush=True)
    print("Note: If this is the first run, it may take 1-2 minutes to download.", flush=True)
    
    # Check for Apple Silicon (MPS) support, otherwise fallback to CPU
    device = "mps" if os.path.exists("/System/Library/Frameworks/Metal.framework") else "cpu"
    print(f">>> Using device: {device.upper()}")
    
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    print(">>> 6. Model loaded successfully! Computing embeddings...", flush=True)
    
    # Generate embeddings for all chunks
    embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)

    print(">>> 7. Building FAISS index...", flush=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save the index and the text chunks for later retrieval
    faiss.write_index(index, "squad_faiss.index")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_texts, f, ensure_ascii=False)

    print("\n✅ Task Complete! Generated 'squad_faiss.index' and 'chunks.json'.")

if __name__ == "__main__":
    main()