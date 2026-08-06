"STORE THE VECTORS IN THE VECTOR DB"


import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embedding_model



#BUILD A VECTOR STORE
def build_vector_store(chunks) :
    """
    Embed the chunk and build a seachable
    FAISS index in memory
    """
    embedding_model = get_embedding_model()
    return FAISS.from_documents(chunks, embedding_model)


#SAVE VECTORS
def save_vector_store(
        vector_store: FAISS, 
        store_path : str = config.VECTOR_STORE_PATH 
        ):

    """
    Save the generated FAISS index in the path
    """
    vector_store.save_local(store_path)


#LOAD VECTORS
def load_vector_stores(path : str = config.VECTOR_STORE_PATH):
    """
    Load a stored FAISS index from disk
    """
    embedding_model = get_embedding_model()
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization = True
    )


#VERIFICATION OF VECTOR STORE
def vector_store_verification(path : str = config.VECTOR_STORE_PATH):
    """
    Check if a saved FAISS index already exists on disk
    """
    return os.path.exists(os.path.join(path, "index.faiss"))


#GET RETRIEVER
def get_retriever(vector_store: FAISS, k : int = config.TOP_K_RESULTS):
    """
    Turns a vector into a retriever that returns the top-k matching chunks.
    """
    return vector_store.as_retriever(
        search_kwargs = {
            "k" : k
        }
    )