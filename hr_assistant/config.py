import os
from dotenv import load_dotenv


load_dotenv()

#SECRET VARIABLES
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


#VECTOR STORE
"""
3 TYPES - 
    1. INMEMORY
    2. PERSISTENT MEMORY (CURRENT ONE)
    3. CLOUD MEMORY
"""
VECTOR_STORE_PATH = os.path.join("data", "faiss_index")


#DATA PATH
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")


#MODELS
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_MODEL_NAME = "openai/gpt-oss-20b"


#TEXT SPLITTING CONFIGS
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


#RETERIVAL RESULTS
TOP_K_RESULTS = 3


#SYSTEM INSTRUCTIONS
SYSTEM_PROMPT = (
    """ You are a friendly HR assistant working for Acme Crop. 
        Always use the search_hr_policy tool to look up 
        facts before answering. 
        If the answer isn't in the search results, say you don't know 
        instead of guessing."""
)



