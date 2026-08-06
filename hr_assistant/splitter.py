"SPLIT OUR DATA IN CHUNKS"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config



def split_to_chunks(documents: str):
    """
    SPLITS THE DOCUMENT DATA TO CHUNKS
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP
    )

    return text_splitter.split_documents(documents)