"READING THE RAW DATA"

from langchain_community.document_loaders import TextLoaders
from hr_assistant import config



def load_document(file_path: str = config.DATA_FILE_PATH):
    """
    LOADS THE DOCUMENT DATA FROM THE FILE
    """
    loader = TextLoaders(file_path, encoding = "utf-8")
    return loader.load()
