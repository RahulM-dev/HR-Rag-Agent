"READING THE RAW DATA"

from langchain_community.document_loaders import TextLoader
from hr_assistant import config


# WILL LOAD THE FILE FROM THE GIVEN PATH
def load_document(file_path: str = config.DATA_FILE_PATH):
    """
    LOADS THE DOCUMENT DATA FROM THE FILE
    """
    loader = TextLoader(file_path, encoding = "utf-8")
    return loader.load()
