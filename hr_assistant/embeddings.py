"GENERATE EMBEDDINGS USING THE EMBEDDING MODDEL"

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config



def get_embedding_model():

    """
    Returns a JinaEmbedding model object
    """

    return JinaEmbeddings(
        model_name = config.EMBEDDING_MODEL_NAME
    ) 