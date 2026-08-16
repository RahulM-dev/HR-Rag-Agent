from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from hr_assistant.llm import get_llm
from hr_assistant.splitter import split_to_chunks
from hr_assistant.tools import create_search_tool
from hr_assistant.vector_store import(
    build_vector_store,
    get_retriever,
    load_vector_stores,
    save_vector_store,
    vector_store_verification
)


def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, reusing the Qdrant Cloud collection if we have one."""

    if vector_store_verification():
        print("Found an existing Qdrant Cloud collection, connecting to it (fast, no re-embedding).")
        return load_vector_stores()

    print("No saved vector store found, building one from scratch...")    
    documents = load_document(file_path)
    chunks = split_to_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk for next time.")
    return vector_store


def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""

    if not config.GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")

    if not config.JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")   

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])

    return agent     


def ask(agent, question: str) -> str :
    """Ask the agent a question and return its final answer as plain text."""

    response = agent.invoke(
        {
            "messages":[
                {
                    "role" : "user",
                    "content" : question
                }
            ]
        }
    )

    return response["messages"][-1].content