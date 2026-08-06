"WRAP THE RETRIEVER AS A TOOL THE AGENT CAN CALL"


from langchain.tools import tool



def create_search_tool(retriever):
    """
    Return a @tool function that searched the HR policy document
    """

    @tool
    def search_hr_policy(question : str):
        """
        Search the HR policy documentfor information about leave, work from home, probation, notice period
        reimbursement, code of conduct, holidays or exit process
        """
        matching_chunks = retriever.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)

    return search_hr_policy