import os
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_openai import ChatOpenAI

from loader import load_docs

docs = load_docs()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt
