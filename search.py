from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from loader import load_docs

llm = Ollama(model="tinyllama")
embeddings = OllamaEmbeddings()
output_parser = StrOutputParser()

docs = load_docs()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever(
    # search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 3}
)
# docs = retriever.get_relevant_documents("What is the price of sharpening a 7 inch western style knife?")
# from pprint import pprint
# pprint(docs)
# exit(0)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "What is the price of sharpening a 7 inch western style knife?"})
print(response["answer"])
