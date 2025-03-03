# Libraries
import os
from typing import List, Any

import glob
import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Environment variables
os.environ["LANGSMITH_PROJECT"] = "qa-assistant-with-rag"
os.environ["LANGSMITH_TRACING"] = "true"

load_dotenv(".env")

# Constants
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"

CONTEXTUALIZE_QUESTION_SYSTEM_PROMPT = "Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."


# Functions
def add_metadata(doc: Any, doc_type: str) -> Any:
    """
    Add doc_type metadata to a lagchain document.

    Args:
        doc (Any): A langchain document which we want to add metadata to
        doc_type (str): The type of document which we want to add

    Returns:
        Any: The langchain document with updated metadata
    """
    doc.metadata["doc_type"] = doc_type
    
    return doc


def split_documents(documents: List[Any]) -> List[Any]:
    """
    Split documents in the knowledge base into chunks.
    
    Args:
        documents (List[Any]): A list of langchain documents to be split.

    Returns:
        List[Any]: A list of langchain document chunks resulting from the split operation.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total number of chunks: {len(chunks)}")
    
    return chunks

def generate_embeddings(chunks: List[Any]) -> Chroma:
    """
    Generate embeddings and store them in a Chroma vector store.
    
    Args:
        chunks (List[Document]): A list of langchain document chunks to generate embeddings for.

    Returns:
        Chroma: The Chroma vector store containing the generated embeddings.
    """
    embeddings = OpenAIEmbeddings()

    # Delete if already exists
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=DB_NAME)
    
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    
    return vectorstore

def contextualize_question(llm: ChatOpenAI, retriever: Any) -> Any:
    """
    Creates a langchain history-aware retriever that formulates standalone questions from a chat history and user input.

    Args:
        llm (ChatOpenAI): The language model used for generating prompts.
        retriever (Any): The retriever used to gather context from the chat history.

    Returns:
        Any: A retriever that takes into account the chat history to create a standalone question.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_QUESTION_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(llm,
                                                             retriever,
                                                             contextualize_q_prompt)
    
    return history_aware_retriever

def answer_question(llm: ChatOpenAI, history_aware_retriever: Any) -> Any:
    """
    Creates a langchain conversational question-answering chain that uses retrieved context and chat history.

    Args:
        llm (ChatOpenAI): The language model used for generating responses.
        history_aware_retriever (Any): The retriever that provides context based on chat history.

    Returns:
        Any: A chain capable of answering questions by leveraging chat history and retrieved context.
    """
    qa_system_prompt = "You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    \n\n{context}"
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

def chat(question: str, history: list) -> str:
    """
    Generates a response to a user's question by streaming answers from a conversational RAG chain.

    Args:
        question (str): The user's question to be processed.
        history (list): A list of previous chat messages to provide context.

    Yields:
        str: A progressively constructed response from the RAG chain.
    """

    stream = conversational_rag_chain.stream({"input": question})
    response = ""
    for chunk in stream:
        response += chunk.get("answer") or ""
        yield response


# Main function
if __name__ == "__main__":
    folders = glob.glob("../knowledge-base/*") # folders containing RAG documents

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder,
                                 glob="**/*.md",
                                 loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"})
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
        
    print(f"Document types found: {set(doc.metadata["doc_type"] for doc in documents)}")
        
    chunks = split_documents(documents)
    vectorstore = generate_embeddings(chunks)
    # retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    history_aware_retriever = contextualize_question(llm, retriever)
    conversational_rag_chain = answer_question(llm, history_aware_retriever)
    
    gr.ChatInterface(chat, type="messages").launch(share=True)