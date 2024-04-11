import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Convert string of URLs to list
def method_get_website_text(urls):
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

#split the text into chunks
def method_get_text_chunks(text):
    #text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(text)
    return doc_splits

#convert text chunks into embeddings and store in vector database
def method_get_vectorstore(document_chunks):
    embeddings = HuggingFaceEmbeddings()
    #embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

    
def get_context_retriever_chain(vector_store,question):
    # Initialize the retriever
    retriever = vector_store.as_retriever()
    
    # Define the RAG template
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    
    # Create the RAG prompt template
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    
    # Initialize the Hugging Face language model (LLM)
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":0.6, "max_length":1024})
    
    # Construct the RAG pipeline
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return after_rag_chain.invoke(question)

def main():
    st.set_page_config(page_title="Chat with websites", page_icon="🤖")
    st.title("Chat with websites")
    
    # sidebar
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")
    
    if website_url is None or website_url == "":
        st.info("Please enter a website URL")
    
    else:
        # Input fields
        question = st.text_input("Question")
        
        # Button to process input and get output
        if st.button('Query Documents'):
            with st.spinner('Processing...'):
                # get pdf text
                raw_text = method_get_website_text(website_url)
                # get the text chunks
                doc_splits = method_get_text_chunks(raw_text)
                # create vector store
                vector_store = method_get_vectorstore(doc_splits)
                #Generate response using the RAG pipeline
                answer = get_context_retriever_chain(vector_store,question)
                # Display the generated answer
                split_string = "Question: " + str(question)
                result = answer.split(split_string)[-1]
                st.text_area("Answer", value=result, height=300, disabled=True)

if __name__ == '__main__':
    main()