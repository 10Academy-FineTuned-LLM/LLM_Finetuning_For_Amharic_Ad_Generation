from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os


data_file = "./data"
embedded_data = "./embedded"


def get_user_input_file(folder_path, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file)
    with open(file_path, "wb") as file:
        file.write(file.read())


def Load_VD(file):
    # Load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever
