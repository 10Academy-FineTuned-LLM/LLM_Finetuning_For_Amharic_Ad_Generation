import argparse
from dataclasses import dataclass
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
CHROMA_PATH = './chromadb/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="the query text")
    args = parser.parse_args()
    query_text = args.query_text

    # embdding function
    embdding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embdding_function)
    pass