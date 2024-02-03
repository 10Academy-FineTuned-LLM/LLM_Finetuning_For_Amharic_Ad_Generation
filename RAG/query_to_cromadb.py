import argparse
from dataclasses import dataclass
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import sys



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

CHROMA_PATH = './chromadb/'

client = OpenAI(
    api_key=OPENAI_API_KEY
)

core_embeddings_model = None
def get_context():
    # instantiate a retriever
    vectorstore = Chroma(persist_directory="./cachce",embedding_function=core_embeddings_model)
    
    retriever = vectorstore.as_retriever()
    #search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.85}
    return retriever

def generate_add(user_input, context):
    template = f'''
    Generate an advertisement given the following context.    
    You must use the following context:
    {context}
    '''   
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": template},{"role": "user", "content": user_input}],
        n=3,
    )

    return response
