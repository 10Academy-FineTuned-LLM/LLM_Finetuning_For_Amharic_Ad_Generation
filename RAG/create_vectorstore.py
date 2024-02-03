from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from transformers import AutoTokenizer, AutoModel
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
import sys
import torch



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Load the tokenizer and model
model_name = 'Davlan/bert-base-multilingual-cased-finetuned-amharic'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



data_path = './data/'
CHROMA_PATH = './chromadb'

def load_documents(data_path):    
    try:
        loader = DirectoryLoader(data_path)
        documents = loader.load()
        print("data loaded sucessfully")
    except:
        print("document not found!")
        return None
    return documents

def split_text(documents:list[Document]):
    try:
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            add_start_index = True
        )
        chunk = text_spliter.split_documents(documents)
        print("data splited successfuly!")
        return chunk
    except:
        print("document not found")


def embed_doc (document):
    


    # Tokenize the text
    encoded_input = tokenizer(document, padding=True, truncation=True, return_tensors='pt')
    
    # Generate the embeddings
    with torch.no_grad():
        outputs1 = model(**encoded_input)

    

    # Extract the embeddings
    embeddings1 = outputs1.last_hidden_state.squeeze(dim=0)
    

    # Print the embeddings
    print(embeddings1)
    return embeddings1

def embed_tochroma (embeddings1):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    try:
        db = Chroma.from_documents(embeddings1,\
                                persist_directory=CHROMA_PATH)
        db.persist()
        print("Vectorstore created successfully!")
    except:
        print("Couldn't create the vectore database")

def save_chunks_to_chroma(chunks:list[Document]):
    #Clear the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(),\
                                persist_directory=CHROMA_PATH)
        db.persist()
        print("Vectorstore created successfully!")
    except:
        print("Couldn't create the vectore database")

def generate_data_store():
    documents = load_documents(data_path)
    chunks = split_text(documents) 
    embeddings = embed_doc(chunks)
    embed_tochroma(embeddings) 
    
    #save_chunks_to_chroma(chunks) 
    print("Vector store created successfuly!") 

def main():
    generate_data_store()      


if __name__ == "__main__":
    main()    
