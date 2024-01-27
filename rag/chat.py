from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from .load_to_chroma import Load_VD
import os


class chat:
    def __init__(self) -> None:
        pass

    def connect_openai(self):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        chat = chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
        return chat

    def qa_chain(self, query, retriver):
        primary_qa = self.connect_openai()
        qa_chain = RetrievalQA.from_chain_type(
            primary_qa, retriever=retriver, return_source_documents=True
        )
        return qa_chain({"query": query})


chat
