import os
import sys
import logging
import streamlit as st
from src.constant import *
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from src.configuration.pinecone_operations import PineconOperation
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
OPENAI_API_VERSION = OPENAI_API_VERSION

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# initiatlizing logger
logger = logging.getLogger(__name__)


class QuestionAnswering:
    def __init__(self, query: str) -> None:
        self.utils = MainUtils()
        self.pinecone = PineconOperation(pinecone_api_key=PINECONE_API_KEY, pinecone_environment=PINECONE_ENVIRONMENT_NAME)
        self.query = query

    def initiate_question_answering(self):
        try:
            
            data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
            logger.info(f"Got data directory - {data_directory}")

            print("Loading Docs....")

            documents = self.utils.load_docs(data_directory=data_directory)
            logger.info("Loaded the documents")

            print("Docs Loaded...!!")

            splitted_docs = self.utils.split_docs(data=documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            logger.info("Documents splitted in the chunks")

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            self.pinecone.create_index(index_name=INDEX_NAME, vector_dimension=VECTOR_DIMENSION, metric=METRIC, pods=PODS)
            logger.info("Pinecone index created")

           #docsearch = Pinecone.from_texts(texts=[d.page_content for d in splitted_docs], embedding=embeddings, index_name=INDEX_NAME)
            docsearch = Pinecone.from_documents(splitted_docs, embedding=embeddings, index_name=INDEX_NAME)
            logger.info("Generated embeddings for the given data")

            similar_docs = self.pinecone.get_similar_docs(docsearch=docsearch, query=self.query, k= NO_OF_CHUNKED_DOC)
            logger.info(f"Found {NO_OF_CHUNKED_DOC} similar docs as per the query")

            llm = OpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
            logger.info(f"Loaded LLM model from OpenAi class. Model name - {LLM_MODEL}")

            chain = load_qa_chain(llm, chain_type="stuff")
            logger.info("Loaded qa chain from langchain")

            answer = chain.run(input_documents=similar_docs, question=self.query)
            logger.info(f"Query --> {self.query}. \
                        Answer ---> {answer}")

            return answer

        except Exception as e:
            raise CustomException(e, sys) from e
        

if __name__ == "__main__":
    text_input = st.text_input('Ask your query:')
    if st.button("Submit"):
        question_answering = QuestionAnswering(query=text_input)

        answer = question_answering.initiate_question_answering()

        st.success(answer)