import os
import openai
import pprint
import json
import requests
import csv
import io
import matplotlib.pyplot as plt
from IPython.core.display import HTML
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader

load_dotenv(find_dotenv())

load_dotenv()


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
OPENAI_API_MODEL = os.environ["OPENAI_API_MODEL"]

from dotenv import find_dotenv, load_dotenv
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

import pandas as pd


from pandasai import SmartDataframe
from pandasai.llm import OpenAI
 

def draft_email(user_input):
    
    from langchain.document_loaders import DirectoryLoader, CSVLoader

    loader = DirectoryLoader(
        "./shashi", glob="**/*.csv", loader_cls=CSVLoader, show_progress=True
    )
    docs = loader.load()    
    
    
    #textsplitter-----------------

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=2,
    )

    docs = text_splitter.split_documents(docs)
    # print(docs[3].page_content)
    #-----------------    
    
    from langchain.embeddings import OpenAIEmbeddings
    openai_embeddings = OpenAIEmbeddings()

    # from langchain.embeddings import HuggingFaceEmbeddings
    # openai_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    #loading vectors into vector db-----------------

    from langchain.vectorstores.faiss import FAISS
    import pickle

    #Very important - db below is used for similarity search and not been used by agents in tools

    db = FAISS.from_documents(docs, openai_embeddings)


    import pickle

    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)
        
    with open("db.pkl", "rb") as f:
        db = pickle.load(f)
        
    query = user_input
    docs = db.similarity_search(query, k=8)

    # HTML(f'<div style="width:50%">{docs[0].page_content}</div>')
    # print(docs[0].page_content)


    import os
    import openai
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain, SequentialChain
    from dotenv import load_dotenv, find_dotenv
    from langchain.prompts import ChatPromptTemplate
    from langchain import HuggingFaceHub
    from langchain import PromptTemplate, LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.output_parsers import ResponseSchema
    from langchain.output_parsers import StructuredOutputParser
    from langchain.memory import ConversationSummaryBufferMemory


    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


    # template = """
    # you are a pediatric dentist and you are writing a key features serial wise for following information: 

    # text: {context}
    # """

    #blog post
    template = """
    You are a SEO expert having expertise in creating landing pages. The landing page should should effectively capture the key points, insights, and information from the Context.

    Focus on maintaining a coherent flow and using proper grammar and language.

    Incorporate relevant headings, subheadings, and bullet points to organize the content.

    Ensure that the tone of the blog post is engaging and informative, catering to {target_audience} audience.

    Feel free to enhance the transcript by adding additional context, examples, and explanations where necessary.

    The goal is to convert context into a polished and valuable written resource while maintaining accuracy and coherence.

    text: {context}

    """
    target_audience = """Age group of 35-45 years old people, who are looking for BIM services"""

    prompt  = PromptTemplate(
        input_variables=["context", "target_audience"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
    response = chain.run({"context": docs, "target_audience": target_audience})

    return response
#G