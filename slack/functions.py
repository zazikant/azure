from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


import os
import openai
import pprint
import json
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
#hello
import requests
import csv

import matplotlib.pyplot as plt
import io

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
    # Define the API endpoint URL and parameters
    url = "http://13.232.224.37:8080/aurum/rest/v1/location/db/findall"
    params = {
        "project_id": 1,
        "user_id": 640,
        "token": "7efbfacb7556e57d0702",
        "page_size": 5
    }

    #

    llm = OpenAI()

    # # Make a GET request for each page and extract the desired fields
    locations = []
    for page_num in range(1, 4):
        params["page_num"] = page_num
        response = requests.get(url, params=params)
        data = response.json()
        for record in data["records"]:
            location = {
                "location_id": record["location_id"],
                "location_name": record["location_name"]
            }
            locations.append(location)


    # # Write the locations to a CSV file
    with open("/shashi/locations.csv", "w", newline="") as csvfile:
        fieldnames = ["location_id", "location_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for location in locations:
            writer.writerow(location)

    # repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    # llm = HuggingFaceHub(
    #     repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
    # )

    df = pd.read_csv("/shashi/locations.csv")

    sdf = SmartDataframe(df, config={"llm": llm})

    sdf.chat(user_input)

    response = sdf.last_code_generated.__str__()

    return response

# agent = create_pandas_dataframe_agent(llm, df, verbose=True)

# response = agent.run("only write the dataframe code logic for" + "what are top 5 location_names with highest occurence" + "strictly write the code logic")

# print(response)
# # Save the plot as a PNG file

# # Extract the code logic from the response
# code_logic = response['output']

# filtered_df = eval(response)

# print(filtered_df)

   



    # Apply the code logic to the entire dataframe
    # filtered_df = eval(code_logic)
    
    # Print the filtered dataframe
    # print(filtered_df)    

    
    # Assign the filtered dataframe to the response variable and return it
    # response = filtered_df
    # return response

    
# #     # Generate a bar plot
#     plt.bar(df["location_id"], df["location_name"])
#     plt.xlabel("Year")
#     plt.ylabel("Time")
#     plt.title("Winning Boston Marathon Times")

#     # Save the plot as a PNG file
#     plt.savefig("./shashi/plot.png")  
    
#     return response # Return the response and the file name

# #do bar plot of top 3 location_names
