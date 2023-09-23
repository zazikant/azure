import os, io
import logging
import time
import sys
import io
import pprint
import json
import requests
import csv
import matplotlib.pyplot as plt

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, abort
from functions import draft_email
from functools import wraps

from openai import OpenAI
from pandas import pd
from pandasai import PandasAI


# Load environment variables from .env file
# load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
OPENAI_API_MODEL = os.environ["OPENAI_API_MODEL"]
SERPER_API_KEY = os.environ["SERPER_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
LANGCHAIN_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


def require_slack_verification(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not verify_slack_request():
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


def verify_slack_request():
    # Get the request headers
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # Check if the timestamp is within five minutes of the current time
    current_timestamp = int(time.time())
    if abs(current_timestamp - int(timestamp)) > 60 * 5:
        return False

    # Verify the request signature
    return signature_verifier.is_valid(
        body=request.get_data().decode("utf-8"),
        timestamp=timestamp,
        signature=signature,
    )


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")

#hello

def my_function(text):
    """
    Custom function to process the text and return a response.
    In this example, the function converts the input text to uppercase.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    response = text.upper()
    return response


# Initialize a WebClient instance
client = WebClient(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_mentions(body, say):
    
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    say("Sure, I'll get right on that!")
    # response = my_function(text)
    # response = draft_email(text)

    # Upload the PNG file as an attachment
    try:
        response = client.files_upload(
            channels=body["event"]["channel"],
            file="/exports/charts/temp_chart.png",
            title="Here is the chart you requested.",
        )
        print(response)
    except SlackApiError as e:
        print(f"Error uploading file: {e.response['error']}")


  
# Demo
@flask_app.route("/slack/events", methods=["POST"])
@require_slack_verification
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """

    return handler.handle(request)

# Run the Flask app
if __name__ == "__main__":
    logging.info("Flask app started")
    flask_app.run(host="0.0.0.0", port=8000)
    
    
#when you want to work with pandas, do that. pandas agent doesnt allow you to save files.

# @app.event("app_mention")
# def handle_mentions(body, say):
    
#         # Create a Pandas DataFrame
#     df = pd.DataFrame({
#         "Year": [2022, 2021, 2020, 2019, 2018],
#         "Name": ["Evans Chebet", "Benson Kipruto", "", "Lawrence Cherono", "Yuki Kawauchi"],
#         "Country": ["KEN", "KEN", "", "KEN", "JPN"],
#         "Time": ["2:06:51", "2:09:51", "", "2:07:57", "2:15:58"]
#     })

#     # Generate a bar plot
#     plt.bar(df["Year"], df["Time"])
#     plt.xlabel("Year")
#     plt.ylabel("Time")
#     plt.title("Winning Boston Marathon Times")

#     # Save the plot as a PNG file
#     plt.savefig("bar_plot.png")
    
    
#     # Upload the PNG file as an attachment
#     try:
#         response = client.files_upload(
#             channels=body["event"]["channel"],
#             file="bar_plot.png",
#             title="Winning Boston Marathon Times",
#             initial_comment="Here is the bar plot of the winning Boston Marathon times."
#         )
#         print(response)
#     except SlackApiError as e:
#         print(f"Error uploading file: {e.response['error']}")


    