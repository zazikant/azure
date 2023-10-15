import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, abort
from functions import draft_email
import logging
from functools import wraps
import time
import sys
import json
import requests


# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
# LANGCHAIN_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
# LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]
# LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
# OPENAI_API_MODEL = os.environ["OPENAI_API_MODEL"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

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

@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    say("Sure, I'll get right on that!")
    # response = my_function(text)
    
    # Extract the email from the text
    email, response = draft_email(text)
    
    # Make the POST request
    url = "https://hook.us1.make.com/ohyonocw701n4ynie637qcm3roe3yrhn"
    headers = {"Content-Type": "application/json"}
    payload = {"email": email, "response": response}
    data = json.dumps(payload)
    # data = {"response": response}
    
    # post_response = requests.post(url, headers=headers, json=data)
    
    post_response = requests.post(url, headers=headers, data=data)

    # Check the response status code
    if post_response.status_code == 200:
        say("POST request successful")
    else:
        say("POST request failed")
    

# Demo
@flask_app.route("/slack/events", methods=["POST"])
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