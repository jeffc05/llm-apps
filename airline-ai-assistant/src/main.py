# Libraries
import os
import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.adapters.openai import convert_openai_messages


# Environment variables
os.environ["LANGSMITH_PROJECT"] = "airline-ai-assistant"
os.environ["LANGSMITH_TRACING"] = "true"

load_dotenv(".env")


# Constants
TICKET_PRICES = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

SYSTEM_MESSAGE = \
"""You are a helpful assistant for an Airline called FlightAI. \
Give short, courteous answers, no more than 1 sentence. \
Always be accurate. If you don't know the answer, say so."""

MODEL_PROVIDER = "openai"
MODEL = "gpt-4o-mini"


# Functions
@tool
def get_ticket_price(destination_city: str) -> str:
    """
    Get the price of a return ticket to the destination city.
    Possible destination cities are: London, Paris, Tokyo, Berlin.
    Call this whenever you need to know the ticket price, for example
    when a customer asks 'How much is a ticket to this city?'.
    
    Parameters
    ----------
    destination_city: str
        The name of the city to which the ticket price is requested.
        
    Returns
    -------
    str
        The ticket price for the specified city, or "Unknown" if the city is not listed.
    """
    return TICKET_PRICES.get(destination_city.lower(), "Unknown")

@tool
def book_flight_to(destination_city: str) -> str:
    """
    Book a flight to the destination city.
    Possible destination cities are: London, Paris, Tokyo, Berlin.
    Call this whenever you need to book a flight, for example when
    a customer asks 'Can you book a flight to this city for me?'.
    
    Parameters
    ----------
    destination_city: str
        The name of the city to which the flight is to be booked.

    Returns
    -------
    str
        Confirmation message if the flight is booked, or an error
        message if the city is not available.
    """
    if destination_city.lower() in TICKET_PRICES.keys():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("../bookings.txt", "a") as f:
            f.write(f"[{now}] Flight booked to {destination_city}\n")
        return f"Flight to {destination_city} has been booked"
    return "We don't fly to that city. Please choose from London, Paris, Tokyo, or Berlin."

def chat(message: str, history: list) -> str:
    """
    Chat with the model, using the tools to get ticket prices and book flights.
    
    Parameters
    ----------
    message : str
        The message to send to the model.
    history : list
        The list of messages to send to the model as context.
    
    Returns
    -------
    str
        The response from the model.
    """
    model = init_chat_model(MODEL, model_provider=MODEL_PROVIDER)
    tools = [get_ticket_price, book_flight_to]
    model_with_tools = model.bind_tools(tools)
    
    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + convert_openai_messages(history) + [HumanMessage(content=message)]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_ticket_price": get_ticket_price, "book_flight_to": book_flight_to}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    response = model_with_tools.invoke(messages)
    return response.content


# Main function
if __name__ == "__main__":
    print("Starting chat interface...")

    gr.ChatInterface(fn=chat, type="messages").launch(share=True)