import os
import pandas as pd
import openai
import boto3
import streamlit as st
from dotenv import load_dotenv
from io import StringIO
import pytz
from collections import Counter
from streamlit_option_menu import option_menu

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("REGION_NAME")
AWS_BUCKET = os.getenv("BUCKET_NAME")
CORRECT_PASSWORD = os.getenv("PASSWORD")

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize the S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize Streamlit app
st.set_page_config(page_title="QMDJ Chatbot", page_icon="🤖")

# Automatically clear cache/session state on start
if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state['initialized'] = True

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there! Please enter your query or click on any of the Frequently Asked Questions to continue."}]
if "query_counter" not in st.session_state:
    st.session_state.query_counter = Counter()

# Main Page
st.title("Qi Men Dun Jia (QMDJ) Chatbot")
st.caption("A Streamlit chatbot powered by JL")

# Function to read data from S3
def read_data_from_s3(bucket_name, file_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_content = response['Body'].read()

        try:
            decoded_content = csv_content.decode('cp1252')
        except UnicodeDecodeError:
            decoded_content = csv_content.decode('ISO-8859-1')

        return pd.read_csv(StringIO(decoded_content))
    except Exception as e:
        st.error(f"Error reading data from S3: {e}")
        return None

# Function to check password
def check_password():
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    user_password = st.text_input("Password", type="password", key="password", on_change=check_password_submit)
    return st.session_state.password_correct

def check_password_submit():
    if st.session_state.password.strip() == CORRECT_PASSWORD.strip():
        st.session_state.password_correct = True
        st.success("Password correct!")
    else:
        st.error("😕 Password incorrect")

# Main application flow
if not check_password():
    st.stop()

# Load data from S3
file_key = 'Qi Men Dun Jia Compendium 2nd Edition.pdf'
data = read_data_from_s3(AWS_BUCKET, file_key)

# Check if data was successfully loaded
if data is None or data.empty:
    st.error("Failed to load data from S3 or the data is empty. Please check your bucket name and file key.")
    st.stop()

# Set the timezone to Singapore Time (SGT)
sgt_timezone = pytz.timezone('Asia/Singapore')

# Function to process user input
def process_user_input(prompt):
    try:
        # Create a prompt for the AI model
        ai_prompt = f"""
        You are a professional AI assistant. 
        Your task is to provide clear, concise, and accurate responses based on relevant replies extracted from a database, 
        to provide a relevant answer based on the user's query, taking into account the ongoing conversation context. 
        Please ensure your tone is friendly and supportive.
        
        Previous conversation context:
        {get_context()}
        
        User's Query:
        {prompt}
        """

        # Make a request to OpenAI's Chat API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use the correct model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.3
        )

        # Extract the assistant's message and return it
        msg = response['choices'][0]['message']['content'].strip()
        return msg
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't process your request."

# Function to get the conversation context
def get_context():
    # Get the last 5 messages as context
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
    return context

# Display chat messages for the chatbot page
if 'messages' not in st.session_state:
    st.session_state.messages = []

# User input box and chat display
if st.session_state.messages:  # Only display chat if there are messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# User input box
if prompt := st.chat_input("Type your message here..."):
    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Process user input
    with st.spinner("Processing your request..."):
        response_msg = process_user_input(prompt)
    
    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response_msg})

    # Display the assistant's response
    st.chat_message("assistant").write(response_msg)
