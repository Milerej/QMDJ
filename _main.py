# Chatbot script using OpenAI and PDF-based responses

import os
import pandas as pd
import openai
import boto3
import streamlit as st
from dotenv import load_dotenv
from io import StringIO
import pytz
import PyPDF2

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
if "query_processed" not in st.session_state:
    st.session_state.query_processed = False
if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = None

# Main Page
st.title("Qi Men Dun Jia (QMDJ) Chatbot")
st.caption("A Streamlit chatbot powered by JL")
with st.expander("Disclaimer", expanded=False, icon="🚨"):
    st.write('''**IMPORTANT NOTICE**: This web application is developed as a proof-of-concept prototype. The information provided here is **NOT intended for actual usage** and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

**Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.**

Always consult with qualified professionals for accurate and personalized advice.

''')
