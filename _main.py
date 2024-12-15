import os
import PyPDF2
import openai
import streamlit as st
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CORRECT_PASSWORD = os.getenv("PASSWORD")

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Streamlit app
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

# Automatically clear cache/session state on start
if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state['initialized'] = True

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything from the PDF document."}]
if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = None

# Main Page
st.title("PDF Chatbot")
st.caption("A chatbot powered by OpenAI to read and respond using a PDF document.")

with st.expander("Disclaimer", expanded=False, icon="🚨"):
    st.write('''**IMPORTANT NOTICE**: This is a prototype. The responses from this chatbot are generated by AI based on the PDF content. Always verify the information you receive.''')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to process user input
def process_user_input(prompt, context):
    # Construct the AI prompt
    ai_prompt = f"""
    You are a helpful assistant. Based on the information in the PDF document, answer the user's query in a concise and clear manner.

    Previous conversation context:
    {context}

    User's Query:
    {prompt}

    Answer:
    """

    # Query OpenAI for a response
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=ai_prompt,
            max_tokens=200,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't process your request at the moment."

# File uploader widget
uploaded_pdf = st.file_uploader("Upload PDF document", type="pdf")

# Process the uploaded PDF
if uploaded_pdf is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_pdf)

    # If the PDF is empty, display an error
    if not pdf_text:
        st.error("Failed to extract text from the PDF. Please try a different document.")
        st.stop()

    # Chat interface
    st.session_state.messages.append({"role": "assistant", "content": "PDF loaded. Ask your question!"})

    # Display the chat
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
            response_msg = process_user_input(prompt, "\n".join([msg['content'] for msg in st.session_state.messages[-5:]]))

        # Store the response
        st.session_state.messages.append({"role": "assistant", "content": response_msg})

        # Display the assistant's response
        st.chat_message("assistant").write(response_msg)
else:
    st.write("Please upload a PDF document to start.")