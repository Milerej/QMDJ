import openai

openai.api_key = "your_openai_api_key"  # Replace with your actual OpenAI API key

def process_user_input(prompt, context):
    try:
        ai_prompt = f"""
        You are a professional AI assistant. 
        Your task is to provide clear, concise, and accurate responses based on relevant replies extracted from a database, 
        to provide a relevant answer based on the user's query, taking into account the ongoing conversation context. 
        Please ensure your tone is friendly and supportive.
        
        Previous conversation context:
        {context}
        
        User's Query:
        {prompt}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use the correct model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message['content'].strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."
