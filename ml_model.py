import os
import openai
import pandas as pd
import json

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the dataset (ensure the path is correct)
dataset = pd.read_csv('dataset/search_keywords.csv')

def extract_data_from_keyword(keyword):
    # Load system content and user content
    system_content = load_prompt('prompts/search_keywords_system_prompt.txt')
    user_content = f"Keywords: {keyword}"

    # Make a request to the OpenAI API to generate a chat completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        max_tokens=150,
        top_p=0.8
    )

    # Extract the completion result and token usage information from the response
    completion = response.choices[0].message['content']
    result = json.loads(completion)
    prompt_tokens_used = response['usage']['prompt_tokens']
    completion_tokens_used = response['usage']['completion_tokens']

    return result, prompt_tokens_used, completion_tokens_used

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()
