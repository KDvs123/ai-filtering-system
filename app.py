from flask import Flask, request, jsonify
import pandas as pd
import json
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/process_keyword": {"origins": "*"}})  # Enable CORS for specific endpoint

dataset = pd.read_csv('dataset/search_keywords.csv')

def extract_data_from_keyword(keyword):
    system_content = load_prompt('prompts/search_keywords_system_prompt.txt')
    user_content = f"Keywords: {keyword}"

    client = OpenAI()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=4096,
        top_p=0.8,
        response_format="json"
    )

    completion = chat_completion.choices[0].message.content
    result = json.loads(completion)
    prompt_tokens_used = chat_completion.usage.prompt_tokens
    completion_tokens_used = chat_completion.usage.completion_tokens

    return result, prompt_tokens_used, completion_tokens_used

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

@app.route('/process_keyword', methods=['POST'])
def process_keyword():
    data = request.get_json()
    keyword = data.get('keyword')
    
    result, prompt_tokens_used, completion_tokens_used = extract_data_from_keyword(keyword)
    
    return jsonify({
        'result': result,
        'prompt_tokens_used': prompt_tokens_used,
        'completion_tokens_used': completion_tokens_used
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
