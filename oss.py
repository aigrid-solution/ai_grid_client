from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
client = OpenAI(
    api_key=AI_GRID_KEY,
    base_url="http://app.ai-grid.io:4000/v1"
)


response = client.chat.completions.create(
    model="voxtral-mini-2602",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)