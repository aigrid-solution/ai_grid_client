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
    model="gpt-oss-120b",
    reasoning_effort="high",
    messages=[
        # {"role": "system", "content": "Reasoning: high"},
        {"role": "user", "content": "Solve 3x+5=20"}
    ],
)

print(response.choices[0].message.content)