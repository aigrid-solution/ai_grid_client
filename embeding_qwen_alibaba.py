from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
client = OpenAI(
    base_url="http://app.ai-grid.io:4000/v1",
    api_key=AI_GRID_KEY,
)

response = client.embeddings.create(
    model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    input="Explain what embeddings are."
)

embedding = response.data[0].embedding
print("Embedding length:", len(embedding))