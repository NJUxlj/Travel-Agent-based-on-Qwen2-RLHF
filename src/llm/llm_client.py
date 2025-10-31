from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os, sys

from dotenv import load_dotenv

load_dotenv()

# Initialize the asynchronous OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url=os.getenv("ZHIPU_ENDPOINT"),
)


# Create the OpenAI Chat Completion Service
LLM = OpenAIChatCompletion(
    ai_model_id=os.getenv("ZHIPU_MODEL_ID"),
    async_client=client,
)
