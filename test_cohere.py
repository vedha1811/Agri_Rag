# test_cohere.py

import os
from dotenv import load_dotenv
import cohere

load_dotenv()

print("KEY:", os.getenv("CO_API_KEY"))

client = cohere.ClientV2(
    api_key=os.getenv("CO_API_KEY")
)

print("SUCCESS")