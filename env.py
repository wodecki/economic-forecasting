import json
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator, ValidationError
from typing import List
import os

from dotenv import load_dotenv

# Load from specific path with override
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API key: {openai_api_key}")
