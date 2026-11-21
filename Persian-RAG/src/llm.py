from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS
)


class GeminiLLM:

    def __init__(self):

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Check your .env file."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model_name = LLM_MODEL_NAME

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS
            )
        )

        if not response.text:
            raise RuntimeError("The LLM returned an empty response.")

        return response.text