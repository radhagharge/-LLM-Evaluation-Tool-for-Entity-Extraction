import os
import time
from openai import OpenAI
from models.base_model import ModelInterface

class OpenAIModel(ModelInterface):
    """Adapter for OpenAI GPT models."""
    

    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("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"))
        self.model_name = model_name

    def extract_entities(self, text: str, entities: list[str]):
        start = time.time()
        prompt = (
            f"Extract the following entities from the text below "
            f"and return the result in JSON format.\n\n"
            f"Entities to extract: {entities}\n\nText:\n{text[:4000]}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        latency = time.time() - start
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        cost = tokens * 0.00001
        return content, latency, tokens, cost
