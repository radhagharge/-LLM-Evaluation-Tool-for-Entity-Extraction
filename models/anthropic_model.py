import os
import time
import anthropic
from models.base_model import ModelInterface


class AnthropicModel(ModelInterface):
    """
    Adapter for Anthropic Claude models (e.g., Claude 3 Sonnet).
    Handles entity extraction and logs latency, token usage, and cost.
    """

    def __init__(self, model_name="claude-3-sonnet-20240229"):
        # Initialize Anthropic client using environment variable
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def extract_entities(self, text: str, entities: list[str]):
        start_time = time.time()

        # Build the prompt
        prompt = (
            f"Extract the following entities from the text below and return the result in JSON format only.\n\n"
            f"Entities to extract: {entities}\n\n"
            f"Text:\n{text[:4000]}"
        )

        # Send request to Claude
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        latency = time.time() - start_time

        # Extract data
        content = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        cost = tokens * 0.000008  # approximate cost

        return content, latency, tokens, cost
