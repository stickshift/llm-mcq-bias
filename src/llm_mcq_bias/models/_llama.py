import logging

import ollama

__all__ = [
    "llama_323b",
]

logger = logging.getLogger(__name__)


def llama_323b(*, prompt: str, options: dict | None = None) -> str:

    # llama 3.2 3B
    model = "llama3.2:3b-instruct-fp16"

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options=options,
    )

    logger.debug(f"Ollama Response: {response}")

    # Parse answer
    answer = response["response"]

    return answer
