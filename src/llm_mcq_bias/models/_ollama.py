import logging

from ollama import generate

__all__ = [
    "ollama",
]

logger = logging.getLogger(__name__)


def ollama(*, model: str, prompt: str, options: dict | None = None) -> str:

    # Send request to ollama
    response = generate(
        model=model,
        prompt=prompt,
        options=options,
    )

    logger.debug(f"Response: {response}")

    # Parse answer
    answer = response["response"]

    return answer
