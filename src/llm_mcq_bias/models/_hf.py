import logging

from transformers import pipeline, TextGenerationPipeline

from llm_mcq_bias.tools import torch_device

__all__ = [
    "hf",
]

logger = logging.getLogger(__name__)


def hf(*, model: str, prompt: str, options: dict | None = None) -> str:
    # Defaults
    options = options if options is not None else {}

    transformer = pipeline(
        "text-generation",
        model=model,
        device=torch_device(),
        **options,
    )

    response = transformer(
        [{"role": "user", "content": prompt}],
        return_full_text=False,
    )[0]

    logger.debug(f"Response: {response}")

    # Parse answer
    answer = response["generated_text"]

    return answer
