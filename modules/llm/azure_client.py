"""
One Azure OpenAI client factory for Whisper and GPT (same resource, different deployments).

* ``AZURE_OPENAI_API_VERSION`` set → ``openai.AzureOpenAI(azure_endpoint=…, api_version=…)``; the endpoint is the
  resource URL, e.g. ``https://<resource>.openai.azure.com/``;
* not set → ``openai.OpenAI(base_url=…)`` for the v1 endpoint ``https://<resource>.openai.azure.com/openai/v1/``
  (what ``speech_to_text_whisper`` already uses).

In both cases ``model=`` is the deployment name from config.yaml.
"""
from typing import Any, Dict


class MissingCredentials(RuntimeError):
    """AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT are not set."""


def create_client(llm: Dict[str, Any]):
    """
    :param llm: The ``llm`` config section (secrets merged from .env by core.config)
    :return: openai client
    """
    key, endpoint = llm.get("azure_api_key"), llm.get("azure_endpoint")
    if not key or not endpoint:
        raise MissingCredentials("Azure OpenAI needs AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (.env)")
    # Few SDK retries / a bounded timeout: callers handle failures (fallback, backoff) themselves
    options = {"api_key": key, "timeout": float(llm.get("request_timeout", 30)),
               "max_retries": int(llm.get("max_retries", 1))}
    if llm.get("azure_api_version"):
        from openai import AzureOpenAI

        return AzureOpenAI(azure_endpoint=endpoint, api_version=llm["azure_api_version"], **options)
    from openai import OpenAI

    return OpenAI(base_url=endpoint, **options)
