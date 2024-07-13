import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import openai
import groq

load_dotenv()

class LLMConfig(BaseModel):
    """
    Configuration class for the LLM settings.
    Uses pydantic for data validation and type checking.
    """
    api_key_env: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

    def get_api_key(self) -> str:
        """
        Retrieve the API key from environment variables.

        Returns:
            str: The API key.
        """
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment for {self.api_key_env}")
        return api_key

class Message(BaseModel):
    """
    Class representing a message in the conversation.
    Ensures the role is one of 'system', 'user', or 'assistant' and contains message content.
    """
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class LLM:
    """
    Wrapper class for interacting with different LLM API providers (OpenAI, Groq).
    """
    def __init__(self, provider: str, config: Dict[str, Any]):
        """
        Initialize the LLM object with the specified provider and configuration.

        Args:
            provider (str): The name of the LLM provider (e.g., 'openai', 'groq').
            config (Dict[str, Any]): Configuration dictionary for the LLM settings.
                - api_key_env: The name of the environment variable containing the API key.
                - model: The name of the model to use for the LLM.
                - temperature: The sampling temperature for the LLM.
                - max_tokens: The maximum number of tokens to generate in the completion.
                - top_p: The nucleus sampling parameter for the LLM.
                - stream: Whether to stream the response from the LLM.
                - stop: List of strings to stop generation at (optional).

        """
        self.provider = provider.lower()
        self.config = LLMConfig(**config)
        self.api_key = self.config.get_api_key()

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "groq":
            self.client = groq.Groq(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_completion(self, messages: List[Message]) -> str:
        """
        Generate a completion for the given messages using the specified LLM provider.

        Args:
            messages (List[Message]): List of messages to be processed by the LLM.

        Returns:
            str: The generated completion text.
        """
        if self.provider == "openai":
            return self._generate_openai_completion(messages)
        elif self.provider == "groq":
            return self._generate_groq_completion(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai_completion(self, messages: List[Message]) -> str:
        """
        Generate a completion using the OpenAI API.

        Args:
            messages (List[Message]): List of messages to be processed by the OpenAI LLM.

        Returns:
            str: The generated completion text.
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[msg.model_dump() for msg in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=self.config.stream,
            stop=self.config.stop,
        )
        return response.choices[0].message.content

    def _generate_groq_completion(self, messages: List[Message]) -> str:
        """
        Generate a completion using the Groq API.

        Args:
            messages (List[Message]): List of messages to be processed by the Groq LLM.

        Returns:
            str: The generated completion text.
        """
        response = self.client.chat.completions.create(
            messages=[msg.model_dump() for msg in messages],
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=self.config.stream,
            stop=self.config.stop,
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    try:
        # Example configuration dictionary
        config_openai = {
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o-2024-05-13",
            "temperature": 1.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "stream": False,
            "stop": None,
        }

        config_groq = {
            "api_key_env": "GROQ_API_KEY",
            "model": "llama3-70b-8192",
            "temperature": 1.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "stream": False,
            "stop": None,
        }

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France? Respond in Hindi and French."),
        ]

        openai_llm = LLM(provider="openai", config=config_openai)
        openai_response = openai_llm.generate_completion(messages=messages)
        
        groq_llm = LLM(provider="groq", config=config_groq)
        groq_response = groq_llm.generate_completion(messages=messages)


        print("OpenAI response:\n{}\n".format(openai_response))
        print("Groq response:\n{}\n".format(groq_response))
    except ValidationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
