from typing import List, Dict

class PromptManager:
    """
    Class for managing prompt structure formatting for different LLM models.
    """
    @staticmethod
    def format_prompt(provider: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format the prompt according to the provider's requirements.

        Args:
            provider (str): The name of the LLM provider.
            messages (List[Dict[str, str]]): List of messages to format.

        Returns:
            List[Dict[str, str]]: Formatted messages.
        """
        if provider == "openai":
            return PromptManager._format_openai_prompt(messages)
        elif provider == "groq":
            return PromptManager._format_groq_prompt(messages)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _format_openai_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # OpenAI's SDK uses the messages as they are
        return messages

    @staticmethod
    def _format_groq_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Groq's SDK uses the messages as they are
        return messages
