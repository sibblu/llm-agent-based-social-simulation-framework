import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from ..interfaces.llm_interface import LLM, LLMConfig, Message

load_dotenv()

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_path = os.getenv("LOG_PATH", "logs")
log_path += "\\persona_generator\\"

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_filename = os.path.join(log_path, f"persona_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# logging.basicConfig(
#     filename=log_filename,
#     level=log_level,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )

logger = logging.getLogger(__name__)

class Persona(BaseModel):
    """
    Class representing the structure of a generated persona.
    """
    persona_id: str
    topic: str
    stance: str
    identity: Dict[str, str]
    initial_belief: list
    timestamp: str
    generation_config: Dict[str, Any]

class PersonaGenerator:
    """
    Class for generating personas based on a given topic using LLM.
    """
    def __init__(self, llm: LLM):
        """
        Initialize the PersonaGenerator with an LLM instance.

        Args:
            llm (LLM): An instance of the LLM class to generate personas.
        """
        self.llm = llm

    def generate_personas(self, topic: str) -> Dict[str, Persona]:
        """
        Generate two personas (one in support and one against the topic).

        Args:
            topic (str): The topic for which personas need to be generated.

        Returns:
            Dict[str, Persona]: A dictionary containing the generated personas.
        """
        personas = {}
        for stance in ["positive", "negative"]:
            persona = self._generate_persona(topic, stance)
            personas[f"{stance}"] = persona
        return personas

    def _generate_persona(self, topic: str, stance: str) -> Persona:
        """
        Generate a persona for a given stance on the topic.

        Args:
            topic (str): The topic for which the persona is generated.
            stance (str): The stance of the persona ('positive' or 'negative').

        Returns:
            Persona: The generated persona.
        """
        system_prompt_str = """
        <ROLE>You are an expert in social demography.</ROLE>
        <TASK>Generate a detailed persona based on the following instructions:
            Each persona should include the following elements:
            1. Identity and demographics: Name, Age, Gender, Ethnicity, Education, Occupation.
            2. Initial Belief: A list of actions and beliefs that the persona would have about the topic in the first person. Should have more than 5 belief-action items.
            3. Output schema is given in the <OUTPUT_SCHEMA> section, strictly adhere the given schema.
            4. Give output strictly in JSON format.
            5. An example persona in the output schema is given in the <EXAMPLE_PERSONA>.    
                    
        </TASK>
        <OUTPUT_SCHEMA>
                {
                    "identity": {
                        "name": "",
                        "age": "",
                        "gender": "",
                        "ethnicity": "",
                        "education": "",
                        "occupation": ""
                    },
                    "initial_belief": ["", ""]
                }
        </OUTPUT_SCHEMA>

        <EXAMPLE_PERSONA>
                For example, if the topic is 'Climate Change' and the persona is a denier, the initial belief should include actions and beliefs such as 'I do not believe in scientific evidence', 'I think climate change is a hoax', etc.

                {
                    "identity": {
                        "name": "ABC XYZ",
                        "age": "45",
                        "gender": "Male",
                        "ethnicity": "Asian",
                        "education": "High School",
                        "occupation": "Chef"
                    },
                    "initial_belief": ["I think climate change is a hoax", "I do not believe in scientific evidence"]
                }
        </EXAMPLE_PERSONA>
        """
        
        user_prompt_str = """
        Generate a persona for the topic '{topic}' with a {stance} stance.
        Include identity details (name, age, gender, ethnicity, education, occupation)
        and a list of initial beliefs/actions that this persona would have in the first person.

        <OUTPUT>generated persona</OUTPUT>
        """
        system_prompt = Message(
            role="system", 
            content=system_prompt_str
        )

        user_prompt = Message(
            role="user",
            content=user_prompt_str.format(topic=topic, stance=stance)
        )

        try:
            logger.info(f"Generating persona for topic: {topic}, stance: {stance}")
            response = self.llm.generate_response(messages=[system_prompt, user_prompt])
            persona_data = json.loads(response)

            persona_id = f"{topic}_{stance}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            persona = Persona(
                persona_id=persona_id,
                topic=topic,
                stance=stance,
                identity=persona_data.get("identity", {}),
                initial_belief=persona_data.get("initial_belief", []),
                timestamp=datetime.now().isoformat(),
                generation_config={
                    "model": self.llm.config.model,
                    "temperature": self.llm.config.temperature,
                    "top_p": self.llm.config.top_p,
                    "max_tokens": self.llm.config.max_tokens,
                }
            )
            logger.info(f"Generated persona: {persona_id}")
            return persona
        except Exception as e:
            logger.exception(f"Error generating persona for topic: {topic}, stance: {stance}")
            raise e

if __name__ == "__main__":
    try:
        # Example configuration for the LLM
        config_openai = {
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o-2024-05-13",
            "temperature": 1.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "stream": False,
            "stop": None,
            "response_format": {"type": "json_object"}
        }

        llm = LLM(provider="openai", config=config_openai)
        persona_generator = PersonaGenerator(llm)

        topic = "Climate Change"
        personas = persona_generator.generate_personas(topic=topic)

        for key, persona in personas.items():
            print(f"Persona ID: {persona.persona_id}")
            print(f"Topic: {persona.topic}")
            print(f"Stance: {persona.stance}")
            print(f"Identity: {persona.identity}")
            print(f"Initial Belief: {persona.initial_belief}")
            print(f"Timestamp: {persona.timestamp}")
            print(f"Generation Config: {persona.generation_config}")
            print()

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
    except Exception as e:
        logger.exception("An error occurred")
        print(f"An error occurred: {e}")
