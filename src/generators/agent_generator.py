import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from .persona_generator import PersonaGenerator, Persona
from ..interfaces.llm_interface import LLM, Message, LLMConfig
import logging

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_path = os.getenv("LOG_PATH", "logs")
log_path += "\\agent_generator\\"

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_filename = os.path.join(
    log_path, f"agent_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    filename=log_filename,
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class MemoryRecord(BaseModel):
    """
    Data structure for an individual memory record in cumulative memory.
    """

    timestamp: str
    type: str  # "read" or "respond"
    content: str


class ReflectiveMemory(BaseModel):
    """
    Data structure for reflective memory.
    """

    reflections: List[str] = []


class AgentConfig(BaseModel):
    """
    Configuration class for the agent settings.
    """

    persona: Persona
    llm_config: LLMConfig
    closed_world: bool = False
    claim: str


class Agent:
    """
    Class for generating and managing LLM-powered agents.
    """

    prompt_templates = {
        "initialize_persona_prompt": """<ROLE>Role play this person:
            {agent_identity}
            </ROLE>
            {agent_name}, given below are your initial beliefs about the theory XYZ:
            <INITIAL_BELIEF>Your initial belief:
            {initial_belief}
            </INITIAL_BELIEF>

            Now, {agent_name}, you will interact and discuss with another person about the theory XYZ. 
            You can decide to change or maintain your belief about the theory XYZ that claims that {claim} after interacting with other person.
            You would respond such that reflects your honest belief, and you would also read other persons’ response.
            After reading other persons' response, you would be asked about your belief about the theory XYZ that {claim}.""",

        "response_prompt": """{agent_name}, given below is your experience so far, and the other person's message for you. Write your response. The response should reflect your current honest belief about the theory XYZ that claims that {claim}.
            <PAST_REFLECTION>Your past reflections:
                {past_reflection}
            </PAST_REFLECTION>
            <MESSAGE>Other person's message:
                {other_person_response}
            </MESSAGE>
            What is your current honest belief about the theory XYZ that claims {claim}? Specifically, focus on your opinion about XYZ after reading the other person’s message.
            Think step by step about your and write your response.

            Your response: 
            As {agent_name}, I """,


        "reflection_prompt": """Given below is your experience so far, and the other person's message and what you responsed to the other person.  
            <PAST_REFLECTION>Your past reflections:
            {past_reflection}
            </PAST_REFLECTION>

            <MESSAGE>Other person's message: 
            {other_person_response}
            </MESSAGE>
            <RESPONSE>Your response to the other person was:
            {agent_response}
            </RESPONSE>
            
            Now, please reflect on this experience. Summarize your updated experience in a few sentences.""",
        "closed_world_prompt": 
            """Remember, throughout the interactions, you are alone in your room with limited access to the Internet. 
                You cannot search for information about XYZ on the Internet. 
                You cannot go out to ask other people about XYZ. Because you are alone in your room, you cannot leave your room to seek information about XYZ. 
                To form your belief about XYZ, you can only rely on your initial belief about XYZ, along with the information you received from other person.""",
    }

    def __init__(self, agent_config: AgentConfig, provider: str):
        """
        Initialize the agent with the given configuration.

        Args:
            agent_config (AgentConfig): Configuration for the agent, including persona, LLM config, and closed-world constraint.
            provider (str): The LLM provider (e.g., 'openai', 'groq').
        """
        self.agent_id = str(uuid.uuid4())
        self.persona = agent_config.persona
        self.llm = LLM(provider=provider, config=agent_config.llm_config.dict())
        self.closed_world = agent_config.closed_world
        self.claim = agent_config.claim
        self.cumulative_memory = []  # List of MemoryRecord
        self.reflective_memory = ReflectiveMemory()
        self.current_reflection = None
        if not self.closed_world:
            self.system_prompt = Message(
                role="system", content=self._initialize_persona_prompt()
            )
        else:
            # append closed world prompt to the persona prompt
            system_prompt_str = (
                self._initialize_persona_prompt()
                + Agent.prompt_templates["closed_world_prompt"]
            )
            self.system_prompt = Message(role="system", content=system_prompt_str)
        logger.info(f"Initialized agent with persona: {json.dumps(self.persona.model_dump(), indent=2)}")

    def _initialize_persona_prompt(self) -> str:
        """
        Initialize the persona prompt for the agent.

        Returns:
            str: The initialized persona prompt.
        """
        persona = self.persona
        return Agent.prompt_templates["initialize_persona_prompt"].format(
            agent_identity=json.dumps(persona.identity, indent=2),
            agent_name=persona.identity["name"],
            initial_belief=persona.initial_belief,
            claim=self.claim,
        )

    def _create_interaction_prompt(self, other_person_response: str) -> str:
        """
        Create the interaction prompt based on the agent's current state.

        Args:
            other_person_response (str): The response from another person.

        Returns:
            str: The interaction prompt.
        """
        return Agent.prompt_templates["response_prompt"].format(
            agent_name=self.persona.identity["name"],
            past_reflection=(
                self.current_reflection
                if self.current_reflection
                else "No past reflections"
            ),
            other_person_response=other_person_response,
            claim=self.claim,
        )

    def _create_reflection_prompt(
        self, other_person_response: str, previous_response: str
    ) -> str:
        """
        Create the reflection prompt based on the agent's current state.

        Args:
            other_person_response (str): The response from another person.
            previous_response (str): The previous response of the agent.

        Returns:
            str: The reflection prompt.
        """
        if not self.reflective_memory.reflections:
            return Agent.prompt_templates["reflection_prompt"].format(
                past_reflection="No past reflections",
                other_person_response=other_person_response,
                agent_response=previous_response,
            )
        return Agent.prompt_templates["reflection_prompt"].format(
            past_reflection=self.current_reflection,
            other_person_response=other_person_response,
            agent_response=previous_response,
        )
    

    def interact(self, other_person_response: str) -> str:
        """
        Handle the agent's interaction process (read, respond, and reflect).

        Args:
            other_person_response (str): The response from another person.

        Returns:
            str: The agent's response.
        """
        user_prompt = Message(
            role="user", content=self._create_interaction_prompt(other_person_response)
        )
        messages = [self.system_prompt, user_prompt]
        try:
            response = self.llm.generate_completion(messages)
            self._update_memories(response, other_person_response)
            logger.info(f"Agent {self.persona.identity['name']} interacted successfully")
            return response
        except Exception as e:
            logger.exception(f"Error during interaction for agent {self.agent_id}")
            raise e

    def reflect(self, other_person_response: str, previous_response: str) -> str:
        """
        Handle the agent's reflection process.

        Args:
            other_person_response (str): The response from another person.
            previous_response (str): The previous response of the agent.

        Returns:
            str: The agent's reflection.
        """
        reflection_prompt = self._create_reflection_prompt(
            other_person_response, previous_response
        )
        reflection_message = Message(role="user", content=reflection_prompt)
        try:
            reflection_response = self.llm.generate_completion([self.system_prompt, reflection_message])
            self.current_reflection = reflection_response
            logger.info(f"Agent {self.persona.identity['name']} reflected successfully")
            return reflection_response
        except Exception as e:
            logger.exception(f"Error during reflection for agent {self.agent_id}")
            raise e

    def _update_memories(self, agent_response: str, other_person_response: str):
        """
        Update the agent's memories (cumulative and reflective) based on the interaction.

        Args:
            agent_response (str): The agent's response.
            other_person_response (str): The response from another person.
        """
        # Update cumulative memory
        timestamp = datetime.now().isoformat()
        if other_person_response:
            self.cumulative_memory.append(
                MemoryRecord(
                    timestamp=timestamp, type="read", content=other_person_response
                )
            )
        self.cumulative_memory.append(
            MemoryRecord(timestamp=timestamp, type="respond", content=agent_response)
        )

        # Update reflective memory
        reflection_response = self.reflect(other_person_response, agent_response)
        self.reflective_memory.reflections.append(reflection_response)

    def save_memory_stream(self, filepath: str):
        """
        Save the agent's memory stream (cumulative memory) to a JSONL file.

        Args:
            filepath (str): The path to the file where the memory stream should be saved.
        """
        if not os.path.exists(filepath):
            open(filepath, 'w').close()
        with open(filepath, "w") as file:
            for record in self.cumulative_memory:
                file.write(record.json() + "\n")
        logger.info(f"Agent {self.agent_id} memory stream saved to {filepath}")


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
            "response_format": {"type": "json_object"},
        }

        # Initialize persona generator and generate personas
        llm = LLM(provider="openai", config=config_openai)
        persona_generator = PersonaGenerator(llm)
        topic = "Climate Change"
        personas = persona_generator.generate_personas(topic=topic)
        claim = "climate change is caused by human activities"
        # Example agent configuration
        agent_config = AgentConfig(
            persona=personas["negative"],
            llm_config=LLMConfig(**config_openai),
            closed_world=True,
            claim=claim,
        )

        # Initialize agent
        provider = "openai"
        agent = Agent(agent_config, provider)

        # Example interaction and reflection
        other_person_response = "Climate change is real and urgent action is needed. Every year, we see the impacts of climate change, such as extreme weather events, rising sea levels, and loss of biodiversity. If we don't act now, the consequences will be catastrophic."
        response = agent.interact(other_person_response)
        print(f"Agent response: {response}")

        reflection = agent.reflect(other_person_response, response)
        print(f"Agent reflection: {reflection}")

        # Save agent memory stream
        # agent.save_memory_stream(f"/data/agent_{agent.agent_id}_memory.jsonl")

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
    except Exception as e:
        logger.exception("An error occurred")
        print(f"An error occurred: {e}")