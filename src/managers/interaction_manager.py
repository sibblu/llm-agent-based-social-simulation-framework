import os
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

from pydantic import ValidationError
from ..generators.agent_generator import Agent

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_path = os.getenv("LOG_PATH", "logs")
log_path += "\\interaction_manager\\"

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_filename = os.path.join(
    log_path, f"interaction_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    filename=log_filename,
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class InteractionManager:
    """
    Class to manage and orchestrate interactions between multiple agents.
    """

    def __init__(self, agents: List[Agent], opening_message: str, max_interactions: int = 100):
        """
        Initialize the InteractionManager with a list of agents and the opening message.

        Args:
            agents (List[Agent]): List of agents participating in the interaction.
            opening_message (str): The opening message to start the interaction.
        """
        self.agents = agents
        self.transcript = []
        self.opening_message = opening_message
        self.max_interactions = max_interactions
        logger.info("Initialized InteractionManager with agents and opening message.")

    def start_interaction(self):
        """
        Start the interaction process between agents.
        """
        current_message = self.opening_message
        continue_interaction = True
        agent_index = 0
        messages_exchanged = 0
        
        print("Starting Interaction...")
        print("Interaction Manager --> {agent}".format(agent=self.agents[agent_index].persona.identity["name"]))
        print("Opening message: {message}".format(message=self.opening_message))
        print("\n-----------------------------------")
        while continue_interaction and messages_exchanged <= self.max_interactions:
            agent = self.agents[agent_index]
            response = agent.interact(current_message)
            print("{agent1} --> {agent2}".format(agent1=agent.persona.identity["name"], agent2=self.agents[(agent_index + 1) % len(self.agents)].persona.identity["name"]))
            print("Response: {response}".format(response=response))
            print("\n-----------------------------------\n")
            timestamp = datetime.now().isoformat()
            self.transcript.append({
                "agent_id": agent.agent_id,
                "stance": agent.persona.stance,
                "response": response,
                "current_reflection": agent.current_reflection,
                "timestamp": timestamp,
                "llm_model": agent.llm.config.model,
            })

            logger.info("Agent {agent_name} responded with: {response}".format(agent_name=agent.persona.identity["name"], response=response))

            if "I have decided to discontinue the conversation" in response:
                logger.info("Agent {agent_name} has decided to discontinue the conversation.".format(agent_name=agent.persona.identity["name"]))                
                continue_interaction = False
                break

            messages_exchanged += 1
            current_message = response
            agent_index = (agent_index + 1) % len(self.agents)

        logger.info("\nInteraction finished.")
        logger.info(f"\nTotal messages exchanged: {messages_exchanged}")

    def save_transcript(self, filepath: str):
        """
        Save the interaction transcript to a JSONL file.

        Args:
            filepath (str): The path to the file where the transcript should be saved.
        """
        if not os.path.exists(filepath):
            open(filepath, 'w').close()
        with open(filepath, "w") as file:
            for record in self.transcript:
                file.write(json.dumps(record) + "\n")
        logger.info(f"Transcript saved to {filepath}")


if __name__ == "__main__":
    from ..generators.agent_generator import AgentConfig, LLMConfig, PersonaGenerator, LLM

    try:
        # Example configuration for the LLM
        config_openai = {
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o-2024-05-13",
            "temperature": 1.0,
            "max_tokens": 2048,
            "top_p": 1.0,
            "stream": False,
            "stop": None,
            "response_format": {"type": "json_object"},
        }
        config_groq = {
            "api_key_env": "GROQ_API_KEY",
            "model": "llama3-70b-8192",
            "temperature": 1.0,
            "max_tokens": 2048,
            "top_p": 1.0,
            "stream": False,
            "stop": None,
            "response_format": {"type": "json_object"},
        }
        llm_provider_openai = "openai"
        llm_provider_groq = "groq"
        # Initialize persona generator and generate personas
        llm = LLM(provider=llm_provider_groq, config=config_groq)
        persona_generator = PersonaGenerator(llm)
        topic = "Climate Change is human induced."
        personas = persona_generator.generate_personas(topic=topic)

        # Example agent configurations
        agent_config_1 = AgentConfig(
            persona=personas["negative"],
            llm_config=LLMConfig(**config_groq),
            closed_world=True,
        )

        agent_config_2 = AgentConfig(
            persona=personas["positive"],
            llm_config=LLMConfig(**config_groq),
            closed_world=True,
        )

        agent_1 = Agent(agent_config_1, llm_provider_groq)
        agent_2 = Agent(agent_config_2, llm_provider_groq)

        # Initialize InteractionManager with agents and opening message
        opening_message = "Climate change is a pressing global issue, because {topic} What are your thoughts?".format(topic=topic)
        interaction_manager = InteractionManager([agent_1, agent_2], opening_message, max_interactions=20)

        # Start interaction
        interaction_manager.start_interaction()

        # Save transcript
        transcript_filepath = "./data/interaction_transcripts/transcript_{timestamp}.jsonl".format(timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'))
        interaction_manager.save_transcript(transcript_filepath)

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
    except Exception as e:
        logger.exception("An error occurred")
        print(f"An error occurred: {e}")
