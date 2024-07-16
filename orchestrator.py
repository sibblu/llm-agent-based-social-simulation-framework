import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.evaluations.evaluation import Evaluation
from src.evaluations.opinion_dynamics import OpinionDynamics
from src.generators.agent_generator import Agent, AgentConfig, LLMConfig
from src.generators.persona_generator import PersonaGenerator
from src.interfaces.llm_interface import LLM
from src.managers.interaction_manager import InteractionManager

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_path = os.getenv("LOG_PATH", "logs")
log_path += "\\orchestrator\\"

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_filename = os.path.join(
    log_path, f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    filename=log_filename,
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Class to orchestrate the entire pipeline of the social simulation framework.
    """
    def __init__(self, topics: List[str], llm_config: Dict[str, Any], provider: str, max_exchanges: int = 30):
        """
        Initialize the Orchestrator with a list of topics, LLM configuration, provider, and the maximum number of message exchanges.

        Args:
            topics (List[str]): List of topics for the simulation.
            llm_config (Dict[str, Any]): Configuration dictionary for the LLM.
            provider (str): The name of the LLM provider (e.g., 'openai', 'groq').
            max_exchanges (int): Maximum number of message exchanges.
        """
        self.topics = topics
        self.llm_config = llm_config
        self.provider = provider
        self.max_exchanges = max_exchanges
        self.data_path = "data"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        logger.info("Initialized Orchestrator with topics, llm_config, provider, and max_exchanges.")

    def run_simulation(self):
        """
        Run the entire simulation pipeline for each topic.
        """
        for topic in self.topics:
            logger.info(f"Starting simulation for topic: {topic}")
            topic_path = os.path.join(self.data_path, topic.replace(" ", "_"))
            if not os.path.exists(topic_path):
                os.makedirs(topic_path)

            # Step 1: No Role Playing and No Interaction Condition
            no_role_play_reports = self.no_role_play_no_interaction(topic)
            no_role_play_reports_path = os.path.join(topic_path, "no_role_play_no_interaction.jsonl")
            self.save_jsonl(no_role_play_reports, no_role_play_reports_path)
            self.evaluate_no_interaction(no_role_play_reports, topic, topic_path, "no_role_play_no_interaction")

            # Step 2: Generate personas
            personas = self.generate_personas(topic)
            self.save_json(personas, os.path.join(topic_path, "personas.json"))

            # Step 3: No Interaction Condition
            no_interaction_reports = self.no_interaction(personas, topic)
            no_interaction_reports_path = os.path.join(topic_path, "no_interaction.jsonl")
            self.save_jsonl(no_interaction_reports, no_interaction_reports_path)
            self.evaluate_no_interaction(no_interaction_reports, topic, topic_path, "no_interaction")

            # Step 4: Initialize agents
            agents = self.initialize_agents(personas, topic)

            # Step 5: Run interaction manager
            opening_message = f"What are your thoughts on the topic: {topic}?"
            interaction_manager = InteractionManager(agents, opening_message)
            interaction_manager.start_interaction(max_exchanges=self.max_exchanges)
            transcript_path = os.path.join(topic_path, "transcript.jsonl")
            interaction_manager.save_transcript(transcript_path)

            # Step 6: Evaluate chat transcript
            evaluated_transcript = self.evaluate_transcript(transcript_path, topic)
            evaluated_transcript_path = os.path.join(topic_path, "evaluated_transcript.jsonl")
            self.save_jsonl(evaluated_transcript, evaluated_transcript_path)

            # Step 7: Calculate opinion dynamics metrics and plot opinion trajectories
            opinion_dynamics = OpinionDynamics(evaluated_transcript)
            metrics = opinion_dynamics.calculate_metrics()
            metrics_path = os.path.join(topic_path, "metrics.json")
            self.save_json(metrics, metrics_path)
            opinion_dynamics.plot_opinion_trajectories(os.path.join(topic_path, "opinion_trajectories.png"))

            logger.info(f"Completed simulation for topic: {topic}")

    def no_role_play_no_interaction(self, topic: str) -> List[Dict[str, Any]]:
        """
        Generate opinion reports without role playing and without interaction.

        Args:
            topic (str): The topic for the opinion reports.

        Returns:
            List[Dict[str, Any]]: The generated opinion reports.
        """
        llm = LLM(provider=self.provider, config=self.llm_config)
        reports = []

        prompt = f"Provide your opinion on the topic: {topic}."

        for i in range(10):
            response = llm.generate_completion([{"role": "user", "content": prompt}])
            report = {
                "message_number": i,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            reports.append(report)
            logger.info(f"Generated opinion report without role playing and interaction: {response}")

        return reports

    def no_interaction(self, personas: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
        """
        Generate opinion reports with role playing but without interaction.

        Args:
            personas (Dict[str, Any]): The personas for generating reports.
            topic (str): The topic for the opinion reports.

        Returns:
            List[Dict[str, Any]]: The generated opinion reports.
        """
        reports = []
        for stance, persona in personas.items():
            agent_config = AgentConfig(
                persona=persona,
                llm_config=LLMConfig(**self.llm_config),
                closed_world=False,
                claim=topic,
            )
            agent = Agent(agent_config, self.provider)

            prompt = f"Provide your opinion on the topic: {topic}."

            for i in range(10):
                response = agent.interact(prompt)
                report = {
                    "message_number": i,
                    "agent_name": persona.identity["name"],
                    "stance": stance,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                reports.append(report)
                logger.info(f"Generated opinion report without interaction: {response}")

        return reports

    def generate_personas(self, topic: str) -> Dict[str, Any]:
        """
        Generate personas for the given topic.

        Args:
            topic (str): The topic for persona generation.

        Returns:
            Dict[str, Any]: The generated personas.
        """
        llm = LLM(provider=self.provider, config=self.llm_config)
        persona_generator = PersonaGenerator(llm)
        personas = persona_generator.generate_personas(topic=topic)
        return personas

    def initialize_agents(self, personas: Dict[str, Any], topic: str) -> List[Agent]:
        """
        Initialize agents with the given personas and topic.

        Args:
            personas (Dict[str, Any]): The personas for initializing agents.
            topic (str): The topic for the agents.

        Returns:
            List[Agent]: The initialized agents.
        """
        agents = []
        for stance, persona in personas.items():
            agent_config = AgentConfig(
                persona=persona,
                llm_config=LLMConfig(**self.llm_config),
                closed_world=False,
                claim=topic,
            )
            agent = Agent(agent_config, self.provider)
            agents.append(agent)
        return agents

    def evaluate_transcript(self, transcript_path: str, topic: str) -> List[Dict[str, Any]]:
        """
        Evaluate the chat transcript to classify opinion scores.

        Args:
            transcript_path (str): The path to the chat transcript.
            topic (str): The topic for evaluation.

        Returns:
            List[Dict[str, Any]]: The evaluated chat transcript.
        """
        with open(transcript_path, "r") as file:
            transcript = [json.loads(line) for line in file]

        evaluation = Evaluation()
        evaluated_transcript = evaluation.evaluate_transcript(transcript, topic)
        return evaluated_transcript

    def evaluate_no_interaction(self, reports: List[Dict[str, Any]], topic: str, topic_path: str, condition: str):
        """
        Evaluate the control condition transcripts and calculate opinion dynamics metrics.

        Args:
            reports (List[Dict[str, Any]]): The control condition reports.
            topic (str): The topic for evaluation.
            topic_path (str): The path to save the evaluation results.
            condition (str): The control condition name.
        """
        evaluation = Evaluation()
        evaluated_reports = evaluation.evaluate_transcript(reports, topic)
        evaluated_reports_path = os.path.join(topic_path, f"evaluated_{condition}.jsonl")
        self.save_jsonl(evaluated_reports, evaluated_reports_path)

        scores = [report["score"] for report in evaluated_reports]
        bias = np.mean(scores)
        diversity = np.std(scores)
        metrics = {"Bias": bias, "Diversity": diversity}
        metrics_path = os.path.join(topic_path, f"{condition}_metrics.json")
        self.save_json(metrics, metrics_path)
        self.plot_opinion_trajectories(scores, os.path.join(topic_path, f"{condition}_trajectories.png"))

    def plot_opinion_trajectories(self, scores: List[float], save_path: str):
        """
        Plot the opinion trajectories over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(1, len(scores) + 1)), scores, marker='o', color='darkblue')
        
        plt.xlabel('Time Step')
        plt.ylabel('Opinion Score')
        plt.title('Opinion Trajectories Over Time')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.ylim(-2, 2)
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        logger.info("Plotted opinion trajectories over time.")

    def save_json(self, data: Any, filepath: str):
        """
        Save data to a JSON file.

        Args:
            data (Any): The data to save.
            filepath (str): The path to the JSON file.
        """
        with open(filepath, "w") as file:
            json.dump(data, file, indent=2)
        logger.info(f"Saved data to {filepath}")

    def save_jsonl(self, data: List[Dict[str, Any]], filepath: str):
        """
        Save data to a JSONL file.

        Args:
            data (List[Dict[str, Any]]): The data to save.
            filepath (str): The path to the JSONL file.
        """
        with open(filepath, "w") as file:
            for record in data:
                file.write(json.dumps(record) + "\n")
        logger.info(f"Saved data to {filepath}")

if __name__ == "__main__":
    try:
        topics = ["Climate Change is human-induced.", "AI will take over many jobs."]
        
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
        provider = "openai"

        orchestrator = Orchestrator(topics, config_openai, provider)
        orchestrator.run_simulation()

    except Exception as e:
        logger.exception("An error occurred in the orchestrator")
        print(f"An error occurred: {e}")
