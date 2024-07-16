import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from datetime import datetime

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_path = os.getenv("LOG_PATH", "logs")
log_path += "\\opinion_dynamics\\"

if not os.path.exists(log_path):
    os.makedirs(log_path)

log_filename = os.path.join(
    log_path, f"opinion_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    filename=log_filename,
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class OpinionDynamics:
    """
    Class to calculate opinion dynamics metrics and plot opinion trajectories.
    """
    def __init__(self, transcript: List[Dict[str, Any]]):
        """
        Initialize the OpinionDynamics class with the chat transcript.

        Args:
            transcript (List[Dict[str, Any]]): The chat transcript to evaluate.
        """
        self.transcript = transcript
        logger.info("Initialized OpinionDynamics with the provided chat transcript.")

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate the Bias (B) and Diversity (D) metrics for each agent.

        Returns:
            Dict[str, Dict[str, float]]: The calculated Bias and Diversity metrics for each agent.
        """
        positive_scores = [entry["score"] for entry in self.transcript if entry["stance"] == "positive"]
        negative_scores = [entry["score"] for entry in self.transcript if entry["stance"] == "negative"]

        metrics = {
            "positive": {
                "Bias": np.mean(positive_scores),
                "Diversity": np.std(positive_scores)
            },
            "negative": {
                "Bias": np.mean(negative_scores),
                "Diversity": np.std(negative_scores)
            }
        }

        logger.info(f"Calculated metrics: {metrics}")
        return metrics

    def plot_opinion_trajectories(self, save_path: str = "opinion_trajectories.png"):
        """
        Plot the opinion trajectories over time.
        """
        positive_scores = [entry["score"] for entry in self.transcript if entry["stance"] == "positive"]
        negative_scores = [entry["score"] for entry in self.transcript if entry["stance"] == "negative"]
        
        # Determine the number of time steps
        time_steps = list(range(1, len(positive_scores) + 1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, positive_scores, marker='o', color='darkblue', label='Positive Stance Agent')
        plt.plot(time_steps, negative_scores, marker='o', color='red', label='Negative Stance Agent')
        
        plt.xlabel('Time Step')
        plt.ylabel('Opinion Score')
        plt.title('Opinion Trajectories Over Time')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.ylim(-2, 2)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        logger.info("Plotted opinion trajectories over time.")

if __name__ == "__main__":
    try:
        evaluated_transcript_filepath = "./data/interaction_transcripts/evaluated_transcript_20240716_054625.jsonl"

        # Load the evaluated transcript
        with open(evaluated_transcript_filepath, "r") as file:
            transcript = [json.loads(line) for line in file]
    
        # Initialize OpinionDynamics with the transcript
        opinion_dynamics = OpinionDynamics(transcript)

        # Calculate metrics
        metrics = opinion_dynamics.calculate_metrics()
        print(f"Positive Agent - Bias: {metrics['positive']['Bias']}, Diversity: {metrics['positive']['Diversity']}")
        print(f"Negative Agent - Bias: {metrics['negative']['Bias']}, Diversity: {metrics['negative']['Diversity']}")

        #path to save the plot
        save_path = "./data/opinion_trajectories.png"
        # Plot opinion trajectories
        opinion_dynamics.plot_opinion_trajectories(save_path)

    except Exception as e:
        logger.exception("An error occurred")
        print(f"An error occurred: {e}")
