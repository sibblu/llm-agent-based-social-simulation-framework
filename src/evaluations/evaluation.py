import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

# Set up logging
# log_level = os.getenv("LOG_LEVEL", "INFO")
# log_path = os.getenv("LOG_PATH", "logs")
# log_path += "\\evaluation\\"

# if not os.path.exists(log_path):
#     os.makedirs(log_path)

# log_filename = os.path.join(
#     log_path, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# )

# logging.basicConfig(
#     level=log_level,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

logger = logging.getLogger(__name__)


class Evaluation:
    """
    Class to evaluate chat transcripts using Flan-T5 XXL model for opinion classification.
    """

    def __init__(self, model_name: str = "google/flan-t5-xl"):
        """
        Initialize the Evaluation class by loading the Flan-T5 XXL model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to load.
        """
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     model_name, device="cpu", load_in_8bit=True
        # )
        # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded model {model_name}")

    def classify_response(self, response: str, claim: str) -> int:
        """
        Classify the opinion score of a response using the Flan-T5 XXL model.

        Args:
            response (str): The response to classify.
            claim (str): The claim for context in the classification.

        Returns:
            int: The classified opinion score between -2 and 2.
        """
        prompt = f"""Here's a statement made by a person after seeing a message.
        --{response}
        --What is their belief about the Theory XYZ that claims that {claim}, right after they saw this message, on a 5-point Likert scale: -2: (Strongly negative opinion about XYZ), 
        -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ), 
        1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ).

        Answer with a single opinion value within the options -2, -1, 0, 1, 2."""

        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids, max_length=10)
        score = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            score = int(score)
            if score in [-2, -1, 0, 1, 2]:
                return score
            else:
                raise ValueError("Invalid score generated")
        except ValueError:
            logger.error(f"Invalid score '{score}' generated for response: {response}")
            return 0

    def evaluate_transcript(
        self, transcript: List[Dict[str, Any]], claim: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the entire chat transcript and append the classified scores.

        Args:
            transcript (List[Dict[str, Any]]): The chat transcript to evaluate.
            claim (str): The claim for context in the classification.

        Returns:
            List[Dict[str, Any]]: The updated transcript with appended scores.
        """
        for entry in transcript:
            response = entry["response"]
            score = self.classify_response(response, claim)
            entry["score"] = score
            logger.info(f"Classified response: {response[:50]}... with score: {score}")
        return transcript

    def save_transcript(self, transcript: List[Dict[str, Any]], filepath: str):
        """
        Save the evaluated transcript to a JSONL file.

        Args:
            transcript (List[Dict[str, Any]]): The evaluated transcript.
            filepath (str): The path to the file where the transcript should be saved.
        """
        if not os.path.exists(filepath):
            open(filepath, "w").close()
        with open(filepath, "w") as file:
            for record in transcript:
                file.write(json.dumps(record) + "\n")
        logger.info(f"Transcript saved to {filepath}")


if __name__ == "__main__":
    try:
        # Example configuration
        transcript_filepath = (
            "./data/interaction_transcripts/transcript_20240716_080039.jsonl"
        )
        evaluated_transcript_filepath = (
            "./data/interaction_transcripts/evaluated_transcript_20240716_080039.jsonl"
        )
        claim = "climate change is caused by human activities"

        # Load the transcript
        with open(transcript_filepath, "r") as file:
            transcript = [json.loads(line) for line in file]

        # Initialize the Evaluation class
        evaluation = Evaluation()

        # Evaluate the transcript
        evaluated_transcript = evaluation.evaluate_transcript(transcript, claim)

        # Save the evaluated transcript
        evaluation.save_transcript(evaluated_transcript, evaluated_transcript_filepath)

    except Exception as e:
        logger.exception("An error occurred")
        print(f"An error occurred: {e}")
