
# LLM Agent-Based Social Simulation Framework
The Social Simulation Framework is a comprehensive system designed to simulate social interactions and evaluate opinion dynamics using generative AI. This framework orchestrates the entire simulation pipeline, including generating personas, initializing agents, managing interactions, formatting prompts, interacting with Large Language Model (LLM) APIs, and evaluating results. The modular design ensures scalability, maintainability, and ease of integration, making it a robust and flexible tool for researchers and developers interested in social simulations and opinion dynamics.

## Features

- **Persona Generation**: Create detailed personas with identity, demographics, and initial beliefs based on given topics.
- **Agent Management**: Generate and manage LLM-powered agents capable of interacting and reflecting on past interactions.
- **Interaction Management**: Facilitate and manage interactions between agents, ensuring adherence to defined protocols.
- **Prompt Management**: Format and structure prompts for interactions, ensuring compatibility with various LLM APIs.
- **LLM Integration**: Seamlessly interact with external LLM APIs like OpenAI and Groq for generating responses and personas.
- **Evaluation Metrics**: Calculate and evaluate opinion dynamics metrics such as bias and diversity.
- **Data Storage**: Efficiently store interaction transcripts, evaluation results, and logs for further analysis.


## Installation
1. Clone the repository:
```bash
git clone https://github.com/sibblu/llm-agent-based-social-simulation-framework.git
cd llm-agent-based-social-simulation-framework
```
2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source .venv/bin/activate # on Windows, use .venv\Scripts\activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Configure the `.env` file with your API keys and other necessary configurations.
2. Run the orchestrator to start the simulation:
```bash
python orchestrator.py
```
3. Check the data and logs directories for interaction transcripts, evaluation results, and logs.

## Architecture, Data-Flow and Sequence Diagrams
Check out the detailed architecture, data-flow, and sequence diagrams in the respective markdown files for a comprehensive understanding of the framework's design and functionality.
- **[Architecture Diagram](architecture.md)**: Overview of the system architecture and components using C4 model.
- **[Data Flow Diagram](data_flow_diagram.md)**: Visualization of data flow and dependencies between system components.
- **[Sequence Diagram](sequence_diagram.md)**: Sequence of interactions between system components during a simulation run.

## Core Python Scripts

The core Python scripts in the Social Simulation Framework repository are organized into several directories, each serving a specific purpose. Below is the detailed file and folder hierarchy for the core scripts.

### Root Directory

social-simulation-framework/ 
├── src/ 
│ ├── generators/ 
│ │ ├── agent_generator.py │ │ ├── persona_generator.py 
│ ├── interfaces/ 
│ │ ├── llm_interface.py 
│ ├── managers/ 
│ │ ├── prompt_manager.py │ │ ├── interaction_manager.py 
│ ├── evaluations/ 
│ │ ├── [opinion_dynamics.py]
├── [orchestrator.py]


### Directory and File Descriptions

#### `src/`

- **Purpose**: Contains the main source code for the framework.

##### `generators/`

- **Purpose**: Contains code for generating agents and personas.
- **Files**:
  - `agent_generator.py`: Code for generating and managing LLM-powered agents.
  - `persona_generator.py`: Code for generating personas based on given topics.

##### `interfaces/`

- **Purpose**: Contains code for interfacing with external systems.
- **Files**:
  - `llm_interface.py`: Code for interacting with LLM APIs like OpenAI and Groq.

##### `managers/`

- **Purpose**: Contains code for managing various aspects of the framework.
- **Files**:
  - `prompt_manager.py`: Code for managing prompt formatting and structure.
  - `interaction_manager.py`: Code for managing interactions between agents.

##### `evaluations/`

- **Purpose**: Contains code for evaluating opinion dynamics metrics.
- **Files**:
  - `opinion_dynamics.py`: Code for evaluating opinion dynamics metrics.

#### Root Files

- **`orchestrator.py`**: Main orchestrator script that coordinates the entire simulation pipeline.

#### Summary

The core Python scripts are organized to separate different concerns such as agent and persona generation, external system interfacing, prompt management, interaction management, and opinion dynamics evaluation. Each directory and file has a specific purpose, making the codebase modular and easy to navigate. This organization facilitates scalability, maintainability, and ease of understanding.
