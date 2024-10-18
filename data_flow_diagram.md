# Data Flow and Dependencies

```mermaid
graph TD
    %% External Entities
    user[Researcher]
    aiProvider[AI Provider]

    %% Containers
    subgraph Social Simulation Framework
        orchestrator[Orchestrator]
        agentGenerator[Agent Generator]
        personaGenerator[Persona Generator]
        interactionManager[Interaction Manager]
        promptManager[Prompt Manager]
        llmInterface[LLM Interface]
        evaluationMetrics[Evaluation Metrics]
        dataStorage[Data Storage]
        logStorage[Log Storage]
    end

    %% Data Flow
    user -->|Configures and runs simulations| orchestrator
    orchestrator -->|Requests personas| personaGenerator
    personaGenerator -->|Provides personas| orchestrator
    orchestrator -->|Initializes agents| agentGenerator
    agentGenerator -->|Manages interactions| interactionManager
    interactionManager -->|Formats prompts| promptManager
    promptManager -->|Provides formatted prompts| interactionManager
    interactionManager -->|Interacts with LLM APIs| llmInterface
    llmInterface -->|Sends requests| aiProvider
    aiProvider -->|Provides responses| llmInterface
    llmInterface -->|Provides responses| interactionManager
    interactionManager -->|Manages interactions| agentGenerator
    agentGenerator -->|Updates memories| dataStorage
    orchestrator -->|Evaluates opinion dynamics| evaluationMetrics
    evaluationMetrics -->|Provides evaluation results| orchestrator
    orchestrator -->|Stores interaction transcripts and evaluation results| dataStorage
    orchestrator -->|Stores logs| logStorage
```

* User to Orchestrator: The user configures and runs simulations through the Orchestrator.
* Orchestrator to Persona Generator: The Orchestrator requests the generation of personas based on the given topics.
* Orchestrator to Agent Generator: The Orchestrator initializes agents using the generated personas.
* Orchestrator to Interaction Manager: The Orchestrator manages interactions between agents.
* Orchestrator to Prompt Manager: The Orchestrator formats prompts for interactions.
* Orchestrator to LLM Interface: The Orchestrator interacts with LLM APIs to generate responses.
* Orchestrator to Evaluation Metrics: The Orchestrator evaluates opinion dynamics using the Evaluation Metrics container.
* Orchestrator to Data Storage: The Orchestrator stores interaction transcripts and evaluation results.
* Orchestrator to Log Storage: The Orchestrator stores logs generated during the simulation process.
* LLM Interface to AI Provider: The LLM Interface sends requests to and receives responses from the AI Provider.
