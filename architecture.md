## Context Diagram

```mermaid
---
theme: default
---
C4Context
    title Social Simulation Framework - Context Diagram

    Person(user, "Researcher", "Person who configures and runs the simulations") 
    System(socialSimFramework, "Social Simulation Framework", "Framework for simulating social interactions and evaluating opinion dynamics")
    System_Ext(aiProvider, "AI Provider", "External LLM API provider like OpenAI or Groq")

    Enterprise_Boundary(b0, "Social Simulation Framework") {
        System(socialSimFramework, "Social Simulation Framework", "Framework for simulating social interactions and evaluating opinion dynamics")
    }

    Rel_R(user, socialSimFramework, "Configures and runs simulations")
    Rel_D(socialSimFramework, aiProvider, "Interacts with LLM API for generating responses")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

## Container Diagram

```mermaid
C4Container
    title Social Simulation Framework - Container Diagram

    Person(user, "Researcher", "Person who configures and runs the simulations")
    System_Ext(aiProvider, "AI Provider", "External LLM API provider like OpenAI or Groq")

    Container_Boundary(c0, "Social Simulation Framework") {
        Container(orchestrator, "Orchestrator", "Python Class", "Coordinates the entire simulation pipeline")
        Container(agentGenerator, "Agent Generator", "Python Class", "Generates and manages LLM-powered agents")
        Container(personaGenerator, "Persona Generator", "Python Class", "Generates personas based on given topics")
        Container(interactionManager, "Interaction Manager", "Python Class", "Manages interactions between agents")
        Container(promptManager, "Prompt Manager", "Python Class", "Manages prompt formatting and structure")
        Container(llmInterface, "LLM Interface", "Python Class", "Interacts with LLM APIs like OpenAI and Groq")
        Container(evaluationMetrics, "Evaluation Metrics", "Python Class", "Calculates opinion dynamics metrics")
        Container(linguisticEvaluation, "Linguistic Evaluation", "Python Class", "Evaluates linguistic aspects of interactions")
        Container(dataStorage, "Data Storage", "File System", "Stores interaction transcripts and evaluation results")
        Container(logStorage, "Log Storage", "File System", "Stores logs")
    }

    Rel_R(user, orchestrator, "Configures and runs simulations")
    Rel_D(orchestrator, personaGenerator, "Generates personas")
    Rel_D(orchestrator, agentGenerator, "Initializes agents")
    Rel_D(orchestrator, interactionManager, "Manages interactions")
    Rel_D(orchestrator, promptManager, "Formats prompts")
    Rel_D(orchestrator, llmInterface, "Interacts with LLM APIs")
    Rel_D(orchestrator, evaluationMetrics, "Evaluates opinion dynamics")
    Rel_D(orchestrator, linguisticEvaluation, "Evaluates linguistic aspects")
    Rel_D(orchestrator, dataStorage, "Stores interaction transcripts and evaluation results")
    Rel_D(orchestrator, logStorage, "Stores logs")

    Rel_U(llmInterface, aiProvider, "Interacts with LLM API for generating responses")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Explanation of the Container Diagram

- External Entities
  - User (Researcher):

    - Role: Configures and runs the simulations.
    - Interaction: Interacts with the Orchestrator to set up and execute simulations.
  - AI Provider:

    - Role: Provides external LLM API services (e.g., OpenAI, Groq).
    - Interaction: The LLM Interface interacts with the AI Provider to generate responses and personas.

#### Containers within the Social Simulation Framework

**Orchestrator:**

    Description: Coordinates the entire simulation pipeline.

    Responsibilities:
		- Configures and runs simulations.
		- Manages the flow of data between other containers.
		- Initiates the generation of personas and agents.
		- Manages interactions and evaluations.

    Interactions:
		- Receives configuration and execution commands from the User.
		- Interacts with other containers to perform its tasks.

**Agent Generator:**

    Description: Generates and manages LLM-powered agents.
	Responsibilities:
		Creates agents based on personas.
		Manages agent interactions and memory updates.
	Interactions:
		Receives persona data from the Orchestrator.
		Interacts with the LLM Interface to generate agent responses.

**Persona Generator:**

    Description: Generates personas based on given topics.
	Responsibilities:
		Creates detailed personas with identity, demographics, and initial beliefs.
	Interactions:
		Receives topic data from the Orchestrator.
		Provides generated personas to the Orchestrator.

**Interaction Manager:**

    Description: Manages interactions between agents.
	Responsibilities:
		Facilitates the exchange of messages between agents.
		Ensures interactions follow the defined protocols.
	Interactions:
		Receives interaction commands from the Orchestrator.
		Manages the flow of messages between agents.

**Prompt Manager:**

    Description: Manages prompt formatting and structure.
	Responsibilities:
		Formats prompts for interactions.
		Ensures prompts are structured correctly for LLM processing.
	Interactions:
		Receives prompt formatting requests from the Orchestrator.
		Provides formatted prompts to the Orchestrator and LLM Interface.

**LLM Interface:**

    Description: Interacts with LLM APIs like OpenAI and Groq.
	Responsibilities:
		Sends requests to external LLM APIs.
		Receives and processes responses from LLM APIs.
	Interactions:
		Receives interaction requests from the Orchestrator and Agent Generator.
		Sends requests to and receives responses from the AI Provider.

**Evaluation Metrics:**

    Description: Calculates opinion dynamics metrics.
	Responsibilities:
		Evaluates interaction transcripts.
		Calculates metrics such as bias and diversity.
	Interactions:
		Receives transcripts from the Orchestrator.
		Provides evaluation results to the Orchestrator.

**Data Storage:**

    Description: Stores interaction transcripts and evaluation results.
	Responsibilities:
		Saves transcripts of interactions.
		Stores evaluation results for further analysis.
	Interactions:
		Receives data from the Orchestrator.
		Provides stored data when requested.

**Log Storage:**

    Description: Stores logs.
	Responsibilities:
		Saves logs generated during the simulation process.
	Interactions:
		Receives log data from the Orchestrator and other containers.
	Provides stored logs for debugging and analysis.

## Component Diagrams
### Component Diagram for Orchestrator
```mermaid
C4Component
    title Component Diagram for Orchestrator

    Person(user, "User", "Researcher who configures and runs simulations")

    Container_Boundary(c0, "Orchestrator") {
        Component(orchestrator, "Orchestrator", "Python Class", "Coordinates the entire simulation pipeline")
        Component(personaGenerator, "Persona Generator", "Python Class", "Generates personas based on given topics")
        Component(agentGenerator, "Agent Generator", "Python Class", "Generates and manages LLM-powered agents")
        Component(interactionManager, "Interaction Manager", "Python Class", "Manages interactions between agents")
        Component(promptManager, "Prompt Manager", "Python Class", "Manages prompt formatting and structure")
        Component(llmInterface, "LLM Interface", "Python Class", "Interacts with LLM APIs like OpenAI and Groq")
        Component(evaluationMetrics, "Evaluation Metrics", "Python Class", "Calculates opinion dynamics metrics")
        Component(dataStorage, "Data Storage", "File System", "Stores interaction transcripts and evaluation results")
        Component(logStorage, "Log Storage", "File System", "Stores logs")
    }

    Rel(user, orchestrator, "Configures and runs simulations")
    Rel(orchestrator, personaGenerator, "Generates personas")
    Rel(orchestrator, agentGenerator, "Initializes agents")
    Rel(orchestrator, interactionManager, "Manages interactions")
    Rel(orchestrator, promptManager, "Formats prompts")
    Rel(orchestrator, llmInterface, "Interacts with LLM APIs")
    Rel(orchestrator, evaluationMetrics, "Evaluates opinion dynamics")
    Rel(orchestrator, dataStorage, "Stores interaction transcripts and evaluation results")
    Rel(orchestrator, logStorage, "Stores logs")

    System_Ext(aiProvider, "AI Provider", "External LLM API provider like OpenAI or Groq")
    Rel(llmInterface, aiProvider, "Interacts with LLM API for generating responses")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```
### Component Diagram for Agent Generator
```mermaid
C4Component
    title Component Diagram for Agent Generator

    Container_Boundary(c1, "Agent Generator") {
        Component(agent, "Agent", "Python Class", "Generates and manages LLM-powered agents")
        Component(memoryStream, "Memory Stream", "Python Class", "Manages agent's memory stream")
    }

    Rel(agent, memoryStream, "Updates and retrieves memory stream")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Component Diagram for Persona Generator
```mermaid
C4Component
    title Component Diagram for Persona Generator

    Container_Boundary(c8, "Persona Generator") {
        Component(personaGenerator, "Persona Generator", "Python Class", "Generates personas based on given topics")
        Component(llm, "LLM", "Python Class", "Interacts with LLM APIs to generate personas")
    }

    Rel(personaGenerator, llm, "Interacts with LLM API for generating personas")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Component Diagram for Interaction Manager
```mermaid
C4Component
    title Component Diagram for Interaction Manager

    Container_Boundary(c2, "Interaction Manager") {
        Component(interactionManager, "Interaction Manager", "Python Class", "Manages interactions between agents")
        Component(agent, "Agent", "Python Class", "Interacts with other agents")
    }

    Rel(interactionManager, agent, "Manages interactions")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### Component Diagram for Prompt Manager
```mermaid
C4Component
    title Component Diagram for Prompt Manager

    Container_Boundary(c3, "Prompt Manager") {
        Component(promptManager, "Prompt Manager", "Python Class", "Manages prompt formatting and structure")
    }

    UpdateLayoutConfig($c4ShapeInRow="1", $c4BoundaryInRow="1")
```

### Component Diagram for LLM Interface
```mermaid
C4Component
    title Component Diagram for LLM Interface

    Container_Boundary(c4, "LLM Interface") {
        Component(llmInterface, "LLM Interface", "Python Class", "Interacts with LLM APIs like OpenAI and Groq")
        Component(promptManager, "Prompt Manager", "Python Class", "Formats prompts for LLM APIs")
    }

    Rel(llmInterface, promptManager, "Formats prompts")
    Rel(llmInterface, aiProvider, "Interacts with LLM API for generating responses")

    UpdateLayoutConfig($c4ShapeInRow="2",$c4BoundaryInRow="1")
```

### Component Diagram for Evaluation Metrics

```mermaid
C4Component
    title Component Diagram for Evaluation Metrics

    Container_Boundary(c5, "Evaluation Metrics") {
        Component(evaluationMetrics, "Evaluation Metrics", "Python Class", "Calculates opinion dynamics metrics")
    }

    UpdateLayoutConfig($c4ShapeInRow="1", $c4BoundaryInRow="1")
```

### Component Diagram for Data Storage

```mermaid
C4Component
    title Component Diagram for Data Storage

    Container_Boundary(c6, "Data Storage") {
        Component(dataStorage, "Data Storage", "File System", "Stores interaction transcripts and evaluation results")
    }

    UpdateLayoutConfig($c4ShapeInRow="1", $c4BoundaryInRow="1")
```

### Component Diagram for Log Storage

```mermaid
C4Component
    title Component Diagram for Log Storage

    Container_Boundary(c7, "Log Storage") {
        Component(logStorage, "Log Storage", "File System", "Stores logs")
    }

    UpdateLayoutConfig($c4ShapeInRow="1", $c4BoundaryInRow="1")
```