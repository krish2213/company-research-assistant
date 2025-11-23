# Company Research Assistant 🔍

An intelligent, conversational AI agent that researches companies and generates structured Account Plans. This project demonstrates advanced agentic behaviors, contextual understanding, and persona adaptation, built with Python and powered by Groq's LLaMA-3.1-8B model.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Agentic Flow Diagram](#agentic-flow-diagram)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Key Talking Points for Demo](#key-talking-points-for-demo)

---

## ✨ Features

### Core Capabilities
- [cite_start]**Company Research**: Fetches data from Wikipedia API with mock data fallback for reliability.
- **Account Plan Generation**: Creates structured 5-section business plans (Overview, Products, Competitors, Opportunities, Risks).
- [cite_start]**Plan Updates**: Allows users to modify any plan section through natural language input.
- [cite_start]**Progress Reporting**: Provides real-time feedback during research operations ("Researching...", "Generating plan...").

### Intelligent Agentic Behaviors
- [cite_start]**Persona Adaptation**: Adjusts communication style (tone, detail level) based on detected user behavior (Confused, Efficient, Chatty)[cite: 24, 31].
- [cite_start]**Clarification Requests**: Prompts the user when company names are ambiguous, misspelled, or resolved via a contextual alias (e.g., "the cloud company").
- **Data Gap Reporting**: Honestly reports low confidence or missing critical fields to the user.
- [cite_start]**Robust Extraction**: Uses LLM-based normalization (`company_normalizer.py`) combined with fuzzy matching to extract company names from complex input[cite: 31].

---

## 🏗️ Architecture

The project utilizes a modular, layered architecture to maintain a clean separation of concerns.

project/
├── app.py                 # CLI interface & main entry point (User I/O loop)
├── agent_logic.py         # Core agent intelligence: intent routing, LLM calls, plan generation
├── research_tools.py      # Data layer: Wikipedia API integration, mock data, normalization
├── state.py               # State management: history, conversation phase, persona signals
├── company_normalizer.py  # Pre-processing: company name extraction, fuzzy matching, alias resolution
├── utils.py               # Helper utilities: intent detection, formatting, validation helpers
└── .env                   # Environment variables (GEMINI_API_KEY or GROQ_API_KEY)

---

## 🎯 Design Decisions & Rationale

| Decision | Rationale |
| :--- | :--- |
| **1. Persona Detection (Signal-Based)** | Tracks simple behavioral signals (e.g., `confusion_count`, `direct_requests`) within `state.py` and uses heuristics in `agent_logic.py` to infer persona. This is **privacy-respecting** and adaptive without requiring explicit user profiling[cite: 31]. |
| **2. Intent Prioritization (Filtering)** | Implements **strict rule-based checks** (`utils.py`) for common words (`yes`, `no`, `help`) *before* resorting to the LLM or complex `potential_research` flow. This prevents the agent from attempting to research confused/off-topic inputs like "I don't know"[cite: 34]. |
| **3. Alias Confirmation Flow** | When an alias (e.g., "the cloud company") is detected in `company_normalizer.py`, the confidence is artificially lowered, forcing a **confirmation step**. This ensures the agent is highly accurate and conversational ("I assume you mean Amazon?") rather than silently making a potentially wrong decision (like researching Microsoft)[cite: 34]. |
| **4. Research Strategy (Fallback)** | Prioritizes fast, reliable mock data for common companies, then attempts real-time Wikipedia API search. This guarantees a **reliable demo experience** while retaining real-world data capability. |
| **5. LLM Integration (Structured JSON)** | All generation tasks (plan creation, company extraction) require output in **valid JSON**. This allows for predictable parsing, validation, and robust error handling within `agent_logic.py` and `company_normalizer.py`. |

---

## 🔄 Agentic Flow Diagram

The agent operates in a closed loop, prioritizing clarification and state resolution before execution. 
1.  **Input & State Update:** User message is received, cleaned, and conversation history/persona signals are updated.
2.  **Intent Detection:** Determine if the input is a **Command** (Show Plan, Update), a **Filter** (Help, Exit), an **Off-Topic** query, or a **Research** request.
3.  **Clarification Check (Priority 1):** If the Phase is `CLARIFYING` (due to low confidence, ambiguity, or alias match), the agent checks if the input resolves the issue (Yes/No/Selection).
4.  **Execution:** Route to the appropriate handler (`handle_research`, `handle_update`, etc.).
5.  **Persona Adaptation:** The generated response is styled (trimmed for Efficient users, verbose for Confused users).
6.  **Output & Phase Update:** Response is displayed, and the session state/phase is updated for the next turn.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Groq API key (Get one at Groq Console)

### Installation

```bash
# 1. Clone/create project directory
mkdir company-research-assistant
cd company-research-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
# NOTE: Ensure you have a requirements.txt file matching the necessary dependencies (groq, requests, python-dotenv, rapidfuzz)
pip install -r requirements.txt

# 4. Create environment file
echo "GROQ_API_KEY=your_api_key_here" > .env

# 5. Run the application
python app.py
📖 UsageInteractive ModeBashpython app.py
Commands ReferenceCommandDescriptionResearch [company]Start researching a companyShow planDisplay current Account PlanUpdate [section] with: [content]Modify a plan sectionHelpShow help informationExit / QuitEnd the session