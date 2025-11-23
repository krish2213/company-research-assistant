# Company Research Assistant 🔍

An intelligent, conversational AI agent that researches companies and generates structured Account Plans. Built with Python and powered by Groq's LLaMA-3.1-8B model, this project demonstrates agentic reasoning, contextual understanding, and adaptive personas.

---

## 📋 Table of Contents
- [Features](#features)
- [Architecture & Modularization](#architecture--modularization)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Agentic Flow Diagram](#agentic-flow-diagram)
- [Evaluation Criteria Adherence](#evaluation-criteria-adherence)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

---

## ✨ Features

### Core Capabilities
- **Company Research**: Retrieves data from the Wikipedia API with robust mock-data fallback to ensure reliability.
- **Account Plan Generation**: Produces 5-section structured business plans (Overview, Products, Competitors, Opportunities, Risks).
- **Plan Updates**: Users can naturally modify any section of the plan.
- **Progress Reporting**: Communicates real-time status (“Researching…”, “Generating plan…”).

### Intelligent Agentic Behaviors
- **Persona Adaptation**: Adjusts tone and verbosity based on user behavior (Confused, Efficient, Chatty).
- **Proactive Clarification**: Confirms ambiguous company names or contextual aliases (e.g., “the cloud company”).
- **Data Gap Reporting**: Transparently communicates missing or low-confidence fields.
- **Robust Extraction**: Uses LLM-based normalization and fuzzy matching to parse company names from complex input.

---

## 🏗️ Architecture & Modularization

A clean, layered architecture enforces strong separation of concerns:

| Module | Responsibility |
|-------|----------------|
| `app.py` | CLI interface, user I/O, top-level execution |
| `agent_logic.py` | Core agent reasoning: intents, personas, LLM interactions |
| `state.py` | Tracks conversation history, phase, persona signals |
| `utils.py` | Helper utilities for intent detection, validation, formatting |
| `company_normalizer.py` | Company name extraction, alias resolution, fuzzy matching |
| `research_tools.py` | Wikipedia API calls, data normalization, mock fallback |

---

## 🎯 Design Decisions & Rationale

### 1. Prioritizing Conversational Quality
- **Alias Confirmation Flow** ensures the agent never silently assumes ambiguous company names.
- Enhances trust by explicitly asking for clarification instead of guessing.

### 2. Unified Intent & Persona Handling
- All research requests flow through `handle_potential_research`.
- Guarantees uniform extraction, validation, and safe fallback behavior.

### 3. Lightweight & Transparent State Management
- A single dictionary-based state structure simplifies debugging and reasoning.
- Account plans use strict JSON formatting for predictable parsing and processing.

### 4. Dynamic Persona Detection
- Signal-based heuristics classify users as Confused, Efficient, or Chatty.
- The agent adapts tone accordingly with no external dependency overhead.

---

## 🔄 Agentic Flow Diagram (Conceptual)

1. **Input → State Update**  
2. **Intent Detection**  
3. **Clarification Phase Check**  
4. **Execution via Handlers**  
5. **Persona-Based Response Adaptation**  
6. **Output → State Update**

---

## 📝 Evaluation Criteria Adherence

| Criterion | Demonstrated Through |
|----------|-----------------------|
| **Conversational Quality** | Clarification flows, adaptive personas, off-topic handling |
| **Agentic Behaviour** | Progress updates, proactive clarifications, data-gap honesty |
| **Technical Implementation** | Modular architecture, JSON structures, mock fallback |
| **Intelligence & Adaptability** | Handling diverse personas and edge-case scenarios |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Groq API Key

### Installation
```bash
# Clone project
mkdir company-research-assistant
cd company-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
echo "GROQ_API_KEY=your_api_key_here" > .env

# Run the application
python app.py
