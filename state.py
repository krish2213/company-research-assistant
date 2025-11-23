"""
state.py - Conversation State Management

This module manages the conversation state using a simple Python dictionary.
It tracks conversation history, user context, research data, and the account plan.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class ConversationPhase(Enum):
    """Tracks where we are in the conversation flow."""
    GREETING = "greeting"
    GATHERING_COMPANY = "gathering_company"
    RESEARCHING = "researching"
    CLARIFYING = "clarifying"
    GENERATING_PLAN = "generating_plan"
    PLAN_READY = "plan_ready"
    UPDATING_PLAN = "updating_plan"
    IDLE = "idle"


class UserPersona(Enum):
    """Detected user interaction style."""
    UNKNOWN = "unknown"
    CONFUSED = "confused"
    EFFICIENT = "efficient"
    CHATTY = "chatty"
    EDGE_CASE = "edge_case"


def create_initial_state() -> Dict[str, Any]:
    """Create a fresh conversation state dictionary."""
    return {
        # Session metadata
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat(),
        
        # Conversation tracking
        "phase": ConversationPhase.GREETING.value,
        "message_count": 0,
        "conversation_history": [],  # List of {"role": str, "content": str}
        
        # User understanding
        "detected_persona": UserPersona.UNKNOWN.value,
        "persona_signals": {
            "confusion_count": 0,
            "off_topic_count": 0,
            "direct_requests": 0,
            "clarification_requests": 0
        },
        
        # Research context
        "target_company": None,
        "company_variants": [],  # Alternative names/spellings found
        "research_data": {
            "raw_data": None,
            "confidence_score": 0.0,
            "data_gaps": [],
            "conflicts": [],
            "sources": []
        },
        
        # Account Plan
        "account_plan": {
            "company_overview": None,
            "key_products_services": None,
            "competitors": None,
            "opportunities": None,
            "risks": None,
            "generated_at": None,
            "last_updated": None,
            "update_history": []
        },
        
        # Pending actions
        "pending_clarification": None,
        "suggested_actions": [],
        
        # Error tracking
        "errors": []
    }


def update_state(state: Dict, updates: Dict) -> Dict:
    """Update state with new values, preserving structure."""
    def deep_update(base: Dict, updates: Dict) -> Dict:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    state["last_activity"] = datetime.now().isoformat()
    return deep_update(state, updates)


def add_message(state: Dict, role: str, content: str) -> Dict:
    """Add a message to conversation history."""
    state["conversation_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    state["message_count"] += 1
    state["last_activity"] = datetime.now().isoformat()
    return state


def get_recent_context(state: Dict, n_messages: int = 10) -> List[Dict]:
    """Get the most recent n messages for context."""
    return state["conversation_history"][-n_messages:]


def update_persona_signals(state: Dict, signal_type: str) -> Dict:
    """Update persona detection signals."""
    if signal_type in state["persona_signals"]:
        state["persona_signals"][signal_type] += 1
    
    # Re-evaluate persona based on signals
    signals = state["persona_signals"]
    
    if signals["confusion_count"] >= 2:
        state["detected_persona"] = UserPersona.CONFUSED.value
    elif signals["off_topic_count"] >= 2:
        state["detected_persona"] = UserPersona.CHATTY.value
    elif signals["direct_requests"] >= 2 and signals["clarification_requests"] == 0:
        state["detected_persona"] = UserPersona.EFFICIENT.value
    
    return state


def set_phase(state: Dict, phase: ConversationPhase) -> Dict:
    """Update the conversation phase."""
    state["phase"] = phase.value
    return state


def add_plan_update(state: Dict, section: str, old_content: str, new_content: str) -> Dict:
    """Track updates to the account plan."""
    state["account_plan"]["update_history"].append({
        "section": section,
        "old_content": old_content,
        "new_content": new_content,
        "updated_at": datetime.now().isoformat()
    })
    state["account_plan"]["last_updated"] = datetime.now().isoformat()
    return state


def get_plan_section(state: Dict, section: str) -> Optional[str]:
    """Get a specific section from the account plan."""
    section_key = section.lower().replace(" ", "_").replace("/", "_")
    return state["account_plan"].get(section_key)


def set_plan_section(state: Dict, section: str, content: str) -> Dict:
    """Set a specific section in the account plan."""
    section_key = section.lower().replace(" ", "_").replace("/", "_")
    
    # Track the update if there was previous content
    old_content = state["account_plan"].get(section_key)
    if old_content:
        add_plan_update(state, section_key, old_content, content)
    
    state["account_plan"][section_key] = content
    return state


def clear_research_data(state: Dict) -> Dict:
    """Clear research data for a new company search."""
    state["research_data"] = {
        "raw_data": None,
        "confidence_score": 0.0,
        "data_gaps": [],
        "conflicts": [],
        "sources": []
    }
    state["target_company"] = None
    state["company_variants"] = []
    return state


def has_complete_plan(state: Dict) -> bool:
    """Check if all plan sections are filled."""
    plan = state["account_plan"]
    required_sections = [
        "company_overview",
        "key_products_services", 
        "competitors",
        "opportunities",
        "risks"
    ]
    return all(plan.get(section) is not None for section in required_sections)


def get_state_summary(state: Dict) -> str:
    """Get a human-readable summary of current state."""
    return f"""
Session: {state['session_id']}
Phase: {state['phase']}
Messages: {state['message_count']}
Target Company: {state['target_company'] or 'Not set'}
Persona: {state['detected_persona']}
Plan Complete: {has_complete_plan(state)}
""".strip()