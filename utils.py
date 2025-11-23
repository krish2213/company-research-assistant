"""
utils.py - Utility Functions (Enhanced)

Helper functions for text processing, formatting, validation, and display.
FIXED: Added logic to strip quotes from update content to prevent false farewell detection.
"""

import re
import textwrap
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ============================================
# NON-COMPANY WORDS (Centralized)
# ============================================

CONFIRMATION_WORDS = {
    "yes", "no", "ok", "okay", "sure", "proceed", "continue", "go ahead",
    "fine", "alright", "right", "correct", "yep", "yup", "nope", "nah",
    "affirmative", "negative", "confirmed", "cancel", "stop", "i don't know", "i dont know"
}

GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "greetings", "howdy",
    "good morning", "good afternoon", "good evening", "morning", "evening"
}

FAREWELL_WORDS = {
    "bye", "goodbye", "thanks", "thank you", "thankyou", "thx",
    "exit", "quit", "done", "that's all", "finished", "end"
}

COMMAND_WORDS = {
    "help", "show", "display", "view", "see", "print", "plan",
    "update", "change", "modify", "edit", "reset", "clear"
}


# ============================================
# TEXT PROCESSING UTILITIES
# ============================================

def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_confirmation_response(text: str) -> bool:
    """Check if text is a confirmation/denial response."""
    text_lower = clean_text(text).lower()
    return text_lower in CONFIRMATION_WORDS


def is_numeric_selection(text: str) -> bool:
    """Check if text is a numeric selection (1, 2, 3, etc.)."""
    text_clean = clean_text(text)
    return bool(re.match(r'^\d+\.?$', text_clean))


def is_greeting(text: str) -> bool:
    """Check if text is a greeting."""
    text_lower = clean_text(text).lower()
    return text_lower in GREETING_WORDS or any(g in text_lower for g in GREETING_WORDS)


def is_farewell(text: str) -> bool:
    """Check if text is a farewell."""
    text_lower = clean_text(text).lower()
    return text_lower in FAREWELL_WORDS or any(f in text_lower for f in FAREWELL_WORDS)


def extract_company_name(text: str) -> Optional[str]:
    """
    Basic company name extraction using patterns.
    NOTE: This is now a FALLBACK - primary extraction uses company_normalizer.py
    """
    text = clean_text(text)
    text_lower = text.lower()
    
    # Never extract these as company names
    if text_lower in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS | COMMAND_WORDS:
        return None
    
    # Never extract pure numbers
    if re.match(r'^\d+\.?$', text_lower):
        return None
    
    # Never extract very short inputs
    if len(text) < 2:
        return None
    
    # Check for explicit research patterns
    patterns = [
        r"(?:research|look up|find|about|for|analyze|tell me about)\s+([A-Za-z][A-Za-z0-9\s&.'-]+?)(?:\s+(?:company|inc|corp|ltd|llc))?(?:\.|$|\?)",
        r"(?:company|organization|firm)\s+(?:called|named)?\s*([A-Za-z][A-Za-z0-9\s&.'-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = clean_text(match.group(1))
            if name.lower() not in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS:
                return name
    
    # If text is 1-3 capitalized words, might be a company name
    words = text.split()
    if 1 <= len(words) <= 3:
        # Check it's not a common word
        if text_lower not in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS | COMMAND_WORDS:
            # Has at least one capital letter or is all caps
            if text[0].isupper() or text.isupper():
                return text
    
    return None


def is_update_request(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the user is requesting to update a plan section.
    Returns (is_update, section_name, new_content).
    """
    text = clean_text(text)
    
    section_names = {
        'overview': 'company_overview',
        'company overview': 'company_overview',
        'products': 'key_products_services',
        'services': 'key_products_services',
        'products/services': 'key_products_services',
        'key products': 'key_products_services',
        'competitors': 'competitors',
        'competition': 'competitors',
        'opportunities': 'opportunities',
        'risks': 'risks',
        'risk': 'risks'
    }
    
    update_patterns = [
        r"(?:update|change|modify|edit|revise)\s+(?:the\s+)?(.+?)\s+(?:with|to|section)[:.]?\s*(.+)",
        r"(?:add|include)\s+(?:to\s+)?(?:the\s+)?(.+?)[:.]?\s*(.+)",
        r"(.+?)\s+(?:should|needs to)\s+(?:say|include|be)[:.]?\s*(.+)"
    ]
    
    for pattern in update_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section_mention = match.group(1).lower().strip()
            new_content = match.group(2).strip()
            
            # CRITICAL FIX: Strip surrounding single/double quotes from the content
            new_content = new_content.strip('"').strip("'")
            
            for key, section in section_names.items():
                if key in section_mention:
                    return True, section, new_content
    
    return False, None, None


def detect_intent(text: str) -> str:
    """
    Detect the user's intent from their message.
    IMPROVED: Better handling of confirmation words and ambiguous inputs.
    """
    text_clean = clean_text(text)
    text_lower = text_clean.lower()
    
    # Remove filler words and punctuation for intent detection
    text_stripped = re.sub(r'^(um+|uh+|er+|ah+|hmm+)[,\s]*', '', text_lower).strip()
    
    # Priority 1: Check for explicit confusion signals (HIGH PRIORITY)
    if detect_confusion_signals(text_lower) or detect_confusion_signals(text_stripped):
        # Explicitly check for "i don't know" or similar to avoid misclassifying it as research
        confusion_phrases = [r"i don'?t (know|understand|get)", r"not sure", r"help me understand"]
        if any(re.search(p, text_lower) for p in confusion_phrases):
             # Ensure a confused user asking a question (e.g. "what is this?") is NOT flagged as off_topic
             return "unclear" 


    # Priority 2: Check for confirmation/denial
    if is_confirmation_response(text_lower) or is_confirmation_response(text_stripped):
        return "confirmation"
    
    # Priority 3: Check for numeric selection
    if is_numeric_selection(text_clean):
        return "selection"
    
    # Priority 4: Check for greetings (including hesitant ones like "um, hi?")
    greeting_patterns = [
        r'^(um+|uh+|er+|ah+)?[,\s]*(hi|hello|hey|hii+|greetings)\b',
        r'^(hi|hello|hey|hii+)\b',
        r'^(um+|uh+|er+|ah+)\??$',
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return "greeting"
    
    if text_stripped in GREETING_WORDS:
        return "greeting"
    
    # Priority 5: Check for farewell
    if is_farewell(text_lower):
        return "farewell"
    
    # Priority 6: Help patterns
    if re.search(r'^help$|^what can you|^how do i|^how to', text_lower):
        return "help"
    
    # Priority 7: View plan patterns
    if re.search(r'(show|display|view|see|print)\s+(the\s+)?(plan|account plan|report)', text_lower):
        return "view_plan"
    
    # Priority 8: Update patterns
    is_update, _, _ = is_update_request(text)
    if is_update:
        return "update"
    
    # Priority 9: Explicit research patterns
    if re.search(r'^(research|look up|find|analyze|tell me about|information on|learn about)\s+', text_lower):
        return "research"
    
    # Priority 10: Off-topic detection
    off_topic_signals = [
    r'\bweather\b', r'\brain\b', r'\bsnow\b', r'\bsunny\b',

    r'\bjoke\b', r'\bfun fact\b', r'\bmake me laugh\b',
    r'\btell me something funny\b',

    r'\bstory\b', r'\btell me a story\b',
    r'\bmovie recommendation\b', r'\bsong recommendation\b',

    r'\brecipe\b', r'\bhow to cook\b', r'\bwhat should i eat\b',

    r'\bbook me a flight\b', r'\bbook a ticket\b', r'\bflight status\b',
    r'\bhotel\b', r'\bwhere is\b', r'\bdirections to\b',

    r'\bsports score\b', r'\bcricket score\b', r'\bwho won the match\b',

    r'\bhow are you\b', r'\bwho are you\b', r'\bwhat are you\b',
    r'\byour name\b', r'\bwhere are you from\b',

    r'\bfix my phone\b', r'\bcalculator\b', r'\bunit conversion\b',

    r'\bsolve this math\b', r'\bcalculate\b',
    r'\bintegral of\b', r'\bdifferentiation\b',

    r'\brelationship advice\b', r'\blove advice\b',
    r'\bgirlfriend\b', r'\bboyfriend\b', r'\bcrush\b',

    r'\bmeaning of life\b', r'\bpurpose of life\b', r'\bexistence\b',

    r'\bwho will win\b', r'\bhoroscope\b', r'\bzodiac\b'
]

    for signal in off_topic_signals:
        if re.search(signal, text_lower):
            return "off_topic"
    
    # Priority 11: Could be a company name - let the normalizer decide
    # Only if it looks like it could be a company (not common words)
    if len(text_clean) >= 2 and text_lower not in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS | COMMAND_WORDS:
        return "potential_research"
    
    return "unclear"


def detect_confusion_signals(text: str) -> bool:
    """Detect if user seems confused."""
    confusion_patterns = [
        r"i don'?t (know|understand|get)",
        r"what (do you mean|should i|is this|is that)",
        r"confused",
        r"not sure",
        r"help me understand",
        r"\?\s*\?+",
        r"huh\??",
        r"um+",
        r"uh+",
        r"i guess"
    ]
    
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in confusion_patterns)


def detect_efficiency_signals(text: str) -> bool:
    """Detect if user prefers efficiency."""
    efficiency_patterns = [
        r"^(just|only|quick)",
        r"skip",
        r"get to the point",
        r"brief",
        r"tl;?dr",
        r"fast",
        r"hurry"
    ]
    
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in efficiency_patterns)


import textwrap

def format_account_plan(plan: dict, box_padding: int = 2, wrap_width: int = 76) -> str:
    """
    Format the account plan inside a clean, flexible ASCII box.
    - Auto-wrap text
    - Auto-calc line lengths
    - Beautiful section headers
    """
    if not plan:
        return "No account plan available."

    # Collect sections
    sections = [
        ("COMPANY OVERVIEW", plan.get("company_overview")),
        ("KEY PRODUCTS / SERVICES", plan.get("key_products_services")),
        ("COMPETITORS", plan.get("competitors")),
        ("OPPORTUNITIES", plan.get("opportunities")),
        ("RISKS", plan.get("risks"))
    ]

    # Prepare content lines
    content_lines = []
    content_lines.append("ACCOUNT PLAN")
    content_lines.append(f"Generated: {plan.get('generated_at', 'N/A')}")
    
    if plan.get("last_updated"):
        content_lines.append(f"Last Updated: {plan['last_updated']}")
    content_lines.append("")  # blank line

    for title, content in sections:
        content_lines.append(f" {title} ")
        if content:
            wrapped = textwrap.fill(content, width=wrap_width)
            content_lines.extend(wrapped.split("\n"))
        else:
            content_lines.append("[Not provided]")
        content_lines.append("")  # blank line after each section

    # Calculate box width
    longest_line = max(len(line) for line in content_lines)
    box_width = longest_line + box_padding * 2 + 2  # 2 for borders

    top = "┌" + "─" * (box_width - 2) + "┐"
    bottom = "└" + "─" * (box_width - 2) + "┘"

    # Build the box
    final_output = [top]
    for line in content_lines:
        padded = line.ljust(longest_line)
        final_output.append("│" + " " * box_padding + padded + " " * box_padding + "│")
    final_output.append(bottom)

    return "\n".join(final_output)


# ============================================
# VALIDATION UTILITIES
# ============================================

def validate_company_name(name: str) -> Tuple[bool, str]:
    """
    Validate a company name.
    Returns (is_valid, error_message or cleaned_name).
    """
    if not name:
        return False, "Company name cannot be empty."
    
    name = clean_text(name)
    
    if len(name) < 2:
        return False, "Company name is too short."
    
    if len(name) > 100:
        return False, "Company name is too long."
    
    if re.match(r'^[\d\s]+$', name):
        return False, "Company name cannot be just numbers."
    
    if re.search(r'[<>{}[\]\\|`~]', name):
        return False, "Company name contains invalid characters."
    
    # Check against non-company words
    if name.lower() in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS:
        return False, f"'{name}' doesn't appear to be a company name."
    
    return True, name


def is_valid_section_name(section: str) -> bool:
    """Check if a section name is valid."""
    valid_sections = [
        'company_overview',
        'key_products_services',
        'competitors',
        'opportunities',
        'risks'
    ]
    return section.lower() in valid_sections


# ============================================
# DISPLAY UTILITIES
# ============================================

def print_agent_response(message: str, thinking: bool = False):
    """Print an agent response with formatting."""
    prefix = "🤔 " if thinking else "🤖 "
    print(f"\n{prefix}{message}\n")


def print_user_prompt():
    """Print the user input prompt."""
    print("You: ", end="")


def print_separator(char: str = "-", length: int = 60):
    """Print a separator line."""
    print(char * length)


def print_welcome():
    """Print welcome message."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         COMPANY RESEARCH ASSISTANT                           ║
║                                                              ║
║  I help you research companies and create Account Plans.     ║
║                                                              ║
║  What I can do:                                              ║
║  • Research any company using available data                 ║
║  • Generate structured Account Plans                         ║
║  • Update plan sections based on your feedback               ║
║                                                              ║
║  Commands: 'help', 'show plan', 'exit'                       ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help information."""
    print("""
┌────────────────────────────────────────────────────────────────┐
│  HOW TO USE THIS ASSISTANT                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  RESEARCH A COMPANY:                                           │
│    • "Research Microsoft"                                      │
│    • "Tell me about Apple"                                     │
│    • Just type a company name like "Tesla"                     │
│                                                                │
│  VIEW YOUR ACCOUNT PLAN:                                       │
│    • "Show plan"                                               │
│    • "Display the account plan"                                │
│                                                                │
│  UPDATE A SECTION:                                             │
│    • "Update risks with: Supply chain concerns"                │
│    • "Change competitors to: Microsoft, Google, Meta"          │
│                                                                │
│  OTHER COMMANDS:                                               │
│    • "help" - Show this help message                           │
│    • "exit" or "quit" - End the conversation                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")


def format_data_conflicts(conflicts: List[Dict]) -> str:
    """Format data conflicts for user review."""
    if not conflicts:
        return ""
    
    output = ["I found some conflicting information:"]
    for i, conflict in enumerate(conflicts, 1):
        output.append(f"  {i}. {conflict.get('description', 'Unknown conflict')}")
        if conflict.get('options'):
            for opt in conflict['options']:
                output.append(f"     - {opt}")
    
    return "\n".join(output)