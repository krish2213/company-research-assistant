"""
agent_logic.py - Core Agent Intelligence (Final Fix: Consolidating Research Logic)

FIXED:
- Removed redundant company extraction logic from handle_research_request.
- All research commands (Intent: 'research') are now redirected to handle_potential_research
  to force them through the comprehensive entity extraction and clarification check.
"""
import textwrap
import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from state import (
    ConversationPhase, UserPersona, update_state, add_message,
    get_recent_context, update_persona_signals, set_phase,
    set_plan_section, has_complete_plan
)
from research_tools import fetch_company_data, format_research_for_prompt, normalize_research_data
from utils import (
    clean_text, is_update_request, detect_intent,
    detect_confusion_signals, detect_efficiency_signals, format_account_plan,
    validate_company_name, is_confirmation_response, is_numeric_selection,
    CONFIRMATION_WORDS, GREETING_WORDS, FAREWELL_WORDS
)
from company_normalizer import (
    extract_company_with_llm, needs_confirmation, format_confirmation_message,
    resolve_contextual_reference, fuzzy_match_company, NON_COMPANY_WORDS
)
def wrap_text(text: str, width: 100) -> str:
    return textwrap.fill(text, width=width)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


# ============================================
# SYSTEM PROMPTS
# ============================================

AGENT_SYSTEM_PROMPT = """You are a professional Company Research Assistant. Your role is to help users research companies and create structured Account Plans.

Your personality traits:
- **Helpful and patient, especially with confused users (CONFUSED persona)**
- **Professional but warm and conversational (UNKNOWN persona)**
- **Concise and direct (EFFICIENT persona)**
- Proactive in offering guidance
- Honest about limitations and data gaps
- Adaptable to user communication styles

Your capabilities:
1. Research companies using available data
2. Generate structured Account Plans with 5 sections
3. Update specific sections of the plan
4. Ask clarifying questions when information is ambiguous

Behavioral guidelines:
- For CONFUSED users: Be extra patient, offer examples, guide step-by-step. Break down complex steps.
- For EFFICIENT users: Be extremely concise, use minimal pleasantries, prioritize facts.
- For CHATTY users: Gently redirect to the task while being friendly. Do not engage in off-topic conversations.
- Always report progress during research.
- Ask for clarification if company name is ambiguous.
- If a message is classified as 'unclear' or 'off_topic', do NOT try to interpret it as a company name.

Current date: {current_date}
"""

PLAN_GENERATION_PROMPT = """Based on the following company research data, generate a comprehensive Account Plan.

RESEARCH DATA:
{research_data}

Generate each section with substantive, actionable content. Even if some data is limited, use your knowledge to provide helpful insights.

REQUIRED SECTIONS:

1. COMPANY OVERVIEW: A clear summary of what the company does, its market position, history, and key facts. (2-4 sentences)

2. KEY PRODUCTS/SERVICES: Their main offerings and what makes them significant in the market.

3. COMPETITORS: Main competitors in their industry and brief notes on competitive positioning.

4. OPPORTUNITIES: Potential business opportunities for engaging with this company.

5. RISKS: Potential risks, challenges, or concerns.

IMPORTANT: Every section MUST have meaningful content. Do not leave any section empty.

Format your response as valid JSON with exactly these keys:
{{
    "company_overview": "...",
    "key_products_services": "...",
    "competitors": "...",
    "opportunities": "...",
    "risks": "..."
}}

Respond ONLY with the JSON object, no additional text."""


# ============================================
# PERSONA ADAPTATION
# ============================================

def get_persona_style(persona: str) -> Dict[str, str]:
    """Get communication style based on detected persona."""
    styles = {
        UserPersona.CONFUSED.value: {
            "tone": "patient and supportive",
            "detail_level": "high with examples",
            "pacing": "step-by-step",
            "extra_guidance": True
        },
        UserPersona.EFFICIENT.value: {
            "tone": "concise and direct",
            "detail_level": "minimal, facts only",
            "pacing": "fast",
            "extra_guidance": False
        },
        UserPersona.CHATTY.value: {
            "tone": "friendly but focused",
            "detail_level": "moderate",
            "pacing": "moderate with gentle redirects",
            "extra_guidance": False
        },
        UserPersona.UNKNOWN.value: {
            "tone": "professional and helpful",
            "detail_level": "moderate",
            "pacing": "normal",
            "extra_guidance": True
        },
        UserPersona.EDGE_CASE.value: {
            "tone": "helpful and clarifying",
            "detail_level": "moderate with validation",
            "pacing": "careful",
            "extra_guidance": True
        }
    }
    return styles.get(persona, styles[UserPersona.UNKNOWN.value])


def adapt_response(response: str, persona: str, state: Dict) -> str:
    """Adapt response based on user persona."""
    
    if persona == UserPersona.EFFICIENT.value:
        # Trim filler words aggressively
        response = re.sub(r'^(Sure!|Of course!|Absolutely!|Great question!|Hello!|Hi there!|I\'d be happy to|Let me)[\s\.,]*', '', response).strip()
        response = response[0].upper() + response[1:] if response else ""
        
    
    if persona == UserPersona.CONFUSED.value:
        # Add supportive closing line if the response isn't a question or an error message
        if not re.search(r'[\?!\n]$', response.strip()) and not response.lower().startswith("❌") and state.get("phase") != ConversationPhase.PLAN_READY.value:
            response += "\n\nRemember, you can ask me to explain anything further!"
    
    return response


# ============================================
# LLM INTERACTION
# ============================================

def call_llm(messages: List[Dict], temperature: float = 0.7) -> str:
    """Make a call to the Groq API."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        # Graceful error message for LLM failures
        return f"❌ I encountered an error communicating with my AI backend: {str(e)}"


def generate_contextual_response(user_message: str, state: Dict, additional_context: str = "") -> str:
    """Generate a contextual response using the LLM."""
    recent_messages = get_recent_context(state, n_messages=6)
    
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    style = get_persona_style(persona)
    
    system_prompt += f"""

Current conversation context:
- Phase: {state.get('phase')}
- Target Company: {state.get('target_company', 'Not set')}
- User Style: {persona} - Use {style['tone']} tone with {style['detail_level']} detail.
- Plan Status: {'Complete' if has_complete_plan(state) else 'Incomplete or not started'}

{additional_context}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in recent_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    
    return call_llm(messages)


# ============================================
# PLAN GENERATION
# ============================================

def generate_account_plan(state: Dict, research_result) -> Tuple[Dict, str]:
    """Generate an Account Plan from research data."""
    research_formatted = format_research_for_prompt(research_result)
    data = research_result.data or {}
    
    messages = [
        {"role": "system", "content": "You are an expert business analyst. Generate structured account plans in JSON format only."},
        {"role": "user", "content": PLAN_GENERATION_PROMPT.format(research_data=research_formatted)}
    ]
    
    try:
        response = call_llm(messages, temperature=0.5)
        
        json_match = response.strip()
        if "```json" in json_match:
            json_match = json_match.split("```json")[1].split("```")[0]
        elif "```" in json_match:
            json_match = json_match.split("```")[1].split("```")[0]
        
        json_match = json_match.strip()
        plan_data = json.loads(json_match)
        
        required = ["company_overview", "key_products_services", "competitors", "opportunities", "risks"]
        for section in required:
            if not plan_data.get(section) or plan_data[section].strip() == "":
                plan_data[section] = generate_fallback_section(section, data)
        
        plan_data["generated_at"] = datetime.now().isoformat()
        plan_data["last_updated"] = None
        plan_data["update_history"] = []
        
        return plan_data, "success"
        
    except (json.JSONDecodeError, IndexError, AttributeError):
        return generate_fallback_plan(data, research_result.company_name), "partial"


def generate_fallback_section(section: str, data: Dict) -> str:
    """Generate fallback content for a section."""
    company_name = data.get("name", "This company")
    industry = data.get("industry", "their industry")
    
    fallbacks = {
        "company_overview": data.get("description", f"{company_name} is a company operating in {industry}."),
        "key_products_services": ", ".join(data.get("products", []) + data.get("services", [])) or f"{company_name} offers various products and services in {industry}.",
        "competitors": ", ".join(data.get("competitors", [])) or f"Key competitors include other major players in {industry}.",
        "opportunities": f"Opportunities include digital transformation initiatives, market expansion, strategic partnerships, and leveraging emerging technologies in {industry}.",
        "risks": f"Key risks include competitive pressure, market volatility, regulatory changes, technology disruption, and talent acquisition challenges in {industry}."
    }
    return fallbacks.get(section, "Information to be added.")


def generate_fallback_plan(data: Dict, company_name: str) -> Dict:
    """Generate a complete fallback plan when LLM fails."""
    industry = data.get("industry", "their industry")
    
    return {
        "company_overview": data.get("description", f"{company_name} is a company in {industry}."),
        "key_products_services": ", ".join(data.get("products", []) + data.get("services", [])) or "Products and services information to be updated.",
        "competitors": ", ".join(data.get("competitors", [])) or "Competitor information to be updated.",
        "opportunities": f"Potential opportunities for {company_name} include: digital transformation initiatives, expansion into new markets, strategic technology partnerships, and innovation in {industry}.",
        "risks": f"Key risks for {company_name} include: competitive pressure from established and emerging players, regulatory and compliance challenges, market volatility, cybersecurity threats, and talent retention in {industry}.",
        "generated_at": datetime.now().isoformat(),
        "last_updated": None,
        "update_history": []
    }


def update_plan_section(state: Dict, section: str, new_content: str) -> Tuple[bool, str]:
    """Update a specific section of the Account Plan."""
    valid_sections = {
        "company_overview": "Company Overview",
        "key_products_services": "Key Products/Services",
        "competitors": "Competitors",
        "opportunities": "Opportunities",
        "risks": "Risks"
    }
    
    section_key = section.lower().replace(" ", "_").replace("/", "_")
    
    if section_key not in valid_sections:
        return False, f"Unknown section: **{section}**. Valid sections are: {', '.join(valid_sections.values())}"
    
    if not state.get("account_plan") or not state["account_plan"].get("company_overview"):
        return False, "No Account Plan exists yet. Please research a company first by saying 'Research [Company Name]'."
    
    old_content = state["account_plan"].get(section_key, "")
    set_plan_section(state, section_key, new_content)
    
    return True, f"Updated **{valid_sections[section_key]}** section successfully."


# ============================================
# INTENT HANDLERS
# ============================================

def handle_research_request(user_message: str, state: Dict, normalized_company: str = None) -> Tuple[str, Dict]:
    """
    Handles research when the intent is explicitly 'research'.
    This function should only be called *after* confirmation/disambiguation,
    or if a simple, high-confidence name was passed.
    """
    
    # CRITICAL FIX: If we are called without a pre-normalized company,
    # it means the agent's initial intent routing was 'research'.
    # We must now call handle_potential_research to get the confirmation logic.
    if normalized_company is None:
        # Redirect to potential research logic for full extraction and confirmation check.
        return handle_potential_research(user_message, state)

    
    # --- START of direct research execution (runs after confirmation) ---
    company_name = normalized_company
    
    # Validate company name
    is_valid, result = validate_company_name(company_name)
    if not is_valid:
        return (
            f"❌ I had trouble with that company name: **{result}**\nCould you please provide a valid company name?",
            state
        )
    
    company_name = result
    state["target_company"] = company_name
    set_phase(state, ConversationPhase.RESEARCHING)
    
    progress_msg = f"🔍 Researching **{company_name}**...\n"
    
    research_result = fetch_company_data(company_name)
    
    if not research_result.success:
        progress_msg += f"\n⚠️ I found limited information about **{company_name}**. "
        progress_msg += "Would you like me to proceed with what I found, or would you like to try a different company name?"
        set_phase(state, ConversationPhase.CLARIFYING)
        state["research_data"]["raw_data"] = research_result.data
        state["research_data"]["confidence_score"] = research_result.confidence
        state["pending_clarification"] = {"type": "low_confidence", "company": company_name}
        return progress_msg, state
    
    # Check for conflicts (multiple matches)
    if research_result.conflicts:
        conflict_msg = f"\n⚠️ I found multiple matches for '*{company_name}*':\n"
        for conflict in research_result.conflicts:
            if conflict["type"] == "ambiguous_name":
                for i, option in enumerate(conflict["options"][:5], 1):
                    conflict_msg += f"  {i}. {option}\n"
        conflict_msg += "\nWhich company did you mean? (Enter the number or type the name)"
        
        state["pending_clarification"] = {
            "type": "company_disambiguation",
            "options": research_result.conflicts[0].get("options", [])[:5]
        }
        set_phase(state, ConversationPhase.CLARIFYING)
        return progress_msg + conflict_msg, state
    
    # Store research data
    state["research_data"]["raw_data"] = normalize_research_data(research_result.data)
    state["research_data"]["confidence_score"] = research_result.confidence
    state["research_data"]["sources"] = research_result.sources
    state["research_data"]["data_gaps"] = research_result.gaps
    
    progress_msg += f"✅ Found information about **{research_result.company_name}**.\n"
    progress_msg += f"📊 Data confidence: {research_result.confidence:.0%}\n"
    
    if research_result.gaps:
        progress_msg += f"📝 Note: Limited data for: {', '.join(research_result.gaps)}\n"
    
    progress_msg += "\nGenerating Account Plan...\n"
    
    plan, status = generate_account_plan(state, research_result)
    state["account_plan"] = plan
    set_phase(state, ConversationPhase.PLAN_READY)
    
    if status == "partial":
        progress_msg += "⚠️ Generated a basic plan. Some sections may need manual enrichment.\n"
    else:
        progress_msg += "✅ Account Plan generated successfully!\n"
    
    # Adapt the plan summary presentation for different personas
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    plan_summary = format_account_plan(state["account_plan"])
    
    if persona == UserPersona.EFFICIENT.value:
        progress_msg += "Plan Ready. Use 'Show plan' to view the full details."
    elif persona == UserPersona.CONFUSED.value:
        progress_msg += f"Here is the plan. Take a look at the **Company Overview** and **Key Products/Services** sections below to get started.\n\n{plan_summary}"
        progress_msg += "\n\nWhat section would you like to review or update next? (e.g., 'Update risks with...')"
    else: # UNKNOWN/CHATTY
        progress_msg += f"Here is the full Account Plan:\n\n{plan_summary}"
        progress_msg += "\n\nYou can update any section by saying something like:\n"
        progress_msg += "'Update risks with: Supply chain vulnerabilities due to global dependencies'"
    
    return progress_msg, state


def handle_confirmation_response(user_message: str, state: Dict) -> Tuple[bool, str, Dict]:
    """Handle yes/no confirmation responses."""
    pending = state.get("pending_clarification")
    if not pending:
        return False, "", state
    
    msg_lower = user_message.lower().strip()
    
    # Check for positive confirmation
    positive_words = {"yes", "y", "yep", "yup", "sure", "ok", "okay", "correct", "right", "proceed", "go ahead", "continue"}
    negative_words = {"no", "n", "nope", "nah", "wrong", "incorrect", "cancel", "stop"}
    
    if pending.get("type") == "company_confirmation":
        company = pending.get("company")
        
        if msg_lower in positive_words:
            state["pending_clarification"] = None
            # Now we call handle_research_request with the confirmed name
            response, new_state = handle_research_request("", state, normalized_company=company)
            return True, response, new_state
        
        if msg_lower in negative_words:
            state["pending_clarification"] = None
            set_phase(state, ConversationPhase.GATHERING_COMPANY)
            
            # IMPROVEMENT: Check if user provided a new company name in the same turn (e.g., "No, I meant Deloitte")
            extraction = extract_company_with_llm(user_message, "\n".join([m.get("content", "") for m in get_recent_context(state, 3)]))
            if extraction.get("is_company_query") and extraction.get("extracted_company") and extraction["extracted_company"] != company:
                new_company = extraction["extracted_company"]
                return True, f"Understood. Let me research **{new_company}** for you.", handle_research_request("", state, normalized_company=new_company)[1]

            return True, "No problem! Please tell me the correct company name you'd like to research.", state
    
    if pending.get("type") == "low_confidence":
        company = pending.get("company")
        
        if msg_lower in positive_words:
            state["pending_clarification"] = None
            # Force proceed with limited data
            response, new_state = handle_direct_research(company, state)
            return True, response, new_state
        
        if msg_lower in negative_words:
            state["pending_clarification"] = None
            set_phase(state, ConversationPhase.GATHERING_COMPANY)
            return True, "No problem! Please provide a different company name to research.", state
    
    return False, "", state


def handle_disambiguation_response(user_message: str, state: Dict) -> Tuple[bool, str, Dict]:
    """Handle user response to a disambiguation prompt."""
    pending = state.get("pending_clarification")
    if not pending or pending.get("type") != "company_disambiguation":
        return False, "", state
    
    options = pending.get("options", [])
    if not options:
        return False, "", state
    
    user_input = user_message.strip().lower()
    selected_company = None
    
    # Check for numeric selection
    number_match = re.search(r'^(\d+)\.?$|^option\s*(\d+)$|^(\d+)\s*[-:.)]', user_input)
    if number_match:
        num_str = number_match.group(1) or number_match.group(2) or number_match.group(3)
        try:
            idx = int(num_str) - 1
            if 0 <= idx < len(options):
                selected_company = options[idx]
        except ValueError:
            pass
    
    # Check for direct name match
    if not selected_company:
        for opt in options:
            # Check for case-insensitive exact match or close match
            if opt.lower() == user_input or opt.lower() in user_input or user_input in opt.lower():
                selected_company = opt
                break
    
    # Check for ordinals
    if not selected_company:
        ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "1st": 0, "2nd": 1, "3rd": 2}
        for word, idx in ordinals.items():
            if word in user_input and idx < len(options):
                selected_company = options[idx]
                break
    
    if selected_company:
        state["pending_clarification"] = None
        state["target_company"] = selected_company
        # FIX: Ensure handle_direct_research's return is unpacked correctly.
        response, new_state = handle_direct_research(selected_company, state)
        return True, response, new_state
    
    return False, "", state


def handle_direct_research(company_name: str, state: Dict) -> Tuple[str, Dict]:
    """
    Handles research execution for confirmed/unambiguous company names.
    This bypasses the extraction/confirmation logic.
    """
    state["target_company"] = company_name
    set_phase(state, ConversationPhase.RESEARCHING)
    
    progress_msg = f"🔍 Researching **{company_name}**...\n"
    
    research_result = fetch_company_data(company_name)
    
    if not research_result.success:
        progress_msg += f"\n⚠️ I found limited information about **{company_name}**. "
        progress_msg += "I'll generate a basic plan with what I found, but you should review and update sections manually."
    
    state["research_data"]["raw_data"] = normalize_research_data(research_result.data)
    state["research_data"]["confidence_score"] = research_result.confidence
    state["research_data"]["sources"] = research_result.sources
    state["research_data"]["data_gaps"] = research_result.gaps
    
    if research_result.success:
        progress_msg += f"✅ Found information about **{research_result.company_name}**.\n"
        progress_msg += f"📊 Data confidence: {research_result.confidence:.0%}\n"
    
    if research_result.gaps:
        progress_msg += f"📝 Note: Limited data for: {', '.join(research_result.gaps)}\n"
    
    progress_msg += "\nGenerating Account Plan...\n"
    
    plan, status = generate_account_plan(state, research_result)
    state["account_plan"] = plan
    set_phase(state, ConversationPhase.PLAN_READY)
    
    if status == "partial":
        progress_msg += "⚠️ Generated a basic plan. Some sections may need manual enrichment.\n"
    else:
        progress_msg += "✅ Account Plan generated successfully!\n"
    
    # Adapt the plan summary presentation for different personas
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    plan_summary = format_account_plan(state["account_plan"])
    
    if persona == UserPersona.EFFICIENT.value:
        progress_msg += "Plan Ready. Use 'Show plan' to view the full details."
    elif persona == UserPersona.CONFUSED.value:
        progress_msg += f"Here is the plan. Take a look at the **Company Overview** and **Key Products/Services** sections below to get started.\n\n{plan_summary}"
        progress_msg += "\n\nWhat section would you like to review or update next? (e.g., 'Update risks with...')"
    else: # UNKNOWN/CHATTY
        progress_msg += f"Here is the full Account Plan:\n\n{plan_summary}"
        progress_msg += "\n\nYou can update any section by saying something like:\n"
        progress_msg += "'Update risks with: Supply chain vulnerabilities due to global dependencies'"
    
    return progress_msg, state


def handle_update_request(user_message: str, state: Dict) -> Tuple[str, Dict]:
    """Handle a request to update a plan section."""
    is_update, section, new_content = is_update_request(user_message)
    
    if not is_update or not section or not new_content:
        return (
            "I'm happy to update the Account Plan! Please specify which section to update and what content to add.\n"
            "For example: 'Update risks with: Regulatory compliance concerns in European markets'",
            state
        )
    if detect_confusion_signals(new_content):
        update_persona_signals(state, "confusion_count")
        return (
            f"I see you're still unsure. I can't update the **{section}** section with content that suggests confusion.\n"
            "Could you try rephrasing the content you want to add?",
            state
        )

    
    success, message = update_plan_section(state, section, new_content)
    
    if success:
        response = f"✅ {message}\n\nHere's the updated plan:\n\n"
        # For efficiency, only show the updated section for efficient users
        if state.get("detected_persona") == UserPersona.EFFICIENT.value:
            updated_section_content = state["account_plan"].get(section)
            response = f"✅ {message}\n\n**Updated Content:**\n{updated_section_content}\n\nUse 'Show plan' to see the full document."
        else:
            response += format_account_plan(state["account_plan"])
        
        return response, state
    else:
        return f"❌ {message}", state


def handle_view_plan(state: Dict) -> Tuple[str, Dict]:
    """Handle request to view the current plan."""
    if not state.get("account_plan") or not state["account_plan"].get("company_overview"):
        return (
            "You don't have an Account Plan yet. Would you like to research a company for you?\n"
            "Just tell me the company name!",
            state
        )
    
    return format_account_plan(state["account_plan"]), state


def handle_help_request(state: Dict) -> Tuple[str, Dict]:
    """Handle help requests."""
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    
    if persona == UserPersona.EFFICIENT.value:
        return "Commands: Research [company], Show plan, Update [section] with: [content], Exit", state
    
    help_text = """
Here's how I can help you:

📊 **Research a Company**
   Just tell me a company name! Examples:
   - "Research Microsoft"
   - "Tell me about Tesla"
   - "Apple"

📋 **View Your Account Plan**
   - "Show plan"
   - "Display account plan"

✏️ **Update Plan Sections**
   - "Update risks with: [your content]"
   - "Change competitors to: [new list]"
   
   Sections you can update:
   • Company Overview
   • Key Products/Services
   • Competitors
   • Opportunities
   • Risks

🚪 **Exit**
   - "exit" or "quit"

What would you like to do?
"""
    return help_text, state


def handle_greeting(state: Dict) -> Tuple[str, Dict]:
    """Handle greeting messages, including hesitant ones."""
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    
    # Check if this seems like a confused/hesitant user
    if persona == UserPersona.CONFUSED.value:
        return (
            "Hello! 👋 No worries if you're not sure where to start - I'm here to guide you, step-by-step.\n\n"
            "I'm your Company Research Assistant. Just tell me a company name like 'Microsoft' or 'Apple' and I'll start researching!\n\n"
            "What company is on your mind?",
            set_phase(state, ConversationPhase.GATHERING_COMPANY)
        )
    
    if persona == UserPersona.EFFICIENT.value:
        return "Hello! Which company would you like to research today?", state
    
    if state.get("target_company"):
        return (
            f"Hello again! We were working on research for **{state['target_company']}**. "
            "Would you like to continue, or research a different company?",
            state
        )
    
    return (
        "Hello! 👋 I'm your Company Research Assistant. I help you research companies and create "
        "detailed Account Plans.\n\nWhich company would you like me to research today?",
        set_phase(state, ConversationPhase.GATHERING_COMPANY)
    )


def handle_farewell(state: Dict) -> Tuple[str, Dict]:
    """Handle farewell messages."""
    if has_complete_plan(state):
        return (
            f"Goodbye! Your Account Plan for **{state.get('target_company', 'the company')}** is ready. "
            "Feel free to come back anytime if you need updates or want to research another company!",
            state
        )
    
    return "Goodbye! Feel free to return whenever you need company research assistance.", state


def handle_off_topic(user_message: str, state: Dict) -> Tuple[str, Dict]:
    """Handle off-topic messages with gentle redirection."""
    update_persona_signals(state, "off_topic_count")
    
    persona = state.get("detected_persona", UserPersona.UNKNOWN.value)
    
    if persona == UserPersona.EFFICIENT.value:
        return "I'm focused on company research. Which company should I research?", state
    
    redirect_responses = [
        "That's an interesting topic! However, I specialize in company research. "
        "Is there a company you'd like me to help you research today?",
        
        "I appreciate the conversation! My expertise is in researching companies and creating Account Plans. "
        "Would you like to explore information about a specific company?",
        
        "I'd love to help with that, but my specialty is company research. "
        "If you have a company in mind you'd like to learn about, I'm your assistant!"
    ]
    
    idx = state.get("message_count", 0) % len(redirect_responses)
    return redirect_responses[idx], state


def handle_unclear(user_message: str, state: Dict) -> Tuple[str, Dict]:
    """Handle unclear or ambiguous messages."""
    phase = state.get("phase")
    
    # PRIORITY 1: Check for pending clarifications (should have been handled in agent() but is a robust fallback)
    if phase == ConversationPhase.CLARIFYING.value:
        # Check if the message can resolve a pending clarification
        was_handled, response, state = handle_confirmation_response(user_message, state)
        if was_handled: return response, state
        
        was_handled, response, state = handle_disambiguation_response(user_message, state)
        if was_handled: return response, state
        
        # If still pending, repeat the last question
        pending = state.get("pending_clarification")
        if pending and pending.get("type") == "company_disambiguation":
            options = pending.get("options", [])
            return (
                f"I'm sorry, I didn't catch that. Please enter the number (1-{len(options)}) or type the company name to continue.",
                state
            )


    # PRIORITY 2: Use persona signals to adapt response and *increase* confusion count
    update_persona_signals(state, "confusion_count")
    
    # PRIORITY 3: Try contextual resolution (did they mean the last company?)
    history = state.get("conversation_history", [])
    resolved = resolve_contextual_reference(user_message, history)
    if resolved:
        return handle_research_request("", state, normalized_company=resolved)
    
    # PRIORITY 4: Use LLM for a helpful, contextual answer
    if phase == ConversationPhase.GATHERING_COMPANY.value:
        return (
            "I'm not sure I understood. Are you trying to tell me a company name to research?\n"
            "You can simply type the company name, like **'Microsoft'** or **'Tesla'**.\n\n"
            "I can also understand descriptions like **'the search engine company'** or **'the iPhone maker'**.",
            state
        )
    
    # Fallback: General request for clarification using LLM for a natural response
    return generate_contextual_response(
        f"I'm not sure what you mean by: '{user_message}'. Can you please clarify?",
        state,
        additional_context="The user's message was unclear. Ask for clarification politely and offer concrete next steps, referring to their current phase/goal."
    ), state


# ============================================
# MAIN AGENT FUNCTION
# ============================================

def agent(user_message: str, state: Dict) -> Tuple[str, Dict]:
    """
    Main agent function that processes user input and returns a response.
    """
    # Clean input
    user_message = clean_text(user_message)
    
    if not user_message:
        return "I didn't receive any input. How can I help you today?", state
    
    # Add user message to history
    add_message(state, "user", user_message)
    
    # Detect user signals for persona adaptation
    if detect_confusion_signals(user_message):
        update_persona_signals(state, "confusion_count")
    if detect_efficiency_signals(user_message):
        update_persona_signals(state, "direct_requests")
    
    # Detect hesitant/uncertain language as confusion signal
    msg_lower = user_message.lower()
    if re.search(r'^(um+|uh+|er+|ah+|hmm+)', msg_lower) or '?' in user_message:
        update_persona_signals(state, "confusion_count")
    
    # PRIORITY 2: Detect intent
    intent = detect_intent(user_message)
    
    # PRIORITY 1: Handle pending clarifications (confirmation, disambiguation)
    # This must be run before the standard intent handlers to process "yes"/"no"/"1"
    phase = state.get("phase")
    if phase == ConversationPhase.CLARIFYING.value:
        pending = state.get("pending_clarification")
        if pending:
            # Check if it's a confirmation/selection response
            if intent in {"confirmation", "selection"} or pending.get("type") == "company_disambiguation":
                
                was_handled, response, new_state = handle_confirmation_response(user_message, state)
                if was_handled:
                    persona = new_state.get("detected_persona", UserPersona.UNKNOWN.value)
                    response = adapt_response(response, persona, new_state)
                    #response = wrap_text(response, width=76)
                    add_message(new_state, "assistant", response)
                    return response, new_state
                
                was_handled, response, new_state = handle_disambiguation_response(user_message, state)
                if was_handled:
                    persona = new_state.get("detected_persona", UserPersona.UNKNOWN.value)
                    response = adapt_response(response, persona, new_state)
                    #response = wrap_text(response, width=76)
                    add_message(new_state, "assistant", response)
                    return response, new_state
    
    # PRIORITY 3: Handle based on intent (Now guaranteed to be defined)
    
    # If a confirmation/selection word is received but not in clarifying phase, treat it as unclear
    if intent in {"confirmation", "selection"}:
        return handle_unclear(user_message, state)
    
    # Standard intent handlers
    handlers = {
        "greeting": lambda: handle_greeting(state),
        "farewell": lambda: handle_farewell(state),
        "help": lambda: handle_help_request(state),
        "view_plan": lambda: handle_view_plan(state),
        "update": lambda: handle_update_request(user_message, state),
        # Intent 'research' and 'potential_research' now both map to the same logic:
        "research": lambda: handle_potential_research(user_message, state), 
        "potential_research": lambda: handle_potential_research(user_message, state),
        "off_topic": lambda: handle_off_topic(user_message, state),
        "unclear": lambda: handle_unclear(user_message, state)
    }
    
    handler = handlers.get(intent, handlers["unclear"])
    
    # FIX: Ensure all handlers return (response, state)
    response, new_state = handler()
    
    # Adapt response based on persona
    persona = new_state.get("detected_persona", UserPersona.UNKNOWN.value)
    response = adapt_response(response, persona, new_state)
    #response = wrap_text(response, width=76)
    # Add agent response to history
    add_message(new_state, "assistant", response)
    
    return response, new_state


def handle_potential_research(user_message: str, state: Dict) -> Tuple[str, Dict]:
    """
    Unified function to extract company names, check for confirmation needs (alias/fuzzy match), 
    and then either enter clarification phase or proceed to research execution.
    """
    msg_lower = user_message.lower().strip()
    
    # CRITICAL FIX: If the message contains explicit confusion signals, route to handle_unclear
    if detect_confusion_signals(user_message):
        return handle_unclear(user_message, state)
    
    # Strip filler words to check underlying intent
    msg_stripped = re.sub(r'^(um+|uh+|er+|ah+|hmm+)[,\s]*', '', msg_lower).strip()
    
    # Don't treat common non-company words as companies
    is_non_company = msg_lower in NON_COMPANY_WORDS or msg_stripped in NON_COMPANY_WORDS or \
                     msg_lower in CONFIRMATION_WORDS | GREETING_WORDS | FAREWELL_WORDS
    
    if is_non_company:
        return handle_unclear(user_message, state)
    
    # IMPROVEMENT: Pre-clean the message to remove 'research' command words
    search_terms = re.sub(
        r'^(research|look up|find|analyze|tell me about|information on|learn about)\s+', 
        '', 
        user_message.lower(), 
        count=1
    ).strip()
    
    # Use LLM-based extraction (where the alias check happens)
    context = "\n".join([m.get("content", "") for m in get_recent_context(state, 3)])
    extraction = extract_company_with_llm(search_terms, context)
    
    if extraction.get("is_company_query") and extraction.get("extracted_company"):
        
        is_alias = extraction.get("is_alias_match", False)
        
        # Check if we need confirmation for misspellings, low confidence, OR ALIAS MATCH
        if needs_confirmation(extraction) or is_alias:
            
            # Pass alias info to state for a cleaner flow
            state["pending_clarification"] = {
                "type": "company_confirmation",
                "company": extraction["extracted_company"],
                "original_input": user_message,
                "confidence": extraction.get("confidence", 0),
                "is_alias": is_alias # Flag set here for alias match
            }
            set_phase(state, ConversationPhase.CLARIFYING)
            return format_confirmation_message(extraction), state
        
        # If confidence is high AND no clarification needed, proceed directly to execution
        return handle_direct_research(extraction["extracted_company"], state)
    
    # Try fuzzy matching as fallback (only for non-research commands)
    if not user_message.lower().startswith("research"):
        fuzzy_result = fuzzy_match_company(user_message, threshold=75)
        if fuzzy_result:
            matched, score = fuzzy_result
            if score >= 85:
                return handle_direct_research(matched, state)
            else:
                # Ask for confirmation
                state["pending_clarification"] = {
                    "type": "company_confirmation",
                    "company": matched,
                    "original_input": user_message,
                    "confidence": score / 100.0,
                    "is_alias": False 
                }
                set_phase(state, ConversationPhase.CLARIFYING)
                return (
                    f"Did you mean **{matched}**?\n"
                    "Please reply 'yes' to confirm or provide the correct company name.",
                    state
                )

    # Not recognized as a company
    # Final Fallback: If it's more than one word, assume it's a non-mock company name and research it (low confidence expected)
    if len(user_message.split()) > 1 and len(user_message) > 5 and not is_non_company:
        return handle_direct_research(user_message, state)
    
    # Truly unclear/single word input that wasn't matched
    return handle_unclear(user_message, state)