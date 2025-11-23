#!/usr/bin/env python3
"""
app.py - Company Research Assistant CLI Application

This is the main entry point for the Company Research Assistant.
It provides a chat-based command-line interface for users to interact
with the research agent.

Usage:
    python app.py
    
Environment Variables:
    GROQ_API_KEY: Your Groq API key for LLaMA-3.1-70B access
"""

import os
import sys
import signal
from datetime import datetime
from dotenv import load_dotenv

# Import local modules
from state import create_initial_state, get_state_summary, ConversationPhase
from agent_logic import agent
from utils import print_welcome, print_help, print_separator, clean_text

# Load environment variables
load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Application configuration."""
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MAX_MESSAGE_LENGTH = 2000
    EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "q"}
    HELP_COMMANDS = {"help", "?", "h"}


# ============================================
# CLI INTERFACE
# ============================================

class CLIInterface:
    """Command-line interface for the Research Assistant."""
    
    def __init__(self):
        self.state = create_initial_state()
        self.running = True
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n👋 Session interrupted. Goodbye!")
        self.running = False
        sys.exit(0)
    
    def validate_environment(self) -> bool:
        """Validate required environment variables."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ ERROR: GROQ_API_KEY environment variable not set.")
            print("Please create a .env file with your Groq API key:")
            print("  GROQ_API_KEY=your_api_key_here")
            return False
        return True
    
    def get_user_input(self) -> str:
        """Get and validate user input."""
        try:
            print("You: ", end="")
            user_input = input().strip()
            
            # Validate input length
            if len(user_input) > Config.MAX_MESSAGE_LENGTH:
                print(f"⚠️ Message too long. Please keep it under {Config.MAX_MESSAGE_LENGTH} characters.")
                return ""
            
            return user_input
            
        except EOFError:
            # Handle piped input ending
            self.running = False
            return "exit"
        except KeyboardInterrupt:
            print("\n")
            return "exit"
    
    def should_exit(self, user_input: str) -> bool:
        """Check if user wants to exit."""
        return clean_text(user_input).lower() in Config.EXIT_COMMANDS
    
    def show_debug_info(self):
        """Show debug information if enabled."""
        if Config.DEBUG:
            print("\n" + "=" * 40)
            print("DEBUG: State Summary")
            print("=" * 40)
            print(get_state_summary(self.state))
            print("=" * 40 + "\n")
    
    def display_response(self, response: str):
        """Display agent response with formatting."""
        print(f"\n🤖 Assistant: {response}\n")
    
    def run(self):
        """Main chat loop."""
        
        # Validate environment
        if not self.validate_environment():
            sys.exit(1)
        
        # Display welcome message
        print_welcome()
        
        # Initial greeting
        # Initial greeting (direct print, NOT sent to agent)
        self.display_response(
    "Hello! 👋 I'm your Company Research Assistant.\n"
    "Which company would you like me to research today?"
    )

        
        # Main conversation loop
        while self.running:
            # Get user input
            user_input = self.get_user_input()
            
            if not user_input:
                continue
            
            # Check for exit
            if self.should_exit(user_input):
                farewell_response, self.state = agent("goodbye", self.state)
                self.display_response(farewell_response)
                self.running = False
                break
            
            # Check for quick help
            if clean_text(user_input).lower() in Config.HELP_COMMANDS:
                print_help()
                continue
            
            # Process through agent
            try:
                response, self.state = agent(user_input, self.state)
                self.display_response(response)
                
                # Show debug info if enabled
                self.show_debug_info()
                
            except Exception as e:
                print(f"\n❌ An error occurred: Unable to fetch data. Recheck company name.")
                if Config.DEBUG:
                    import traceback
                    traceback.print_exc()
                print("Please try again or type 'help' for assistance.\n")
        
        print("\nThank you for using Company Research Assistant! 👋\n")


# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point."""
    
    # Check for help arguments
    if len(sys.argv) > 1 and (sys.argv[1].lower() == "--help" or sys.argv[1].lower() == "-h"):
        print("""
Company Research Assistant - CLI

Usage:
    python app.py                           Run interactive chat
    python app.py --help                    Show this help

Environment:
    GROQ_API_KEY    Your Groq API key (required)
    DEBUG=true      Enable debug output (optional)
""")
        return
    
    # Run interactive mode
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()