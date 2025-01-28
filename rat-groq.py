#!/usr/bin/env python3
from openai import OpenAI
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Configuration - Use absolute path for .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THINKING_MODEL = "deepseek-r1-distill-llama-70b"
ANSWERING_MODEL = "llama-3.3-70b-versatile"

class GroqDualModelChain:
    def __init__(self):
        # Configure Groq client
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
        self.chat_history = []
        self.test_groq_connection()

    def test_groq_connection(self):
        try:
            self.groq_client.models.list()
            rprint("[green]Groq API connection successful[/]")
        except Exception as e:
            rprint(f"\n[red]Groq connection failed: {str(e)}[/]")
            rprint("[yellow]Ensure your API key is valid and Groq API is accessible[/]")
            exit(1)

    def stream_thinking(self, user_input):
        rprint(Panel.fit("🤔 [bold cyan]Groq Thinking Process with DeepSeek[/]", border_style="cyan"))
        start_time = time.time()
        
        prompt = f"Perform comprehensive analysis of: {user_input}"
        
        try:
            response = self.groq_client.chat.completions.create(
                model=THINKING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.7
            )
            
            full_reasoning = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    reasoning = chunk.choices[0].delta.content
                    full_reasoning += reasoning
                    rprint(f"[blue]{reasoning}[/]", end="", flush=True)
            
            elapsed = time.time() - start_time
            rprint(f"\n\n[yellow]Cognitive processing time: {elapsed:.1f}s[/]\n")
            return full_reasoning
            
        except Exception as e:
            rprint(f"\n[red]Groq Thinking Error: {str(e)}[/]")
            return "Thinking process failed"

    def stream_response(self, user_input, reasoning):
        rprint(Panel.fit("💡 [bold green]Groq Answering with Versatile LLaMA[/]", border_style="green"))
        
        combined_prompt = f"""**User Query**: {user_input}
        
        **Analysis**: {reasoning}
        
        **Instructions**: Provide detailed answer in plain text format."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=ANSWERING_MODEL,
                messages=[{"role": "user", "content": combined_prompt}],
                stream=True,
                temperature=0.7,
                max_tokens=2000
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            
            self.chat_history.append((user_input, full_response))
            print("\n")
            return full_response
            
        except Exception as e:
            rprint(f"\n[red]Groq Response Error: {str(e)}[/]")
            return "Response generation failed"

def main():
    chain = GroqDualModelChain()
    session = PromptSession(style=Style.from_dict({'prompt': 'cyan bold'}))
    
    rprint(Panel.fit(
        "[bold magenta]RAT - Groq Dual Model Cognitive System[/]",
        subtitle="[yellow]DeepSeek Thinking | Versatile Answering[/]",
        border_style="magenta"
    ))
    
    while True:
        try:
            user_input = session.prompt("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                rprint("\n[magenta]Terminating cognitive session... Goodbye![/]")
                break
                
            if user_input.lower() == 'clear':
                chain.chat_history = []
                rprint("\n[cyan]Working memory cleared![/]\n")
                continue
                
            reasoning = chain.stream_thinking(user_input)
            response = chain.stream_response(user_input, reasoning)
            
        except KeyboardInterrupt:
            continue

if __name__ == "__main__":
    main()