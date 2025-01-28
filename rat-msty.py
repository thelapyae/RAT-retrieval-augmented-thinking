#!/usr/bin/env python3
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time

# Model Constants
DEEPSEEK_MODEL = "deepseek-r1:latest"  # Local model name
LOCAL_MODEL = "llama3.2:3b"
LOCAL_API_BASE = "http://localhost:10000/v1"

class ModelChain:
    def __init__(self):
        # Single client for both local models
        self.client = OpenAI(
            base_url=LOCAL_API_BASE,
            api_key="no-key-required"
        )
        
        self.conversation_history = []
        self.show_reasoning = True
        self.reasoning_buffer = ""

    def get_model_display_name(self):
        return LOCAL_MODEL

    def get_deepseek_reasoning(self, user_input):
        start_time = time.time()
        self.conversation_history.append({"role": "user", "content": user_input})
        
        if self.show_reasoning:
            rprint("\n[blue]Local Reasoning Process[/]")
        
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=self.conversation_history,
            stream=True,
            max_tokens=1024,
            temperature=0.7
        )

        reasoning_content = ""
        
        try:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    reasoning_content += content_piece
                    if self.show_reasoning:
                        print(content_piece, end="", flush=True)
        except Exception as e:
            rprint(f"\n[red]Error in reasoning model: {str(e)}[/]")
            return "Reasoning model error"

        elapsed_time = time.time() - start_time
        time_str = f"{elapsed_time:.1f}s"
        rprint(f"\n\n[yellow]Local Reasoning ({time_str})[/]")
        
        if self.show_reasoning:
            print("\n")
        return reasoning_content

    def get_local_response(self, user_input, reasoning):
        system_prompt = """You are a helpful assistant. Use this reasoning to craft your response:
        {reasoning}
        
        Respond conversationally without markdown formatting.
        Keep responses concise and focused."""

        try:
            response = self.client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.format(reasoning=reasoning)},
                    {"role": "user", "content": user_input}
                ],
                stream=True,
                max_tokens=1024,
                temperature=0.8
            )
            
            full_response = ""
            rprint(f"\n[green]{self.get_model_display_name()}[/]")
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    full_response += content_piece
                    print(content_piece, end="", flush=True)
            print("\n")
            
            self.conversation_history.append({"role": "assistant", "content": full_response})
            return full_response
            
        except Exception as e:
            rprint(f"\n[red]Error in writing model: {str(e)}[/]")
            rprint("[yellow]Check local model server status[/]")
            return "Response generation error"

def main():
    chain = ModelChain()
    
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    rprint(Panel.fit(
        "[bold cyan]Fully Local RAT (Multi-Step Thinking)[/]",
        title="[bold cyan]RAT-MSTY 🧠[/]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/]")
    rprint(" • [bold red]'quit'[/] - Exit")
    rprint(" • [bold magenta]'reasoning'[/] - Toggle reasoning display")
    rprint(" • [bold magenta]'clear'[/] - Reset conversation\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()
            
            if user_input.lower() in ['quit', 'exit']:
                rprint("\n[cyan]Shutting down local RAT... Goodbye! 👋[/]")
                break

            if user_input.lower() == 'clear':
                chain.conversation_history = []
                rprint("\n[magenta]Conversation history cleared![/]\n")
                continue

            if user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning display: {status}[/]\n")
                continue
            
            reasoning = chain.get_deepseek_reasoning(user_input)
            if "error" in reasoning.lower():
                continue  # Skip response generation if reasoning failed
                
            local_response = chain.get_local_response(user_input, reasoning)
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()