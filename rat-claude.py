from openai import OpenAI
import os
import anthropic
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time

# Model Constants
DEEPSEEK_MODEL = "deepseek-reasoner"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# Load environment variables
load_dotenv()

class ModelChain:
    def __init__(self):
        # Initialize DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.deepseek_messages = []
        self.claude_messages = []
        self.current_model = CLAUDE_MODEL
        self.show_reasoning = True

    def set_model(self, model_name):
        self.current_model = model_name

    def get_model_display_name(self):
        return self.current_model

    def get_deepseek_reasoning(self, user_input):
        start_time = time.time()
        self.deepseek_messages.append({"role": "user", "content": user_input})

        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")

        response = self.deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=1,
            messages=self.deepseek_messages,
            stream=True
        )

        reasoning_content = ""
        final_content = ""

        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_piece = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_piece
                if self.show_reasoning:
                    print(reasoning_piece, end="", flush=True)
            elif chunk.choices[0].delta.content:
                final_content += chunk.choices[0].delta.content

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")

        if self.show_reasoning:
            print("\n")
        return reasoning_content

    def get_claude_response(self, user_input, reasoning):
        # Create messages with proper format
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        }

        assistant_prefill = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<thinking>{reasoning}</thinking>"
                }
            ]
        }

        messages = [user_message, assistant_prefill]

        rprint(f"[green]{self.get_model_display_name()}[/]", end="")

        try:
            with self.claude_client.messages.stream(
                model=self.current_model,
                messages=messages,
                max_tokens=8000
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text

            self.claude_messages.extend([
                user_message,
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": full_response}]
                }
            ])
            self.deepseek_messages.append({"role": "assistant", "content": full_response})

            print("\n")
            return full_response

        except Exception as e:
            rprint(f"\n[red]Error in response: {str(e)}[/]")
            return "Error occurred while getting response"

def main():
    chain = ModelChain()

    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)

    rprint(Panel.fit(
        "[bold cyan]Retrival augmented thinking[/]",
        title="[bold cyan]RAT 🧠[/]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <name>'[/] to change the Claude model")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'clear'[/] to clear chat history\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()

            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'clear':
                chain.deepseek_messages = []
                chain.claude_messages = []
                rprint("\n[magenta]Chat history cleared![/]\n")
                continue

            if user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                chain.set_model(new_model)
                print(f"\nChanged model to: {chain.get_model_display_name()}\n")
                continue

            if user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning process is now {status}[/]\n")
                continue

            reasoning = chain.get_deepseek_reasoning(user_input)
            claude_response = chain.get_claude_response(user_input, reasoning)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()