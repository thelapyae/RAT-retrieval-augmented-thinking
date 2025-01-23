# 🧠 RAT (Retrieval Augmented Thinking)

> *Enhancing AI responses through structured reasoning and knowledge retrieval*

RAT is a powerful tool that improves AI responses by leveraging DeepSeek's reasoning capabilities to guide other models through a structured thinking process.

## How It Works

RAT employs a two-stage approach:
1. **Reasoning Stage** (DeepSeek): Generates detailed reasoning and analysis for each query
2. **Response Stage** (OpenRouter): Utilizes the reasoning context to provide informed, well-structured answers

This approach ensures more thoughtful, contextually aware, and reliable responses.

## 🎯 Features

- 🤖 **Model Selection**: Flexibility to choose from various OpenRouter models
- 🧠 **Reasoning Visibility**: Toggle visibility of the AI's thinking process
- 📝 **Smart Code Handling**: Intelligent detection and syntax highlighting of code blocks
- 📋 **Quick Copy**: Seamless code snippet copying to clipboard
- 🔄 **Context Awareness**: Maintains conversation context for more coherent interactions

## ⚙️ Requirements

• Python 3.11 or higher  
• A .env file containing:
  ```
  DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
  OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
  ```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/juice.git
   cd juice
   ```

2. Set up your environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install as a local package:
   ```bash
   pip3 install -e .
   ```

This will install JUICE as a command-line tool, allowing you to run it from anywhere by simply typing `juice`!

## 📖 Usage

1. Ensure your .env file is configured with:
   ```
   DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
   OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
   ```

2. Run JUICE from anywhere:
   ```bash
   juice
   ```

3. Available commands:
   - Enter your question to get a reasoned response
   - Use "model <name>" to switch OpenRouter models
   - Type "reasoning" to show/hide the thinking process
   - Type "quit" to exit

## ⚡ Quick Start

```bash
# Install directly from the repository
git clone https://github.com/yourusername/juice.git
cd juice
python3 -m venv venv
source venv/bin/activate
pip3 install -e .

# Run from anywhere!
juice
```

## 🤝 Contributing

Interested in improving JUICE?

1. Fork the repository
2. Create your feature branch
3. Make your improvements
4. Submit a Pull Request

## 📜 License

This project is available under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Built for developers who value thoughtful AI interactions!
