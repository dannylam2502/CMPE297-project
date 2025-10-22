Quoc Dung Lam
Stephen Taylor
Samuel Tsao
Adam Sabry

# 🤖 Chatbot Project (CMPE 297)

This project implements a chat bot that verifies information

## 🚀 Setup and Installation

Follow these steps to set up the project environment and install dependencies. These instructions are optimized for a Windows environment using **Git Bash** (MINGW64).

### Prerequisites

1.  **Python:** Python 3.8 or later installed on your system.
2.  **OpenAI API Key:** You must have a valid OpenAI API key.

### Step 1: Clone the Repository and Create the Virtual Environment

Open your Git Bash terminal and navigate to your project directory.

```bash
# Navigate into the project folder
cd CMPE297-project

# 1. Create a virtual environment named .venv
python -m venv .venv

# Activate the environment using the source command
source .venv/Scripts/activate

# Install required libraries
pip install -r requirements.txt

# ollama install
1. install from website
2. pull model `ollama pull lama3.1`
3. confirm with `ollama list`

# Alternatively with no venv
py -m pip install <module>

# run unit tests
pytest -vs


# .env
OPENAI_API_KEY="sk-************************************"

# Run the chatbot script
python chatbot.py