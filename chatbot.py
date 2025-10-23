import os
import sys
import PyPDF2
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, APIError, APIConnectionError, AuthenticationError

# --- Configuration & Initialization ---

# 1. Load Environment Variables and Validate API Key
DOTENV_PATH = find_dotenv()
print(f"Searching for .env file at: {DOTENV_PATH}")

# Load .env variables, overriding existing system variables
load_dotenv(dotenv_path=DOTENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("\n--- FATAL ERROR ---")
    print("OPENAI_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

# Basic check for common key malformation (spaces/quotes)
if OPENAI_API_KEY.startswith((" '", '"', ' ')):
    print("\n--- WARNING: KEY MALFORMATION DETECTED ---")
    print("Key appears to have leading spaces/quotes. Attempting to clean...")
    # Clean and re-set the environment variable
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip().strip("'").strip('"')

# Initialize the OpenAI Client
client = OpenAI()

# --- Core Functions ---

def load_resume(file_path):
    """Loads text from a PDF or TXT file."""
    text = ""
    if file_path.endswith(".pdf"):
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"Resume file not found at: {file_path}")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            sys.exit(1)
            
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Only .pdf or .txt resumes supported.")
    return text

def chat_complete_messages(messages):
    """Calls the OpenAI Chat Completions API with robust error handling."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print("\n--- AUTHENTICATION ERROR ---")
        print("Invalid OpenAI API Key. Please check your key in the .env file.")
    except APIConnectionError:
        print("\n--- CONNECTION ERROR ---")
        print("Could not connect to OpenAI. Check your network or proxy settings.")
    except APIError as e:
        print(f"\n--- API ERROR: {e.status_code} ---")
        print(f"An OpenAI API error occurred: {e.message}")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---")
        print(f"An unexpected error occurred: {e}")
        
    return None # Return None on any failure

# --- Main Logic ---

# Load Data
try:
    # Assuming the PDF exists in the correct location
    RESUME_TEXT = load_resume("Samuel_tsao_Resume_09_24_2025.docx.pdf")
except (ValueError, FileNotFoundError) as e:
    print(f"\n--- DATA LOAD ERROR ---")
    print(e)
    sys.exit(1)

LINKEDIN_SUMMARY = "https://www.linkedin.com/in/samuel-tsao/"
ADDITIONAL_INFO = "I'm currently learning AI product development and exploring LLM APIs."

# Initial Context for the LLM
SYSTEM_PROMPT = "You are a helpful assistant that verifies information and clears up information. Use only the provided information, and do not give information without strong confidence"

BASE_CONTEXT = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": (
        f"Resume:\n{RESUME_TEXT}\n\n"
        f"LinkedIn:\n{LINKEDIN_SUMMARY}\n\n"
        f"Additional Info:\n{ADDITIONAL_INFO}"
    )}
]

chatContext = BASE_CONTEXT.copy()

print("\n--- Chatbot Initialized ---")
print("Context Loaded. Ask a question about Samuel Tsao.")
print("---------------------------\n")

while True:
    user_input = input("say something that I can verify (or type 'exit' to quit): ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user message to context
    chatContext.append({"role": "user", "content": user_input})

    # Get response from the model
    response = chat_complete_messages(chatContext)
    
    if response:
        print("\nChatBot:", response)
        # Add assistant response to context for conversation history
        chatContext.append({"role": "assistant", "content": response})

    print(f"\n{'-'*20}\n")