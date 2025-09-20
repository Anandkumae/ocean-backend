import os
from dotenv import load_dotenv
from pathlib import Path

print("=== Testing Environment Variables ===")
print(f"Current working directory: {os.getcwd()}")

# Try to load .env file
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
print(f"Looking for .env file at: {env_path}")
print(f"File exists: {env_path.exists()}")

# Load environment variables
load_dotenv(env_path, override=True)

# Check if the API key is loaded
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"\nAPI Key loaded: {api_key is not None}")
if api_key:
    print(f"API Key length: {len(api_key)} characters")
    print(f"Starts with: {api_key[:5]}...")
    print(f"Ends with: ...{api_key[-5:]}")
else:
    print("ERROR: OPENROUTER_API_KEY not found in environment variables")
    print("Please check that your .env file is in the correct location and contains the OPENROUTER_API_KEY")
