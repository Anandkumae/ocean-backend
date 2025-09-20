import uvicorn
from app import app, API_KEY, HEADERS

print("\n=== FastAPI Configuration Test ===")
print(f"API Key loaded in app: {bool(API_KEY)}")
print(f"Headers configured: {bool(HEADERS)}")

if API_KEY:
    print("\n=== Testing API Key in Headers ===")
    auth_header = HEADERS.get('Authorization', '')
    print(f"Authorization header: {auth_header[:15]}..." if auth_header else "No Authorization header found")
