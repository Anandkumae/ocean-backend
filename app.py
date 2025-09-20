from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pandas as pd
import os
import uvicorn
import requests
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

# OpenRouter API Configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-70b-instruct"  # Using Llama 3 model through OpenRouter

# Debug: Print current directory and environment
print("\n=== Debug Information ===")
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("Environment variables:", {k: '***' if 'KEY' in k else v for k, v in os.environ.items() if 'API' in k or 'KEY' in k})

# Load environment variables
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
print(f"\nLoading environment variables from: {env_path}")
print(f"File exists: {env_path.exists()}")
load_dotenv(env_path, override=True)

# Get the API key
API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"OPENROUTER_API_KEY loaded: {API_KEY is not None}")
if API_KEY:
    print(f"API Key length: {len(API_KEY)} characters")
    print(f"API Key starts with: {API_KEY[:5]}...")
    print(f"API Key ends with: ...{API_KEY[-5:]}")
    
    # Validate API key format
    if not API_KEY.startswith('sk-or-v1-'):
        print("\n!!! WARNING: Invalid OpenRouter API key format !!!")
        print("API key should start with 'sk-or-v1-'. Please check your API key in the OpenRouter dashboard.")
else:
    print("\n!!! WARNING: OPENROUTER_API_KEY not found in environment variables !!!")
    print("Please make sure your .env file is in the correct location and contains the OPENROUTER_API_KEY")

# Set up headers for API requests
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://github.com/Anandkumae/Argo_Ocean",
    "X-Title": "Ocean Agro",
    "Content-Type": "application/json"
}

# Global variable to store the data
data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n=== Starting application lifespan ===")
    try:
        # Load data when the application starts
        print("Loading data...")
        global data
        data = load_data()
        print("Data loaded successfully")
        print(f"Data type: {type(data)}")
        if data is not None and hasattr(data, 'shape'):
            print(f"Data shape: {data.shape}")
        yield
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up when the application shuts down
        print("\n=== Cleaning up application ===")
        data = None

app = FastAPI(
    title="FloatChat Backend",
    description="Backend API for FloatChat application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to handle CORS headers and preflight requests
@app.middleware("http")
async def add_cors_headers(request, call_next):
    # Handle preflight requests
    if request.method == "OPTIONS":
        response = Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "600"  # 10 minutes
            }
        )
    else:
        # Process the request and add CORS headers to the response
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

def load_data():
    """Load the data from CSV file"""
    print("\n=== Starting data load ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to find the CSV file in different possible locations
    csv_paths = [
        os.path.join("processed", "indian_ocean_argo.csv"),
        os.path.join("..", "data", "processed", "indian_ocean_argo.csv"),
        "indian_ocean_argo.csv"
    ]
    
    # Add absolute paths for better debugging
    abs_paths = [os.path.abspath(p) for p in csv_paths]
    
    print("\nSearching for data files in the following locations:")
    for i, path in enumerate(abs_paths):
        exists = "EXISTS" if os.path.exists(path) else "NOT FOUND"
        print(f"{i+1}. {path} - {exists}")
    
    for path in csv_paths:
        try:
            abs_path = os.path.abspath(path)
            print(f"\nAttempting to load: {abs_path}")
            
            if os.path.exists(abs_path):
                print(f"File found. Loading data...")
                df = pd.read_csv(abs_path)
                
                # Basic data validation
                if df.empty:
                    print("WARNING: Loaded an empty DataFrame!")
                
                print(f"Successfully loaded data from: {abs_path}")
                print(f"Data shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"First few rows:\n{df.head(2).to_string()}")
                
                return df
            else:
                print(f"File not found: {abs_path}")
                
        except Exception as e:
            print(f"\nERROR loading {path}:")
            import traceback
            traceback.print_exc()
    
    # If we get here, no valid file was found
    error_msg = "\nERROR: Could not find the data file. Tried the following locations:\n" + "\n".join([f"- {p}" for p in abs_paths])
    print(error_msg)
    
    # Create a minimal valid DataFrame to prevent crashes
    print("\nCreating a minimal valid DataFrame to prevent application crash...")
    sample_data = {
        'latitude': [0.0],
        'longitude': [0.0],
        'time': ['2023-01-01'],
        'pressure': [0.0],
        'temperature': [25.0]
    }
    df = pd.DataFrame(sample_data)
    print("Created sample DataFrame with shape:", df.shape)
    
    return df

# Load data when the module is imported
data = load_data()

# Convert date column to datetime if it exists
if data is not None and 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')


class Question(BaseModel):
    question: str

class QuestionQuery(BaseModel):
    question: str

@app.get("/ask")
async def ask_get(question: str):
    """
    Handle GET requests to the /ask endpoint.
    Example: /ask?question=What is oceanography?
    """
    return await ask_question(question)

@app.post("/ask")
async def ask_post(question_data: Question):
    """
    Handle POST requests to the /ask endpoint.
    """
    if not question_data or not question_data.question:
        raise HTTPException(status_code=400, detail="Question is required")
    return await ask_question(question_data.question)

async def ask_question(question_text: str):
    """
    Common function to handle the question processing.
    """
    if not question_text:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check if API key is available
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Configuration Error",
                "details": "OpenRouter API key is not configured in environment variables"
            }
        )
    
    print(f"Using OpenRouter API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    
    try:
        # Prepare the request payload for OpenRouter
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an ocean data assistant. Answer questions about oceanography, marine life, and related topics clearly and concisely."},
                {"role": "user", "content": question_text}
            ]
        }
        
        # Add additional parameters to the payload
        payload.update({
            "max_tokens": 1000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        })
        
        # Print debug info
        print("\n=== Sending request to OpenRouter API ===")
        print(f"API URL: {API_URL}")
        print(f"Model: {MODEL}")
        print(f"Headers: { {k: '***' if k.lower() == 'authorization' else v for k, v in HEADERS.items()} }")
        
        # Make the API call to OpenRouter
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=30  # 30 seconds timeout
        )
        
        # Print response info
        print("\n=== Received response ===")
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text[:500]}")
        
        response.raise_for_status()  # This will raise an exception for 4XX/5XX responses
        
        # Extract the response content
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content']
            return {"answer": answer}
        else:
            raise ValueError("Unexpected response format from OpenRouter API")
            
    except requests.exceptions.RequestException as e:
        print(f"\n=== Error Details ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        error_detail = str(e)
        response_info = {}
        
        if hasattr(e, 'response') and e.response is not None:
            try:
                response_info = {
                    "status_code": e.response.status_code,
                    "headers": dict(e.response.headers),
                    "body": e.response.text
                }
                print(f"Response info: {response_info}")
                
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        if isinstance(error_json['error'], dict):
                            error_detail = error_json['error'].get('message', str(error_json['error']))
                            
                            # Add specific handling for authentication errors
                            if e.response.status_code == 401:
                                error_detail = "Authentication failed. Please check your OpenRouter API key. " \
                                             "Make sure it's correct and hasn't been revoked. " \
                                             f"Key format should be: sk-or-v1-... (received: {API_KEY[:10]}...)"
                        else:
                            error_detail = str(error_json['error'])
                    else:
                        error_detail = str(error_json)
                except:
                    error_detail = e.response.text or str(e)
            except Exception as inner_e:
                print(f"Error processing response: {str(inner_e)}")
        
        # Provide more helpful error messages based on status code
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 401:
                error_detail = "Authentication failed. Please check your OpenRouter API key. " \
                             "Make sure it's correct and hasn't been revoked. " \
                             f"Key format should be: sk-or-v1-... (received: {API_KEY[:10]}...)"
            elif e.response.status_code == 429:
                error_detail = "Rate limit exceeded. Please check your OpenRouter plan limits."
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process your request with OpenRouter API",
                "details": error_detail,
                "debug": {
                    "api_key_exists": bool(API_KEY),
                    "api_key_starts_with": API_KEY[:10] if API_KEY else None,
                    "api_key_length": len(API_KEY) if API_KEY else 0,
                    "request_url": API_URL,
                    "response_status": e.response.status_code if hasattr(e, 'response') and e.response else None
                }
            }
        )

@app.get("/")
def home():
    return {"message": "FloatChat Backend is running ",
            "status": "running",
            "data_loaded": not data.empty}

@app.get("/data")
def get_data(limit: int = 10):
    """Return first N rows"""
    if data is None or data.empty:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Data not loaded or empty",
                "message": "Please check the server logs for more information.",
                "status": "error"
            }
        )
    return {
        "status": "success",
        "count": len(data.head(limit)),
        "data": data.head(limit).to_dict(orient="records")
    }

@app.get("/stats")
def stats():
    """Basic summary stats"""
    if data.empty:
        raise HTTPException(status_code=503, detail="Data not loaded. Please check server logs.")
    
    try:
        return {
            "rows": len(data),
            "columns": list(data.columns),
            "time_range": [str(data['time'].min()), str(data['time'].max())],
            "lat_range": [data['latitude'].min(), data['latitude'].max()],
            "lon_range": [data['longitude'].min(), data['longitude'].max()],
        }
    except KeyError as e:
        missing_column = str(e).strip("'")
        raise HTTPException(
            status_code=500,
            detail=f"Required column '{missing_column}' not found in the dataset. Available columns: {list(data.columns)}"
        )

@app.get("/query")
def query_data(lat: float, lon: float, radius: float = 1.0):
    """Fetch nearby points within a radius (deg)"""
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        subset = data[
            (data['latitude'].between(lat - radius, lat + radius)) &
            (data['longitude'].between(lon - radius, lon + radius))
        ]
        return subset.head(50).to_dict(orient="records")
    except KeyError as e:
        missing_column = str(e).strip("'")
        raise HTTPException(
            status_code=500,
            detail=f"Required column '{missing_column}' not found in the dataset. Available columns: {list(data.columns)}"
        )

@app.get("/api/floats")
async def get_filtered_floats(
    lat_min: Optional[float] = Query(None, description="Minimum latitude (-90 to 90)"),
    lat_max: Optional[float] = Query(None, description="Maximum latitude (-90 to 90)"),
    lon_min: Optional[float] = Query(None, description="Minimum longitude (-180 to 180)"),
    lon_max: Optional[float] = Query(None, description="Maximum longitude (-180 to 180)"),
    depth_max: Optional[float] = Query(None, description="Maximum depth in meters"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Filter Argo floats based on various parameters.
    Returns a list of floats that match all specified criteria.
    """
    print(f"\n=== /api/floats called with params: lat_min={lat_min}, lat_max={lat_max}, "
          f"lon_min={lon_min}, lon_max={lon_max}, depth_max={depth_max}, "
          f"start_date={start_date}, end_date={end_date} ===")
    
    if data is None:
        error_msg = "Error: Data not loaded. Check if the data file exists and is being loaded correctly."
        print(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)
    
    # Log available columns for debugging
    print("\n=== Available columns in data ===")
    print(list(data.columns))
    print("==============================\n")
    
    try:
        print(f"Data loaded successfully. Shape: {data.shape}, Columns: {list(data.columns)}")
        
        # Start with a copy of the data
        filtered_data = data.copy()
        
        # Apply filters with validation
        if lat_min is not None:
            print(f"Applying latitude minimum filter: {lat_min}")
            filtered_data = filtered_data[filtered_data['latitude'] >= lat_min]
        if lat_max is not None:
            print(f"Applying latitude maximum filter: {lat_max}")
            filtered_data = filtered_data[filtered_data['latitude'] <= lat_max]
        if lon_min is not None:
            print(f"Applying longitude minimum filter: {lon_min}")
            filtered_data = filtered_data[filtered_data['longitude'] >= lon_min]
        if lon_max is not None:
            print(f"Applying longitude maximum filter: {lon_max}")
            filtered_data = filtered_data[filtered_data['longitude'] <= lon_max]
        if depth_max is not None:
            print(f"Applying depth maximum filter: {depth_max}")
            # Check if depth column exists, if not try common alternatives
            depth_columns = [col for col in filtered_data.columns if 'depth' in col.lower()]
            if not depth_columns:
                print("Warning: No depth column found. Available columns:", list(filtered_data.columns))
            else:
                depth_col = depth_columns[0]  # Use the first matching column
                print(f"Using column '{depth_col}' for depth filtering")
                filtered_data = filtered_data[filtered_data[depth_col] <= depth_max]
            
        # Date filtering
        if 'date' in filtered_data.columns:
            print(f"Date column found in data. Available date range: {filtered_data['date'].min()} to {filtered_data['date'].max()}")
            if start_date:
                try:
                    start_date_dt = pd.to_datetime(start_date)
                    print(f"Applying start date filter: {start_date} ({start_date_dt})")
                    filtered_data = filtered_data[filtered_data['date'] >= start_date_dt]
                except Exception as e:
                    error_msg = f"Invalid start_date format: {start_date}. Use YYYY-MM-DD format."
                    print(f"{error_msg} Error: {str(e)}")
                    raise ValueError(error_msg) from e
                    
            if end_date:
                try:
                    end_date_dt = pd.to_datetime(end_date)
                    print(f"Applying end date filter: {end_date} ({end_date_dt})")
                    filtered_data = filtered_data[filtered_data['date'] <= end_date_dt]
                except Exception as e:
                    error_msg = f"Invalid end_date format: {end_date}. Use YYYY-MM-DD format."
                    print(f"{error_msg} Error: {str(e)}")
                    raise ValueError(error_msg) from e
        else:
            print("Warning: 'date' column not found in data. Date filtering will be skipped.")
        
        print(f"Found {len(filtered_data)} matching records")
        
        # Convert to list of dicts for JSON serialization
        result = filtered_data.to_dict(orient='records')
        
        # Ensure all numeric values are JSON serializable
        for item in result:
            for key, value in item.items():
                if pd.isna(value):
                    item[key] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    item[key] = value.isoformat()
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"{error_msg}\nTraceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
