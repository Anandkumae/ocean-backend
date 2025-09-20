from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import uvicorn
import requests
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel

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

# Enable CORS with specific origins
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",  # Alternative localhost
    "http://localhost:8000",  # For direct API access
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

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

# OpenRouter API configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"  # you can also try deepseek-coder, etc.

@app.get("/ask")
def ask(question: str):
    """
    Ask a question to the ocean data assistant using OpenRouter's API.
    
    Args:
        question (str): The question to ask the assistant
        
    Returns:
        dict: Contains the assistant's answer or an error message
    """
    if not API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key is not configured. Please set the OPENROUTER_API_KEY environment variable in a .env file"
        )
        
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an ocean data assistant. Answer questions about oceanography, marine life, and related topics clearly and concisely."},
            {"role": "user", "content": question}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        try:
            return {
                "status": "success",
                "answer": result["choices"][0]["message"]["content"]
            }
        except (KeyError, IndexError) as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to parse API response",
                    "details": str(e),
                    "response": result
                }
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to communicate with OpenRouter API",
                "details": str(e)
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
