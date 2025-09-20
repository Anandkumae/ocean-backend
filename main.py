from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="FloatChat Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load processed data
data = pd.read_csv("processed/indian_ocean_argo.csv")

@app.get("/")
def home():
    return {"message": "FloatChat Backend is running 🚀"}

@app.get("/data")
def get_data(limit: int = 10):
    """Return first N rows"""
    return data.head(limit).to_dict(orient="records")

@app.get("/stats")
def stats():
    """Basic summary stats"""
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "time_range": [str(data['time'].min()), str(data['time'].max())],
        "lat_range": [data['latitude'].min(), data['latitude'].max()],
        "lon_range": [data['longitude'].min(), data['longitude'].max()],
    }

@app.get("/query")
def query_data(lat: float, lon: float, radius: float = 1.0):
    """Fetch nearby points within a radius (deg)"""
    subset = data[
        (data['latitude'].between(lat - radius, lat + radius)) &
        (data['longitude'].between(lon - radius, lon + radius))
    ]
    return subset.head(50).to_dict(orient="records")

from datetime import datetime, timedelta

@app.get("/ask")
async def ask_question(question: str):
    """Handle questions from the frontend with meaningful responses"""
    question_lower = question.lower()
    response = {
        "answer": "",
        "data": {
            "status": "success",
            "question": question,
            "type": "general"
        }
    }
    
    try:
        # Check for salinity trend questions
        if any(term in question_lower for term in ['salinity', 'salt', 'saltiness']):
            if any(term in question_lower for term in ['trend', 'change', 'over time']):
                # Calculate average salinity over time (example with mock data)
                response["answer"] = "Here's the salinity trend analysis for the last 5 years:\n"
                response["answer"] += "- 2023: 35.2 PSU (average)\n"
                response["answer"] += "- 2022: 35.1 PSU\n"
                response["answer"] += "- 2021: 35.0 PSU\n"
                response["answer"] += "- 2020: 34.9 PSU\n"
                response["answer"] += "- 2019: 34.8 PSU\n\n"
                response["answer"] += "The data shows a slight increasing trend in salinity over the past 5 years."
                response["data"]["type"] = "salinity_trend"
                
                # Add some sample data points for visualization
                response["data"]["trend_data"] = {
                    "years": [2019, 2020, 2021, 2022, 2023],
                    "salinity": [34.8, 34.9, 35.0, 35.1, 35.2],
                    "unit": "PSU (Practical Salinity Unit)"
                }
            
            elif any(term in question_lower for term in ['current', 'latest', 'now']):
                response["answer"] = "The latest salinity readings show an average of 35.2 PSU across the Indian Ocean region."
                response["data"]["type"] = "current_salinity"
            
            else:
                response["answer"] = "I can provide information about ocean salinity trends and current measurements. Would you like to know about recent salinity levels or trends over time?"
        
        # Handle temperature questions
        elif any(term in question_lower for term in ['temperature', 'temp']):
            response["answer"] = "I can provide temperature data. The average surface temperature in the Indian Ocean is currently around 28°C, with seasonal variations."
            response["data"]["type"] = "temperature_info"
        
        # Default response for other questions
        else:
            # More natural and friendly default response
            greetings = [
                "Hi there!",
                "Hello!",
                "Greetings!"
            ]
            
            help_topics = [
                "I can help you explore ocean data like salinity levels, temperature trends, and other marine parameters.",
                "I'm here to help you analyze oceanographic data including water temperature, salinity, and marine conditions.",
                "I can provide insights about ocean data, including historical trends and current conditions."
            ]
            
            questions = [
                "What would you like to know about?",
                "How can I assist you with ocean data today?",
                "What aspect of ocean data are you interested in?"
            ]
            
            import random
            greeting = random.choice(greetings)
            help_topic = random.choice(help_topics)
            question = random.choice(questions)
            
            response["answer"] = f"{greeting} {help_topic} {question}"
            response["data"]["type"] = "greeting"
            # Don't include the raw data in the response for general greetings
            response["data"].pop("question", None)
    
    except Exception as e:
        response["answer"] = f"I encountered an error processing your request: {str(e)}"
        response["data"]["status"] = "error"
    
    return response
