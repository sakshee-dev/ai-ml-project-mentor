from fastapi import FastAPI
from dotenv import load_dotenv
import os
import uvicorn

from app.models.user_input import UserInput
from app.services.dataset_service import recommend_datasets

load_dotenv()

app = FastAPI(title="AI ML Project Mentor")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@app.get("/")
def home():
    return {"message": "AI ML Project Mentor is running 🚀"}


@app.get("/env-check")
def check_env():
    return {"openai_key_loaded": True if OPENAI_API_KEY else False}


@app.post("/recommend")
def recommend(user_input: UserInput):
    matches = recommend_datasets(
        topic=user_input.topic,
        difficulty=user_input.difficulty,
        subtopic=user_input.subtopic
    )

    if not matches:
        return {"message": "No matching datasets found.", "results": []}

    return {
        "message": "Recommended datasets found.",
        "results": matches
    }


if __name__ == "__main__":
    print("===================================")
    print("AI ML Project Mentor Starting...")
    print("===================================")

    if OPENAI_API_KEY:
        print("OPENAI_API_KEY loaded successfully")
    else:
        print("WARNING: OPENAI_API_KEY not found in .env")

    print("Starting FastAPI server...")
    print("Open browser at: http://127.0.0.1:8000")
    print("API docs at: http://127.0.0.1:8000/docs")
    print("===================================")

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)