import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
import uuid
import time
import json
import asyncio
import httpx
from base_logger import logger
from models import (Message, OpenAIChatRequest, ChatCompletionChoice, UsageInfo, ChatCompletionResponse,
                    ModelPermission, ModelInfo, ModelListResponse)


# -------------------------
# Available Gemini Models
# -------------------------
gemini_models = {"Gemini": [], "Gemini Experimental": []}

# Create the FastAPI lifespan event handler
async def lifespan(app: FastAPI):
    # Initial fetch and start periodic background task
    await fetch_available_models()  # Initial fetch
    asyncio.create_task(update_models_periodically())
    yield

# Update app initialization to use lifespan event handler
app = FastAPI(lifespan=lifespan)

# Process Gemini Keys
lock = asyncio.Lock()
gemini_keys = os.getenv("GEMINI_KEY", "").split(",")
if gemini_keys is None:
    raise RuntimeError("Please set the GEMINI_KEY environment variable.")

# Initialize the Gemini client with the first key
current_key_index = 0
logger.info(f"Gemini key {current_key_index}: *****{gemini_keys[current_key_index][:5]} is used")
gemini_client = genai.Client(api_key=gemini_keys[current_key_index])

STATIC_API_KEY = os.getenv("API_KEY")
if not STATIC_API_KEY:
    raise RuntimeError("API_KEY environment variable is not set.")


async def verify_api_key(authorization: str = Header(...)):
    authorization = authorization.replace("Bearer ", "")
    if authorization != STATIC_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# Rotates to the next key on 418
async def rotate_key_on_418():
    global current_key_index, gemini_client
    async with lock:
        if current_key_index < len(gemini_keys) - 1:
            current_key_index += 1
            gemini_client = genai.Client(api_key=gemini_keys[current_key_index])
        else:
            current_key_index = 0
            gemini_client = genai.Client(api_key=gemini_keys[current_key_index])


# -------------------------------------------
# Function to fetch available Gemini models
# -------------------------------------------
async def fetch_available_models():
    global gemini_models
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://generativelanguage.googleapis.com/v1beta/models")
        data = resp.json()
        models_list = data.get("models", [])
        new_models = {"Gemini": [], "Gemini Experimental": []}
        for model in models_list:
            model_id = model.get("id", model) if isinstance(model, dict) else model
            if "exp" in model_id.lower() or "preview" in model_id.lower():
                new_models["Gemini Experimental"].append(model_id)
            else:
                new_models["Gemini"].append(model_id)
        gemini_models = new_models
        logger.info("Gemini models updated: " + json.dumps(gemini_models))
    except Exception as e:
        logger.error(f"Error fetching models: {e}")


# --------------------------------------------------
# Background task for periodically updating models
# --------------------------------------------------
async def update_models_periodically():
    while True:
        await fetch_available_models()
        await asyncio.sleep(3600)  # update every 1 hour


# -------------------------
#   Endpoint: /v1/models
# -------------------------
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """
    Return a minimal OpenAI-like list of models.
    For demonstration, only one model is shown (gpt-4).
    """
    data = []
    for owned_by, models in gemini_models.items():
        for model_id in models:
            model_info = ModelInfo(
                id=model_id,
                owned_by=owned_by,
                permission=[ModelPermission()]
            )
            data.append(model_info)
    return ModelListResponse(data=data)


# -------------------------
#   Endpoint: /v1/chat/completions
# -------------------------
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def create_chat_completion(request: OpenAIChatRequest):
    """
    Mock an OpenAI /v1/chat/completions endpoint that:
      1) Accepts Chat Completion requests in an OpenAI-like format.
      2) Internally calls Gemini's API using our hardcoded key.
      3) Returns either a full JSON response or a streaming (chunked) response
         in an OpenAI-like format (depending on the 'stream' flag).
    """

    # Extract the user prompt from the last message
    model_requested = request.model
    user_prompt = request.messages[-1].content if request.messages else ""
    model_temperature = request.temperature
    logger.info(f"Received request for model: {model_requested}; length: {len(user_prompt)}")

    # Call the Gemini API
    try:
        gemini_response = gemini_client.models.generate_content(
            model=model_requested,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=model_temperature
            )
        )
    except HTTPException as e:
        if e.status_code == 418:
            await rotate_key_on_418()
            gemini_response = gemini_client.models.generate_content(
                model=model_requested,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=model_temperature
                )
            )
        else:
            raise HTTPException(status_code=e.status_code, detail=e.detail)

    response_text = gemini_response.text

    # --------------------------------------
    # If the user wants a streaming response
    # --------------------------------------
    if request.stream:
        def stream_chunks():
            """
            Yields response in small chunks in OpenAI-like streaming format.
            Each chunk is a JSON line prefixed with 'data:'.
            """
            chunk_size = 30  # Number of characters per chunk (arbitrary example)

            # Generate partial chunks of text
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i : i + chunk_size]

                # Build a chunk in OpenAI's streaming response format
                data = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            # 'delta' holds the incremental content
                            "delta": {"content": chunk},
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }

                # Prefix with 'data:' for SSE and double-newline to end the event
                yield f"data: {json.dumps(data)}\n\n"

            # Send a final event with finish_reason="stop"
            finish_data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(finish_data)}\n\n"

            # Final marker
            yield "data: [DONE]\n\n"

        # Return a StreamingResponse
        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    # ------------------------------------------------
    # Otherwise, return the entire response at once
    # ------------------------------------------------
    standard_resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    # Use the Gemini text as the entire content
                    "content": response_text,
                    "refusal": None
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 10,
            "total_tokens": 18,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "system_fingerprint": None
    }

    return standard_resp


if __name__ == "__main__":
    # Run the app (for local testing)
    uvicorn.run(app, host="0.0.0.0", port=8300)
