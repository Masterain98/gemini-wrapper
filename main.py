import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
from google import genai
import uuid
import time
import json
from base_logger import logger


# -------------------------
# Available Gemini Models
# -------------------------
gemini_models = {
    "Gemini": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro"
    ],
    "Gemini Experimental": [
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.0-flash-exp",
        "learnlm-1.5-pro-experimental"
    ]
}
allowed_models = [model_id for models in gemini_models.values() for model_id in models]


# Create the FastAPI app
app = FastAPI()

# Hard-coded Gemini API key (for demonstration)
GEMINI_API_KEY = os.getenv("GEMINI_KEY", None)
if GEMINI_API_KEY is None:
    raise RuntimeError("Please set the GEMINI_KEY environment variable.")

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
#  Pydantic Request Models
# -------------------------
class Message(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Message]
    # New field to enable/disable streaming
    stream: Optional[bool] = False

# -------------------------
#  OpenAI-Like Response Models
# -------------------------
class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# For the /v1/models endpoint
class ModelPermission(BaseModel):
    id: str = "permission-id"
    object: str = "model_permission"
    created: int = int(time.time())
    allow_create_engine: bool = True
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = True
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str
    permission: List[ModelPermission]

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# -------------------------
#   Endpoint: /v1/models
# -------------------------
@app.get("/v1/models")
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
@app.post("/v1/chat/completions")
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
    logger.info(f"Received request for model: {model_requested}; length: {len(user_prompt)}")

    # Call the Gemini API
    gemini_response = gemini_client.models.generate_content(
        model=model_requested,
        contents=user_prompt
    )
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
