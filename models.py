from pydantic import BaseModel
from typing import List, Optional, Literal
import time


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
    temperature: Optional[float] = 0.7

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