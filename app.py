import os
import json
import time
import uuid
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Ki2API - Claude Sonnet 4 OpenAI Compatible API",
    description="Simple Docker-ready OpenAI-compatible API for Claude Sonnet 4",
    version="1.0.0"
)

# Configuration
API_KEY = os.getenv("API_KEY", "ki2api-key-2024")
KIRO_ACCESS_TOKEN = os.getenv("KIRO_ACCESS_TOKEN")
KIRO_REFRESH_TOKEN = os.getenv("KIRO_REFRESH_TOKEN")
KIRO_BASE_URL = "https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse"
PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK"

# Model mapping
MODEL_NAME = "claude-sonnet-4-20250514"
CODEWHISPERER_MODEL = "CLAUDE_SONNET_4_20250514_V1_0"

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4000
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]

# Token management
class TokenManager:
    def __init__(self):
        self.access_token = KIRO_ACCESS_TOKEN
        self.refresh_token = KIRO_REFRESH_TOKEN
        self.refresh_url = "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken"

    async def refresh_tokens(self):
        if not self.refresh_token:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.refresh_url,
                    json={"refreshToken": self.refresh_token},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                self.access_token = data.get("accessToken")
                return self.access_token
        except Exception as e:
            print(f"Token refresh failed: {e}")
            return None

    def get_token(self):
        return self.access_token

token_manager = TokenManager()

# Build CodeWhisperer request
def build_codewhisperer_request(messages: List[ChatMessage]):
    conversation_id = str(uuid.uuid4())
    
    # Extract system prompt and user messages
    system_prompt = ""
    user_messages = []
    
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        else:
            user_messages.append(msg)
    
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")
    
    # Build history
    history = []
    for i in range(0, len(user_messages) - 1, 2):
        if i + 1 < len(user_messages):
            history.append({
                "userInputMessage": {
                    "content": user_messages[i].content,
                    "modelId": CODEWHISPERER_MODEL,
                    "origin": "AI_EDITOR"
                }
            })
            history.append({
                "assistantResponseMessage": {
                    "content": user_messages[i + 1].content,
                    "toolUses": []
                }
            })
    
    # Build current message
    current_message = user_messages[-1]
    content = current_message.content
    if system_prompt:
        content = f"{system_prompt}\n\n{content}"
    
    return {
        "profileArn": PROFILE_ARN,
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": {
                    "content": content,
                    "modelId": CODEWHISPERER_MODEL,
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {}
                }
            },
            "history": history
        }
    }

# Make API call to Kiro/CodeWhisperer
async def call_kiro_api(messages: List[ChatMessage], stream: bool = False):
    token = token_manager.get_token()
    if not token:
        raise HTTPException(status_code=401, detail="No access token available")
    
    request_data = build_codewhisperer_request(messages)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                KIRO_BASE_URL,
                headers=headers,
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 403:
                # Try to refresh token
                new_token = await token_manager.refresh_tokens()
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    response = await client.post(
                        KIRO_BASE_URL,
                        headers=headers,
                        json=request_data,
                        timeout=120
                    )
            
            response.raise_for_status()
            return response
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API call failed: {str(e)}")

# API endpoints
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ki2api"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Only {MODEL_NAME} is supported")
    
    if request.stream:
        return await create_streaming_response(request)
    else:
        return await create_non_streaming_response(request)

async def create_non_streaming_response(request: ChatCompletionRequest):
    response = await call_kiro_api(request.messages, stream=False)
    
    # Parse response
    response_text = response.text
    
    return ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )

async def create_streaming_response(request: ChatCompletionRequest):
    response = await call_kiro_api(request.messages, stream=True)
    
    async def generate():
        # Send initial response
        initial_chunk = {
            'id': f'chatcmpl-{uuid.uuid4()}',
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': MODEL_NAME,
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant'},
                'finish_reason': None
            }]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        # Read response and stream content
        content = ""
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if 'content' in data:
                        content += data['content']
                        chunk = {
                            'id': f'chatcmpl-{uuid.uuid4()}',
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': MODEL_NAME,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': data['content']},
                                'finish_reason': None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                except:
                    continue
        
        # Send final response
        final_chunk = {
            'id': f'chatcmpl-{uuid.uuid4()}',
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': MODEL_NAME,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ki2api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)