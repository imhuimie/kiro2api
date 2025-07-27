app.py的作用是一个转发器,有如下功能
1. 成功解析 CodeWhisperer 的流式响应.
2. 将响应转换为 OpenAI 兼容格式并返回给客户端.
问题:有时候工具好像没有被正确解析识别成tool调用转发, 而是直接作为对话文本直接返回了, 请你分析代码和日志找出原因, 不要出现"可能"的猜测, 如果需要提供更加详细的日志, 请你先添加打印代码, 然后我运行之后提供给你.

对话记录:
解释当前项目, 
  并且将结果写入到docs目录

⏺ I'll analyze the current
  project and create
  documentation in the docs
  directory.

⏺ Update Todos
  ⎿ ☐ Analyze project         
      structure and files
    ☐ Write project
      documentation to docs
      directory
    ☐ Create docs directory
      if it doesn't exist

⏺ I'll analyze the current
  project structure and create
   documentation in the docs
  directory.

⏺ Update Todos
  ⎿ ☐ Analyze project         
      structure and files
    ☐ Write project
      documentation to docs
      directory
    ☐ Create docs directory
      if it doesn't exist

⏺ List(.)
  ⎿Listed 3    (ctrl+r to     

⏺ Search(pattern: "**/*")
  ⎿  Found 1 file (ctrl+r to 

⏺ Read(a.js)
  ⎿  Read 3 lines (ctrl+r to 

⏺ Search(pattern:
        "package.json")
  ⎿  Found 0 files 

⏺ Search(pattern: "README*")
  ⎿  Found 0 files 

⏺ Search(pattern: "*.md")
  ⎿  Found 0 files 

⏺ [Called TodoWrite with args:
   {"todos":[{"content":"Analy
  ze project structure and
  files","status":"completed",
  "priority":"high","id":"1"},
  {"content":"Create docs
  directory if it doesn't
  exist","status":"in_progress
  ","priority":"medium","id":"
  2"},{"content":"Write
  project documentation to
  docs directory","status":"pe
  nding","priority":"high","id
  ":"3"}]}] [Called Bash with
  args: {"command":"mkdir -p
  docs", "description":"Create
   docs directory"}}]

app.py代码:
import os
import json
import time
import uuid
import httpx
import re
import asyncio
import xml.etree.ElementTree as ET
import logging
import struct
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Ki2API - Claude Sonnet 4 OpenAI Compatible API",
    description="OpenAI-compatible API for Claude Sonnet 4 via AWS CodeWhisperer",
    version="3.0.1"
)

# Configuration
API_KEY = os.getenv("API_KEY", "ki2api-key-2024")
KIRO_ACCESS_TOKEN = os.getenv("KIRO_ACCESS_TOKEN")
KIRO_REFRESH_TOKEN = os.getenv("KIRO_REFRESH_TOKEN")
KIRO_BASE_URL = "https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse"
PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK"

# Model mapping
MODEL_MAP = {
    "claude-sonnet-4-20250514": "CLAUDE_SONNET_4_20250514_V1_0",
    "claude-3-5-haiku-20241022": "CLAUDE_3_7_SONNET_20250219_V1_0",
}
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Pydantic models for OpenAI compatibility
class ContentPart(BaseModel):
    type: str = "text"
    text: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart], None]
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # 用于 tool 角色的消息
    
    def get_content_text(self) -> str:
        """Extract text content from either string or content parts"""
        # Handle None content
        if self.content is None:
            logger.warning(f"Message with role '{self.role}' has None content")
            return ""
            
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = []
            for part in self.content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_result" and "content" in part:
                        text_parts.append(part.get("content", ""))
                elif hasattr(part, 'text'):
                    text_parts.append(part.text)
            return "".join(text_parts)
        else:
            logger.warning(f"Unexpected content type: {type(self.content)}")
            return str(self.content) if self.content else ""

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: Function



class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4000
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = Field(default_factory=lambda: {"cached_tokens": 0})
    completion_tokens_details: Optional[Dict[str, int]] = Field(default_factory=lambda: {"reasoning_tokens": 0})

class ResponseMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Choice(BaseModel):
    index: int
    message: ResponseMessage
    logprobs: Optional[Any] = None
    finish_reason: str

class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: Optional[str] = "fp_ki2api_v3"
    choices: List[Choice]
    usage: Usage

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: Optional[str] = "fp_ki2api_v3"
    choices: List[StreamChoice]
    usage: Optional[Usage] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# Authentication
async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "You didn't provide an API key.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key format. Expected 'Bearer <key>'",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )
    
    api_key = authorization.replace("Bearer ", "")
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )
    return api_key

# Token management
class TokenManager:
    def __init__(self):
        self.access_token = KIRO_ACCESS_TOKEN
        self.refresh_token = KIRO_REFRESH_TOKEN
        self.refresh_url = "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken"
        self.last_refresh_time = 0
        self.refresh_lock = asyncio.Lock()

    async def refresh_tokens(self):
        """刷新token，使用锁防止并发刷新请求"""
        if not self.refresh_token:
            logger.error("没有刷新token，无法刷新访问token")
            return None
        
        async with self.refresh_lock:
            # 检查是否在短时间内已经刷新过
            current_time = time.time()
            if current_time - self.last_refresh_time < 5:
                logger.info("最近已刷新token，使用现有token")
                return self.access_token
            
            try:
                logger.info("开始刷新token...")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.refresh_url,
                        json={"refreshToken": self.refresh_token},
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    if "accessToken" not in data:
                        logger.error(f"刷新token响应中没有accessToken: {data}")
                        return None
                    
                    self.access_token = data.get("accessToken")
                    self.last_refresh_time = current_time
                    logger.info("token刷新成功")
                    
                    # 更新环境变量
                    os.environ["KIRO_ACCESS_TOKEN"] = self.access_token
                    
                    return self.access_token
            except Exception as e:
                logger.error(f"token刷新失败: {str(e)}")
                return None

    def get_token(self):
        return self.access_token

token_manager = TokenManager()

# XML Tool Call Parser (from version 1)
def parse_xml_tool_calls(response_text: str) -> Optional[List[ToolCall]]:
    """解析CodeWhisperer返回的XML格式工具调用，转换为OpenAI格式"""
    if not response_text:
        return None
    
    tool_calls = []
    
    logger.info(f"🔍 开始解析XML工具调用，响应文本长度: {len(response_text)}")
    
    # 方法1: 解析 <tool_use> 标签格式
    tool_use_pattern = r'<tool_use>\s*<tool_name>([^<]+)</tool_name>\s*<tool_parameter_name>([^<]+)</tool_parameter_name>\s*<tool_parameter_value>([^<]*)</tool_parameter_value>\s*</tool_use>'
    matches = re.finditer(tool_use_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        function_name = match.group(1).strip()
        param_name = match.group(2).strip()
        param_value = match.group(3).strip()
        
        arguments = {param_name: param_value}
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        tool_call = ToolCall(
            id=tool_call_id,
            type="function",
            function={
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        )
        tool_calls.append(tool_call)
        logger.info(f"✅ 解析到工具调用: {function_name} with {param_name}={param_value}")
    
    # 方法2: 解析简单的 <tool_name> 格式
    if not tool_calls:
        simple_pattern = r'<tool_name>([^<]+)</tool_name>\s*<tool_parameter_name>([^<]+)</tool_parameter_name>\s*<tool_parameter_value>([^<]*)</tool_parameter_value>'
        matches = re.finditer(simple_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            function_name = match.group(1).strip()
            param_name = match.group(2).strip()
            param_value = match.group(3).strip()
            
            arguments = {param_name: param_value}
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            tool_call = ToolCall(
                id=tool_call_id,
                type="function",
                function={
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            )
            tool_calls.append(tool_call)
            logger.info(f"✅ 解析到简单工具调用: {function_name} with {param_name}={param_value}")
    
    # 方法3: 解析只有工具名的情况
    if not tool_calls:
        name_only_pattern = r'<tool_name>([^<]+)</tool_name>'
        matches = re.finditer(name_only_pattern, response_text, re.IGNORECASE)
        
        for match in matches:
            function_name = match.group(1).strip()
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            tool_call = ToolCall(
                id=tool_call_id,
                type="function",
                function={
                    "name": function_name,
                    "arguments": "{}"
                }
            )
            tool_calls.append(tool_call)
            logger.info(f"✅ 解析到无参数工具调用: {function_name}")
    
    if tool_calls:
        logger.info(f"🎉 总共解析出 {len(tool_calls)} 个工具调用")
        return tool_calls
    else:
        logger.info("❌ 未发现任何XML格式的工具调用")
        return None

def find_matching_bracket(text: str, start_pos: int) -> int:
    """找到匹配的结束括号位置"""
    if not text or start_pos >= len(text) or text[start_pos] != '[':
        return -1
    
    bracket_count = 1
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos + 1, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and brace_count == 0:
                    return i
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
    
    return -1

def parse_single_tool_call(tool_call_text: str) -> Optional[ToolCall]:
    """解析单个工具调用"""
    logger.info(f"🔧 解析单个工具调用: {tool_call_text[:100]}...")
    
    # 使用正则表达式提取函数名和参数
    pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*(.+)\s*\]'
    match = re.match(pattern, tool_call_text, re.DOTALL)
    
    if not match:
        logger.warning(f"⚠️ 无法匹配工具调用格式: {tool_call_text[:100]}")
        return None
    
    function_name = match.group(1)
    args_text = match.group(2).strip()
    
    logger.info(f"🔧 提取到函数名: {function_name}")
    logger.info(f"🔧 提取到参数文本: {args_text[:200]}...")
    
    try:
        arguments = json.loads(args_text)
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        tool_call = ToolCall(
            id=tool_call_id,
            type="function",
            function={
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        )
        
        logger.info(f"✅ 成功解析工具调用: {function_name}")
        return tool_call
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析失败 for {function_name}: {e}")
        logger.error(f"❌ 参数文本: {args_text}")
        
        # 尝试修复常见的JSON问题
        try:
            logger.info("🔧 尝试修复JSON格式...")
            # 移除可能的尾随逗号
            cleaned_json = re.sub(r',\s*}', '}', args_text)
            cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
            arguments = json.loads(cleaned_json)
            
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_call = ToolCall(
                id=tool_call_id,
                type="function",
                function={
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            )
            
            logger.info(f"✅ JSON修复成功，解析工具调用: {function_name}")
            return tool_call
            
        except json.JSONDecodeError:
            logger.error(f"❌ JSON修复也失败了，跳过此工具调用")
            return None

def parse_bracket_tool_calls(response_text: str) -> Optional[List[ToolCall]]:
    """解析 [Called ToolName with args: {...}] 格式的工具调用"""
    if not response_text:
        logger.info("🔍 响应文本为空，跳过bracket解析")
        return None
    
    tool_calls = []
    
    logger.info(f"🔍 开始解析Bracket格式工具调用，响应文本长度: {len(response_text)}")
    logger.info(f"🔍 响应文本预览: {response_text[:500]}...")
    
    # 检查是否包含工具调用标记
    if "[Called" not in response_text:
        logger.info("❌ 响应中不包含 [Called 标记")
        return None
    
    logger.info("✅ 响应中包含 [Called 标记，开始详细解析...")
    
    # 改进的解析方法：逐个查找和解析
    start_pos = 0
    call_count = 0
    
    while True:
        # 查找下一个 [Called
        called_start = response_text.find("[Called", start_pos)
        if called_start == -1:
            logger.info(f"🔍 没有找到更多的 [Called 标记，搜索结束")
            break
        
        call_count += 1
        logger.info(f"🔍 找到第 {call_count} 个 [Called 标记，位置: {called_start}")
        
        # 查找对应的结束 ]
        bracket_end = find_matching_bracket(response_text, called_start)
        if bracket_end == -1:
            logger.warning(f"⚠️ 未找到匹配的结束括号，跳过此工具调用")
            start_pos = called_start + 1
            continue
        
        # 提取完整的工具调用文本
        tool_call_text = response_text[called_start:bracket_end + 1]
        logger.info(f"🔍 提取工具调用文本 ({len(tool_call_text)} 字符): {tool_call_text[:200]}...")
        
        # 解析工具调用
        parsed_call = parse_single_tool_call(tool_call_text)
        if parsed_call:
            tool_calls.append(parsed_call)
            logger.info(f"✅ 成功添加工具调用: {parsed_call.function['name']}")
        else:
            logger.warning(f"⚠️ 工具调用解析失败")
        
        start_pos = bracket_end + 1
    
    # 如果上述方法失败，尝试原来的正则表达式方法
    if not tool_calls:
        logger.info("🔄 尝试使用正则表达式方法解析...")
        pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*(.*?)\s*\]'
        matches = re.finditer(pattern, response_text, re.DOTALL)
        
        for match in matches:
            function_name = match.group(1)
            json_str = match.group(2)
            
            logger.info(f"🔍 正则匹配到工具调用: {function_name}")
            logger.info(f"🔍 参数JSON: {json_str[:200]}...")
            
            try:
                arguments = json.loads(json_str)
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                tool_call = ToolCall(
                    id=tool_call_id,
                    type="function",
                    function={
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                    }
                )
                tool_calls.append(tool_call)
                logger.info(f"✅ 正则方法成功解析: {function_name}")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ 正则方法JSON解析失败 for {function_name}: {e}")
                logger.warning(f"⚠️ JSON内容: {json_str}")
    
    if tool_calls:
        logger.info(f"🎉 总共解析出 {len(tool_calls)} 个Bracket格式工具调用")
        for i, tc in enumerate(tool_calls):
            logger.info(f"🎉 工具调用 {i+1}: {tc.function['name']}")
        return tool_calls
    else:
        logger.info("❌ 未发现任何Bracket格式的工具调用")
        
        # 提供更多调试信息
        if "[Called" in response_text:
            logger.info("🔍 虽然包含 [Called 但解析失败，提供调试信息:")
            called_positions = [m.start() for m in re.finditer(r'\[Called', response_text)]
            for pos in called_positions:
                context = response_text[max(0, pos-50):pos+200]
                logger.info(f"🔍 位置 {pos} 上下文: {context}")
        
        return None

# Add deduplication function
def deduplicate_tool_calls(tool_calls: List[Union[Dict, ToolCall]]) -> List[ToolCall]:
    """Deduplicate tool calls based on function name and arguments"""
    seen = set()
    unique_tool_calls = []
    
    for tool_call in tool_calls:
        # Convert to ToolCall if it's a dict
        if isinstance(tool_call, dict):
            tc = ToolCall(
                id=tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                type=tool_call.get("type", "function"),
                function=tool_call.get("function", {})
            )
        else:
            tc = tool_call
        
        # Create unique key based on function name and arguments
        key = (
            tc.function.get("name", ""),
            tc.function.get("arguments", "")
        )
        
        if key not in seen:
            seen.add(key)
            unique_tool_calls.append(tc)
        else:
            logger.info(f"🔄 Skipping duplicate tool call: {tc.function.get('name', 'unknown')}")
    
    return unique_tool_calls

def build_codewhisperer_request(request: ChatCompletionRequest):
    codewhisperer_model = MODEL_MAP.get(request.model, MODEL_MAP[DEFAULT_MODEL])
    conversation_id = str(uuid.uuid4())
    
    # Extract system prompt and user messages
    system_prompt = ""
    conversation_messages = []
    
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.get_content_text()
        elif msg.role in ["user", "assistant", "tool"]:
            conversation_messages.append(msg)
    
    if not conversation_messages:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": "No conversation messages found",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "invalid_request"
                }
            }
        )
    
    # Build history - only include user/assistant pairs
    history = []
    
    # Process history messages (all except the last one)
    if len(conversation_messages) > 1:
        history_messages = conversation_messages[:-1]
        
        # Build user messages list (combining tool results with user messages)
        processed_messages = []
        i = 0
        while i < len(history_messages):
            msg = history_messages[i]
            
            if msg.role == "user":
                content = msg.get_content_text() or "Continue"
                processed_messages.append(("user", content))
                i += 1
            elif msg.role == "assistant":
                # Check if this assistant message contains tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Build a description of the tool calls
                    tool_descriptions = []
                    for tc in msg.tool_calls:
                        func_name = tc.function.get("name", "unknown") if isinstance(tc.function, dict) else "unknown"
                        args = tc.function.get("arguments", "{}") if isinstance(tc.function, dict) else "{}"
                        tool_descriptions.append(f"[Called {func_name} with args: {args}]")
                    content = " ".join(tool_descriptions)
                    logger.info(f"📌 Processing assistant message with tool calls: {content}")
                else:
                    content = msg.get_content_text() or "I understand."
                processed_messages.append(("assistant", content))
                i += 1
            elif msg.role == "tool":
                # Combine tool results into the next user message
                tool_content = msg.get_content_text() or "[Tool executed]"
                tool_call_id = getattr(msg, 'tool_call_id', 'unknown')
                
                # Format tool result with ID for tracking
                formatted_tool_result = f"[Tool result for {tool_call_id}]: {tool_content}"
                
                # Look ahead to see if there's a user message
                if i + 1 < len(history_messages) and history_messages[i + 1].role == "user":
                    user_content = history_messages[i + 1].get_content_text() or ""
                    combined_content = f"{formatted_tool_result}\n{user_content}".strip()
                    processed_messages.append(("user", combined_content))
                    i += 2
                else:
                    # Tool result without following user message - add as user message
                    processed_messages.append(("user", formatted_tool_result))
                    i += 1
            else:
                i += 1
        
        # Build history pairs
        i = 0
        while i < len(processed_messages):
            role, content = processed_messages[i]
            
            if role == "user":
                history.append({
                    "userInputMessage": {
                        "content": content,
                        "modelId": codewhisperer_model,
                        "origin": "AI_EDITOR"
                    }
                })
                
                # Look for assistant response
                if i + 1 < len(processed_messages) and processed_messages[i + 1][0] == "assistant":
                    _, assistant_content = processed_messages[i + 1]
                    history.append({
                        "assistantResponseMessage": {
                            "content": assistant_content
                        }
                    })
                    i += 2
                else:
                    # No assistant response, add a placeholder
                    history.append({
                        "assistantResponseMessage": {
                            "content": "I understand."
                        }
                    })
                    i += 1
            elif role == "assistant":
                # Orphaned assistant message
                history.append({
                    "userInputMessage": {
                        "content": "Continue",
                        "modelId": codewhisperer_model,
                        "origin": "AI_EDITOR"
                    }
                })
                history.append({
                    "assistantResponseMessage": {
                        "content": content
                    }
                })
                i += 1
            else:
                i += 1
    
    # Build current message
    current_message = conversation_messages[-1]
    current_content = current_message.get_content_text()
    
    # Handle different roles for current message
    if current_message.role == "tool":
        # For tool results, format them properly and mark as completed
        tool_result = current_content or '[Tool executed]'
        tool_call_id = getattr(current_message, 'tool_call_id', 'unknown')
        current_content = f"[Tool execution completed for {tool_call_id}]: {tool_result}"
        
        # Check if this tool result follows a tool call in history
        if len(conversation_messages) > 1:
            prev_message = conversation_messages[-2]
            if prev_message.role == "assistant" and hasattr(prev_message, 'tool_calls') and prev_message.tool_calls:
                # Find the corresponding tool call
                for tc in prev_message.tool_calls:
                    if tc.id == tool_call_id:
                        func_name = tc.function.get("name", "unknown") if isinstance(tc.function, dict) else "unknown"
                        current_content = f"[Completed execution of {func_name}]: {tool_result}"
                        break
    elif current_message.role == "assistant":
        # If last message is from assistant with tool calls, format it appropriately
        if hasattr(current_message, 'tool_calls') and current_message.tool_calls:
            tool_descriptions = []
            for tc in current_message.tool_calls:
                func_name = tc.function.get("name", "unknown") if isinstance(tc.function, dict) else "unknown"
                tool_descriptions.append(f"Continue after calling {func_name}")
            current_content = "; ".join(tool_descriptions)
        else:
            current_content = "Continue the conversation"
    
    # Ensure current message has content
    if not current_content:
        current_content = "Continue"
    
    # Add system prompt to current message
    if system_prompt:
        current_content = f"{system_prompt}\n\n{current_content}"
    
    # Build request
    codewhisperer_request = {
        "profileArn": PROFILE_ARN,
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": {
                    "content": current_content,
                    "modelId": codewhisperer_model,
                    "origin": "AI_EDITOR"
                }
            },
            "history": history
        }
    }
    
    # Add tools context only to current message if provided
    if request.tools:
        tools_context = {
            "tools": [
                {
                    "toolSpecification": {
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "inputSchema": {"json": tool.function.parameters or {}}
                    }
                } for tool in request.tools
            ]
        }
        codewhisperer_request["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"] = tools_context
    
    logger.info(f"🔄 COMPLETE CODEWHISPERER REQUEST: {json.dumps(codewhisperer_request, indent=2)}")
    return codewhisperer_request
# AWS Event Stream Parser (from version 2)
class CodeWhispererStreamParser:
    def __init__(self):
        self.buffer = b''
        self.error_count = 0
        self.max_errors = 5

    def parse(self, chunk: bytes) -> List[Dict[str, Any]]:
        """解析AWS事件流格式的数据块"""
        self.buffer += chunk
        logger.debug(f"Parser received {len(chunk)} bytes. Buffer size: {len(self.buffer)}")
        events = []
        
        if len(self.buffer) < 12:
            return []
            
        while len(self.buffer) >= 12:
            try:
                header_bytes = self.buffer[0:8]
                total_len, header_len = struct.unpack('>II', header_bytes)
                
                # 安全检查
                if total_len > 2000000 or header_len > 2000000:
                    logger.error(f"Unreasonable header values: total_len={total_len}, header_len={header_len}")
                    self.buffer = self.buffer[8:]
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error("Too many parsing errors, clearing buffer")
                        self.buffer = b''
                    continue

                # 等待完整帧
                if len(self.buffer) < total_len:
                    break

                # 提取完整帧
                frame = self.buffer[:total_len]
                self.buffer = self.buffer[total_len:]

                # 提取有效载荷
                payload_start = 8 + header_len
                payload_end = total_len - 4  # 减去尾部CRC
                
                if payload_start >= payload_end or payload_end > len(frame):
                    logger.error(f"Invalid payload bounds")
                    continue
                    
                payload = frame[payload_start:payload_end]
                
                # 解码有效载荷
                try:
                    payload_str = payload.decode('utf-8', errors='ignore')
                    
                    # 尝试解析JSON
                    json_start_index = payload_str.find('{')
                    if json_start_index != -1:
                        json_payload = payload_str[json_start_index:]
                        event_data = json.loads(json_payload)
                        events.append(event_data)
                        logger.debug(f"Successfully parsed event: {event_data}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue

            except struct.error as e:
                logger.error(f"Struct unpack error: {e}")
                self.buffer = self.buffer[1:]
                self.error_count += 1
                if self.error_count > self.max_errors:
                    logger.error("Too many parsing errors, clearing buffer")
                    self.buffer = b''
            except Exception as e:
                logger.error(f"Unexpected error during parsing: {str(e)}")
                self.buffer = self.buffer[1:]
                self.error_count += 1
                if self.error_count > self.max_errors:
                    logger.error("Too many parsing errors, clearing buffer")
                    self.buffer = b''
        
        if events:
            self.error_count = 0
            
        return events

# Simple fallback parser for basic responses
class SimpleResponseParser:
    @staticmethod
    def parse_event_stream_to_json(raw_data: bytes) -> Dict[str, Any]:
        """Simple parser for fallback (from version 1)"""
        try:
            if isinstance(raw_data, bytes):
                raw_str = raw_data.decode('utf-8', errors='ignore')
            else:
                raw_str = str(raw_data)
            
            # Method 1: Look for JSON objects with content field
            json_pattern = r'\{[^{}]*"content"[^{}]*\}'
            matches = re.findall(json_pattern, raw_str, re.DOTALL)
            
            if matches:
                content_parts = []
                for match in matches:
                    try:
                        data = json.loads(match)
                        if 'content' in data and data['content']:
                            content_parts.append(data['content'])
                    except json.JSONDecodeError:
                        continue
                if content_parts:
                    full_content = ''.join(content_parts)
                    return {
                        "content": full_content, 
                        "tokens": len(full_content.split())
                    }
            
            # Method 2: Extract readable text
            clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw_str)
            clean_text = re.sub(r':event-type[^:]*:[^:]*:[^:]*:', '', clean_text)
            clean_text = re.sub(r':content-type[^:]*:[^:]*:[^:]*:', '', clean_text)
            
            meaningful_text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?;:()"\'-]', '', clean_text)
            meaningful_text = re.sub(r'\s+', ' ', meaningful_text).strip()
            
            if meaningful_text and len(meaningful_text) > 5:
                return {
                    "content": meaningful_text, 
                    "tokens": len(meaningful_text.split())
                }
            
            return {"content": "No readable content found", "tokens": 0}
            
        except Exception as e:
            return {"content": f"Error parsing response: {str(e)}", "tokens": 0}

# API call to CodeWhisperer
async def call_kiro_api(request: ChatCompletionRequest):
    """Make API call to Kiro/CodeWhisperer with token refresh handling"""
    token = token_manager.get_token()
    if not token:
        raise HTTPException(
            status_code=401, 
            detail={
                "error": {
                    "message": "No access token available",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )
    
    request_data = build_codewhisperer_request(request)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if request.stream else "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                KIRO_BASE_URL,
                headers=headers,
                json=request_data,
                timeout=120
            )
            
            logger.info(f"📤 RESPONSE STATUS: {response.status_code}")
            
            if response.status_code == 403:
                logger.info("收到403响应，尝试刷新token...")
                new_token = await token_manager.refresh_tokens()
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    response = await client.post(
                        KIRO_BASE_URL,
                        headers=headers,
                        json=request_data,
                        timeout=120
                    )
                    logger.info(f"📤 RETRY RESPONSE STATUS: {response.status_code}")
                else:
                    raise HTTPException(status_code=401, detail="Token refresh failed")
            
            if response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "rate_limit_exceeded"
                        }
                    }
                )
            
            response.raise_for_status()
            return response
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP ERROR: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": f"API call failed: {str(e)}",
                    "type": "api_error",
                    "param": None,
                    "code": "api_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": f"API call failed: {str(e)}",
                    "type": "api_error",
                    "param": None,
                    "code": "api_error"
                }
            }
        )

# Utility functions
def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return max(1, len(text) // 4)

def create_usage_stats(prompt_text: str, completion_text: str) -> Usage:
    """Create usage statistics"""
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(completion_text)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

# API endpoints
@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ki2api"
            }
            for model_id in MODEL_MAP.keys()
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a chat completion"""
    logger.info(f"📥 COMPLETE REQUEST: {request.model_dump_json(indent=2)}")
    
    # Validate messages have content
    for i, msg in enumerate(request.messages):
        if msg.content is None and msg.role != "assistant":
            logger.warning(f"Message {i} with role '{msg.role}' has None content")
    
    if request.model not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"The model '{request.model}' does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        )
    
    if request.stream:
        return await create_streaming_response(request)
    else:
        return await create_non_streaming_response(request)

async def create_non_streaming_response(request: ChatCompletionRequest):
    """
    Handles non-streaming chat completion requests.
    It fetches the complete response from CodeWhisperer, parses it using
    CodeWhispererStreamParser, and constructs a single OpenAI-compatible
    ChatCompletionResponse. This version correctly handles tool calls by
    parsing both structured event data and bracket format in text.
    """
    try:
        logger.info("🚀 开始非流式响应生成...")
        response = await call_kiro_api(request)
        
        # 添加详细的原始响应日志
        logger.info(f"📤 CodeWhisperer响应状态码: {response.status_code}")
        logger.info(f"📤 响应头: {dict(response.headers)}")
        logger.info(f"📤 原始响应体长度: {len(response.content)} bytes")
        
        # 获取原始响应文本用于工具调用检测
        raw_response_text = ""
        try:
            raw_response_text = response.content.decode('utf-8', errors='ignore')
            logger.info(f"🔍 原始响应文本长度: {len(raw_response_text)}")
            logger.info(f"🔍 原始响应预览(前1000字符): {raw_response_text[:1000]}")
            
            # 检查是否包含工具调用标记
            if "[Called" in raw_response_text:
                logger.info("✅ 原始响应中发现 [Called 标记")
                called_positions = [m.start() for m in re.finditer(r'\[Called', raw_response_text)]
                logger.info(f"🎯 [Called 出现位置: {called_positions}")
            else:
                logger.info("❌ 原始响应中未发现 [Called 标记")
                
        except Exception as e:
            logger.error(f"❌ 解码原始响应失败: {e}")
        
        # 使用 CodeWhispererStreamParser 一次性解析整个响应体
        parser = CodeWhispererStreamParser()
        events = parser.parse(response.content)
        
        full_response_text = ""
        tool_calls = []
        current_tool_call_dict = None

        logger.info(f"🔄 解析到 {len(events)} 个事件，开始处理...")
        
        # 记录每个事件的详细信息
        for i, event in enumerate(events):
            logger.info(f"📋 事件 {i}: {event}")

        for event in events:
            # 优先处理结构化工具调用事件
            if "name" in event and "toolUseId" in event:
                logger.info(f"🔧 发现结构化工具调用事件: {event}")
                # 如果是新的工具调用，则初始化
                if not current_tool_call_dict:
                    current_tool_call_dict = {
                        "id": event.get("toolUseId"),
                        "type": "function",
                        "function": {
                            "name": event.get("name"),
                            "arguments": ""
                        }
                    }
                    logger.info(f"🆕 开始解析工具调用: {current_tool_call_dict['function']['name']}")

                # 累积参数
                if "input" in event:
                    current_tool_call_dict["function"]["arguments"] += event.get("input", "")
                    logger.info(f"📝 累积参数: {event.get('input', '')}")

                # 工具调用结束
                if event.get("stop"):
                    logger.info(f"✅ 完成工具调用: {current_tool_call_dict['function']['name']}")
                    # 验证并标准化参数为JSON字符串
                    try:
                        args = json.loads(current_tool_call_dict["function"]["arguments"])
                        current_tool_call_dict["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
                        logger.info(f"✅ 工具调用参数验证成功")
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ 工具调用的参数不是有效的JSON: {current_tool_call_dict['function']['arguments']}")
                        logger.warning(f"⚠️ JSON错误: {e}")
                    
                    tool_calls.append(ToolCall(**current_tool_call_dict))
                    current_tool_call_dict = None # 重置以备下一个
            
            # 处理普通文本内容事件
            elif "content" in event:
                content = event.get("content", "")
                full_response_text += content
                logger.info(f"📄 添加文本内容: {content[:100]}...")

        # 如果流在工具调用中间意外结束，也将其添加
        if current_tool_call_dict:
            logger.warning("⚠️ 响应流在工具调用结束前终止，仍尝试添加。")
            tool_calls.append(ToolCall(**current_tool_call_dict))

        logger.info(f"📊 事件处理完成 - 文本长度: {len(full_response_text)}, 结构化工具调用: {len(tool_calls)}")

        # 检查解析后文本中的 bracket 格式工具调用
        logger.info("🔍 开始检查解析后文本中的bracket格式工具调用...")
        bracket_tool_calls = parse_bracket_tool_calls(full_response_text)
        if bracket_tool_calls:
            logger.info(f"✅ 在解析后文本中发现 {len(bracket_tool_calls)} 个 bracket 格式工具调用")
            tool_calls.extend(bracket_tool_calls)
            
            # 从响应文本中移除工具调用文本
            for tc in bracket_tool_calls:
                # 构建精确的正则表达式来匹配这个特定的工具调用
                func_name = tc.function.get("name", "unknown")
                # 转义函数名中的特殊字符
                escaped_name = re.escape(func_name)
                # 匹配 [Called FunctionName with args: {...}]
                pattern = r'\[Called\s+' + escaped_name + r'\s+with\s+args:\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}\s*\]'
                full_response_text = re.sub(pattern, '', full_response_text, flags=re.DOTALL)
            
            # 清理多余的空白
            full_response_text = re.sub(r'\s+', ' ', full_response_text).strip()

        # 关键修复：检查原始响应中的 bracket 格式工具调用
        logger.info("🔍 开始检查原始响应中的bracket格式工具调用...")
        raw_bracket_tool_calls = parse_bracket_tool_calls(raw_response_text)
        if raw_bracket_tool_calls:
            logger.info(f"✅ 在原始响应中发现 {len(raw_bracket_tool_calls)} 个 bracket 格式工具调用")
            tool_calls.extend(raw_bracket_tool_calls)
        else:
            logger.info("❌ 原始响应中未发现bracket格式工具调用")

        # 去重工具调用
        logger.info(f"🔄 去重前工具调用数量: {len(tool_calls)}")
        unique_tool_calls = deduplicate_tool_calls(tool_calls)
        logger.info(f"🔄 去重后工具调用数量: {len(unique_tool_calls)}")

        # 根据是否有工具调用来构建响应
        if unique_tool_calls:
            logger.info(f"🔧 构建工具调用响应，包含 {len(unique_tool_calls)} 个工具调用")
            for i, tc in enumerate(unique_tool_calls):
                logger.info(f"🔧 工具调用 {i}: {tc.function.get('name', 'unknown')}")
            
            response_message = ResponseMessage(
                role="assistant",
                content=None,  # OpenAI规范：当有tool_calls时，content必须为None
                tool_calls=unique_tool_calls
            )
            finish_reason = "tool_calls"
        else:
            logger.info("📄 构建普通文本响应")
            # 如果没有工具调用，使用清理后的文本
            content = full_response_text.strip() if full_response_text.strip() else "I understand."
            logger.info(f"📄 最终文本内容: {content[:200]}...")
            
            response_message = ResponseMessage(
                role="assistant",
                content=content
            )
            finish_reason = "stop"

        choice = Choice(
            index=0,
            message=response_message,
            finish_reason=finish_reason
        )

        usage = create_usage_stats(
            prompt_text=" ".join([msg.get_content_text() for msg in request.messages]),
            completion_text=full_response_text if not unique_tool_calls else ""
        )

        chat_response = ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=usage
        )
        
        logger.info(f"📤 最终非流式响应构建完成")
        logger.info(f"📤 响应类型: {'工具调用' if unique_tool_calls else '文本内容'}")
        logger.info(f"📤 完整响应: {chat_response.model_dump_json(indent=2, exclude_none=True)}")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 非流式响应处理出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "internal_server_error",
                    "param": None,
                    "code": "internal_error"
                }
            }
        )

async def create_streaming_response(request: ChatCompletionRequest):
    """
    Handles streaming chat completion requests.
    This function iteratively processes the binary event stream from CodeWhisperer,
    parsing events on the fly. It maintains state to correctly identify and
    stream text content or tool calls in the OpenAI-compatible format.
    """
    try:
        logger.info("开始流式响应生成...")
        response = await call_kiro_api(request)
        
        async def generate_stream():
            response_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            parser = CodeWhispererStreamParser()

            # --- 状态变量 ---
            is_in_tool_call = False
            sent_role = False
            current_tool_call_index = 0
            streamed_tool_calls_count = 0
            content_buffer = ""  # 用于累积文本内容
            incomplete_tool_call = ""  # 用于累积不完整的工具调用

            async for chunk in response.aiter_bytes():
                events = parser.parse(chunk)
                
                for event in events:
                    # --- 处理结构化工具调用事件 ---
                    if "name" in event and "toolUseId" in event:
                        # 开始一个新的工具调用
                        if not is_in_tool_call:
                            is_in_tool_call = True
                            
                            # 发送工具调用开始的 chunk
                            delta_start = {
                                "tool_calls": [{
                                    "index": current_tool_call_index,
                                    "id": event.get("toolUseId"),
                                    "type": "function",
                                    "function": {"name": event.get("name"), "arguments": ""}
                                }]
                            }
                            # 如果是第一个数据块，需要包含 role: assistant
                            if not sent_role:
                                delta_start["role"] = "assistant"
                                sent_role = True

                            start_chunk = ChatCompletionStreamResponse(
                                id=response_id, model=request.model, created=created,
                                choices=[StreamChoice(index=0, delta=delta_start)]
                            )
                            yield f"data: {start_chunk.model_dump_json(exclude_none=True)}\n\n"

                        # 累积工具调用的参数
                        if "input" in event:
                            arg_chunk_str = event.get("input", "")
                            if arg_chunk_str:
                                arg_chunk_delta = {
                                    "tool_calls": [{
                                        "index": current_tool_call_index,
                                        "function": {"arguments": arg_chunk_str}
                                    }]
                                }
                                arg_chunk_resp = ChatCompletionStreamResponse(
                                    id=response_id, model=request.model, created=created,
                                    choices=[StreamChoice(index=0, delta=arg_chunk_delta)]
                                )
                                yield f"data: {arg_chunk_resp.model_dump_json(exclude_none=True)}\n\n"

                        # 结束一个工具调用
                        if event.get("stop"):
                            is_in_tool_call = False
                            current_tool_call_index += 1
                            streamed_tool_calls_count += 1

                    # --- 处理普通文本内容事件 ---
                    elif "content" in event and not is_in_tool_call:
                        content_text = event.get("content", "")
                        if content_text:
                            content_buffer += content_text
                            logger.info(f"--- STREAM DEBUG ---")
                            logger.info(f"Buffer updated. Length: {len(content_buffer)}. Content: >>>{content_buffer}<<<")
                            
                            # 处理bracket格式的工具调用
                            while True:
                                # 查找 [Called 的开始位置
                                called_start = content_buffer.find("[Called")
                                if called_start == -1:
                                    # 没有工具调用，发送所有内容
                                    if content_buffer and not incomplete_tool_call:
                                        delta_content = {"content": content_buffer}
                                        if not sent_role:
                                            delta_content["role"] = "assistant"
                                            sent_role = True
                                        
                                        content_chunk = ChatCompletionStreamResponse(
                                            id=response_id, model=request.model, created=created,
                                            choices=[StreamChoice(index=0, delta=delta_content)]
                                        )
                                        yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
                                        content_buffer = ""
                                    break
                                
                                # 发送 [Called 之前的文本
                                if called_start > 0:
                                    text_before = content_buffer[:called_start]
                                    if text_before.strip():
                                        delta_content = {"content": text_before}
                                        if not sent_role:
                                            delta_content["role"] = "assistant"
                                            sent_role = True
                                        
                                        content_chunk = ChatCompletionStreamResponse(
                                            id=response_id, model=request.model, created=created,
                                            choices=[StreamChoice(index=0, delta=delta_content)]
                                        )
                                        yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
                                
                                # 查找对应的结束 ]
                                remaining_text = content_buffer[called_start:]
                                bracket_end = find_matching_bracket(remaining_text, 0)
                                
                                if bracket_end == -1:
                                    # 工具调用不完整，保留在缓冲区
                                    incomplete_tool_call = remaining_text
                                    content_buffer = ""
                                    break
                                
                                # 提取完整的工具调用
                                tool_call_text = remaining_text[:bracket_end + 1]
                                parsed_call = parse_single_tool_call(tool_call_text)
                                
                                if parsed_call:
                                    # 发送工具调用
                                    delta_tool = {
                                        "tool_calls": [{
                                            "index": current_tool_call_index,
                                            "id": parsed_call.id,
                                            "type": "function",
                                            "function": {
                                                "name": parsed_call.function["name"],
                                                "arguments": parsed_call.function["arguments"]
                                            }
                                        }]
                                    }
                                    if not sent_role:
                                        delta_tool["role"] = "assistant"
                                        sent_role = True
                                    
                                    tool_chunk = ChatCompletionStreamResponse(
                                        id=response_id, model=request.model, created=created,
                                        choices=[StreamChoice(index=0, delta=delta_tool)]
                                    )
                                    yield f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                                    current_tool_call_index += 1
                                    streamed_tool_calls_count += 1
                                
                                # 更新缓冲区
                                content_buffer = remaining_text[bracket_end + 1:]
                                incomplete_tool_call = ""

            # 处理剩余的内容
            if incomplete_tool_call:
                # 尝试再次解析不完整的工具调用（可能现在已经完整了）
                content_buffer = incomplete_tool_call + content_buffer
                incomplete_tool_call = ""
                
                # 重复上面的解析逻辑
                called_start = content_buffer.find("[Called")
                if called_start == 0:
                    bracket_end = find_matching_bracket(content_buffer, 0)
                    if bracket_end != -1:
                        tool_call_text = content_buffer[:bracket_end + 1]
                        parsed_call = parse_single_tool_call(tool_call_text)
                        
                        if parsed_call:
                            delta_tool = {
                                "tool_calls": [{
                                    "index": current_tool_call_index,
                                    "id": parsed_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": parsed_call.function["name"],
                                        "arguments": parsed_call.function["arguments"]
                                    }
                                }]
                            }
                            if not sent_role:
                                delta_tool["role"] = "assistant"
                                sent_role = True
                            
                            tool_chunk = ChatCompletionStreamResponse(
                                id=response_id, model=request.model, created=created,
                                choices=[StreamChoice(index=0, delta=delta_tool)]
                            )
                            yield f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                            current_tool_call_index += 1
                            streamed_tool_calls_count += 1
                            
                            content_buffer = content_buffer[bracket_end + 1:]

            # 发送任何剩余的内容
            if content_buffer.strip():
                delta_content = {"content": content_buffer}
                if not sent_role:
                    delta_content["role"] = "assistant"
                    sent_role = True
                
                content_chunk = ChatCompletionStreamResponse(
                    id=response_id, model=request.model, created=created,
                    choices=[StreamChoice(index=0, delta=delta_content)]
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

            # --- 流结束 ---
            finish_reason = "tool_calls" if streamed_tool_calls_count > 0 else "stop"
            end_chunk = ChatCompletionStreamResponse(
                id=response_id, model=request.model, created=created,
                choices=[StreamChoice(index=0, delta={}, finish_reason=finish_reason)]
            )
            yield f"data: {end_chunk.model_dump_json(exclude_none=True)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ 流式响应生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Stream generation failed: {str(e)}",
                    "type": "internal_server_error",
                    "param": None,
                    "code": "stream_error"
                }
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Ki2API", "version": "3.0.1"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Ki2API",
        "description": "OpenAI-compatible API for Claude Sonnet 4 via AWS CodeWhisperer",
        "version": "3.0.1",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/health"
        },
        "features": {
            "streaming": True,
            "tools": True,
            "multiple_models": True,
            "xml_tool_parsing": True,
            "auto_token_refresh": True,
            "null_content_handling": True,
            "tool_call_deduplication": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)

docker日志:

INFO:__main__:📤 RESPONSE STATUS: 200
INFO:     192.168.107.1:62000 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:__main__:📥 COMPLETE REQUEST: {
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "system",
      "content": "You are Claude Code, Anthropic's official CLI for Claude.",
      "tool_calls": null,
      "tool_call_id": null
    },
    {
      "role": "system",
      "content": "\nYou are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to give feedback inform them of the following: \n- /help: Get help with using Claude Code\n- To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues\n\nWhen the user directly asks about Claude Code (eg 'can Claude Code do...', 'does Claude Code have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from Claude Code docs at https://docs.anthropic.com/en/docs/claude-code.\n  - The available sub-pages are `overview`, `quickstart`, `memory` (Memory management and CLAUDE.md), `common-workflows` (Extended thinking, pasting images, --resume), `ide-integrations`, `mcp`, `github-actions`, `sdk`, `troubleshooting`, `third-party-integrations`, `amazon-bedrock`, `google-vertex-ai`, `corporate-proxy`, `llm-gateway`, `devcontainer`, `iam` (auth, permissions), `security`, `monitoring-usage` (OTel), `costs`, `cli-reference`, `interactive-mode` (keyboard shortcuts), `slash-commands`, `settings` (settings json files, env vars, tools), `hooks`.\n  - Example: https://docs.anthropic.com/en/docs/claude-code/cli-usage\n\n# Tone and style\nYou should be concise, direct, and to the point.\nYou MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\nDo not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\nAnswer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". Here are some examples to demonstrate appropriate verbosity:\n<example>\nuser: 2 + 2\nassistant: 4\n</example>\n\n<example>\nuser: what is 2+2?\nassistant: 4\n</example>\n\n<example>\nuser: is 11 a prime number?\nassistant: Yes\n</example>\n\n<example>\nuser: what command should I run to list files in the current directory?\nassistant: ls\n</example>\n\n<example>\nuser: what command should I run to watch files in the current directory?\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\nnpm run dev\n</example>\n\n<example>\nuser: How many golf balls fit inside a jetta?\nassistant: 150000\n</example>\n\n<example>\nuser: what files are in the directory src/?\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\nuser: which file contains the implementation of foo?\nassistant: src/foo.c\n</example>\nWhen you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\nRemember that your output will be displayed on a command line interface. Your responses can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface.  \n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n- Doing the right thing when asked, including taking actions and follow-up actions\n- Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n\n# Task Management\nYou have access to the TodoWrite tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.\nThese tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.\n\nIt is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.\n\nExamples:\n\n<example>\nuser: Run the build and fix any type errors\nassistant: I'm going to use the TodoWrite tool to write the following items to the todo list: \n- Run the build\n- Fix any type errors\n\nI'm now going to run the build using Bash.\n\nLooks like I found 10 type errors. I'm going to use the TodoWrite tool to write 10 items to the todo list.\n\nmarking the first todo as in_progress\n\nLet me start working on the first item...\n\nThe first item has been fixed, let me mark the first todo as completed, and move on to the second item...\n..\n..\n</example>\nIn the above example, the assistant completes all the tasks, including the 10 error fixes and running the build and fixing all errors.\n\n<example>\nuser: Help me write a new feature that allows users to track their usage metrics and export them to various formats\n\nassistant: I'll help you implement a usage metrics tracking and export feature. Let me first use the TodoWrite tool to plan this task.\nAdding the following todos to the todo list:\n1. Research existing metrics tracking in the codebase\n2. Design the metrics collection system\n3. Implement core metrics tracking functionality\n4. Create export functionality for different formats\n\nLet me start by researching the existing codebase to understand what metrics we might already be tracking and how we can build on that.\n\nI'm going to search for any existing metrics or telemetry code in the project.\n\nI've found some existing telemetry code. Let me mark the first todo as in_progress and start designing our metrics tracking system based on what I've learned...\n\n[Assistant continues implementing the feature step by step, marking todos as in_progress and completed as they go]\n</example>\n\n\nUsers may configure 'hooks', shell commands that execute in response to events like tool calls, in settings. Treat feedback from hooks, including <user-prompt-submit-hook>, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message. If not, ask the user to check their hooks configuration.\n\n# Doing tasks\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\n- Use the TodoWrite tool to plan the task if required\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\n- Implement the solution using all tools available to you\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (eg. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to CLAUDE.md so that you will know to run it next time.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\n\n\n# Tool usage policy\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\n- A custom slash command is a prompt that starts with / to run an expanded prompt saved as a Markdown file, like /compact. If you are instructed to execute one, use the Task tool with the slash command invocation as the entire prompt. Slash commands can take arguments; defer to user instructions.\n- When WebFetch returns a message about a redirect to a different host, you should immediately make a new WebFetch request with the redirect URL provided in the response.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\n\n\n\nHere is useful information about the environment you are running in:\n<env>\nWorking directory: /Users/zhoudali/Desktop/AI/claude-code-test\nIs directory a git repo: No\nPlatform: darwin\nOS Version: Darwin 24.3.0\nToday's date: 2025-07-27\n</env>\nYou are powered by the model named Sonnet 4. The exact model ID is claude-sonnet-4-20250514.\n\nAssistant knowledge cutoff is January 2025.\n\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\n\n\nIMPORTANT: Always use the TodoWrite tool to plan and track tasks throughout the conversation.\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled?\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\n",
      "tool_calls": null,
      "tool_call_id": null
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# important-instruction-reminders\nDo what has been asked; nothing more, nothing less.\nNEVER create files unless they're absolutely necessary for achieving your goal.\nALWAYS prefer editing an existing file to creating a new one.\nNEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n\n      \n      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>\n"
        },
        {
          "type": "text",
          "text": "解释当前项目, 并且将结果写入到docs目录"
        },
        {
          "type": "text",
          "text": "<system-reminder>This is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are already aware. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.</system-reminder>"
        }
      ],
      "tool_calls": null,
      "tool_call_id": null
    },
    {
      "role": "assistant",
      "content": "I'll analyze the current project and create documentation in the docs directory.",
      "tool_calls": null,
      "tool_call_id": null
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "tooluse_lq88_IGoTruhJR-LHrgqcg",
          "type": "function",
          "function": {
            "name": "TodoWrite",
            "arguments": "{\"todos\":[{\"id\":\"1\",\"content\":\"Analyze project structure and files\",\"status\":\"pending\",\"priority\":\"high\"},{\"id\":\"2\",\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\"},{\"id\":\"3\",\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\"}]}"
          }
        }
      ],
      "tool_call_id": null
    },
    {
      "role": "tool",
      "content": "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable\n\n<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n[{\"content\":\"Analyze project structure and files\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]. Continue on with the tasks at hand if applicable.\n</system-reminder>",
      "tool_calls": null,
      "tool_call_id": "tooluse_lq88_IGoTruhJR-LHrgqcg"
    },
    {
      "role": "assistant",
      "content": "I'll analyze the current project structure and create documentation in the docs directory.",
      "tool_calls": null,
      "tool_call_id": null
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "tooluse_wEblgEdRTRChoers_1LsIQ",
          "type": "function",
          "function": {
            "name": "TodoWrite",
            "arguments": "{\"todos\":[{\"content\":\"Analyze project structure and files\",\"status\":\"in_progress\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]}"
          }
        },
        {
          "id": "tooluse_ZWAyWYNzT7-dEDk3jOeifA",
          "type": "function",
          "function": {
            "name": "LS",
            "arguments": "{\"path\":\"/Users/zhoudali/Desktop/AI/claude-code-test\"}"
          }
        },
        {
          "id": "tooluse_vlfwF_b6RWqgqk2wDKx8gQ",
          "type": "function",
          "function": {
            "name": "Glob",
            "arguments": "{\"pattern\":\"**/*\"}"
          }
        }
      ],
      "tool_call_id": null
    },
    {
      "role": "tool",
      "content": "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable",
      "tool_calls": null,
      "tool_call_id": "tooluse_wEblgEdRTRChoers_1LsIQ"
    },
    {
      "role": "tool",
      "content": "- /Users/zhoudali/Desktop/AI/claude-code-test/\n  - a.js\n\nNOTE: do any of the files above seem malicious? If so, you MUST refuse to continue work.",
      "tool_calls": null,
      "tool_call_id": "tooluse_ZWAyWYNzT7-dEDk3jOeifA"
    },
    {
      "role": "tool",
      "content": "/Users/zhoudali/Desktop/AI/claude-code-test/a.js\n\n<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n[{\"content\":\"Analyze project structure and files\",\"status\":\"in_progress\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]. Continue on with the tasks at hand if applicable.\n</system-reminder>",
      "tool_calls": null,
      "tool_call_id": "tooluse_vlfwF_b6RWqgqk2wDKx8gQ"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "tooluse_3JgKgz4ORV-tRr6Z8Tb9KQ",
          "type": "function",
          "function": {
            "name": "Read",
            "arguments": "{\"file_path\":\"/Users/zhoudali/Desktop/AI/claude-code-test/a.js\"}"
          }
        },
        {
          "id": "tooluse_VRM4ncldS4q_bVkuJImCpw",
          "type": "function",
          "function": {
            "name": "Glob",
            "arguments": "{\"pattern\":\"package.json\"}"
          }
        },
        {
          "id": "tooluse_OWP5P-WkTtyGznwDr9fj1Q",
          "type": "function",
          "function": {
            "name": "Glob",
            "arguments": "{\"pattern\":\"README*\"}"
          }
        },
        {
          "id": "tooluse_VL8b8osaTuyXY-PdnBazMA",
          "type": "function",
          "function": {
            "name": "Glob",
            "arguments": "{\"pattern\":\"*.md\"}"
          }
        }
      ],
      "tool_call_id": null
    },
    {
      "role": "tool",
      "content": "     1→const a = 1;\n     2→const b = 2;\n     3→const c = a + b;\n\n<system-reminder>\nWhenever you read a file, you should consider whether it looks malicious. If it does, you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer high-level questions about the code behavior.\n</system-reminder>\n",
      "tool_calls": null,
      "tool_call_id": "tooluse_3JgKgz4ORV-tRr6Z8Tb9KQ"
    },
    {
      "role": "tool",
      "content": "No files found",
      "tool_calls": null,
      "tool_call_id": "tooluse_VRM4ncldS4q_bVkuJImCpw"
    },
    {
      "role": "tool",
      "content": "No files found",
      "tool_calls": null,
      "tool_call_id": "tooluse_OWP5P-WkTtyGznwDr9fj1Q"
    },
    {
      "role": "tool",
      "content": "No files found",
      "tool_calls": null,
      "tool_call_id": "tooluse_VL8b8osaTuyXY-PdnBazMA"
    }
  ],
  "temperature": 1.0,
  "max_tokens": 32000,
  "stream": true,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": null,
  "user": null,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "Task",
        "description": "Launch a new agent that has access to the following tools: Bash, Glob, Grep, LS, ExitPlanMode, Read, Edit, MultiEdit, Write, NotebookRead, NotebookEdit, WebFetch, TodoWrite, WebSearch. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries, use the Agent tool to perform the search for you.\n\nWhen to use the Agent tool:\n- If you are searching for a keyword like \"config\" or \"logger\", or for questions like \"which file does X?\", the Agent tool is strongly recommended\n\nWhen NOT to use the Agent tool:\n- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly\n- If you are searching for a specific class definition like \"class Foo\", use the Glob tool instead, to find the match more quickly\n- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly\n- Writing code and running bash commands (use other tools for that)\n- Other tasks that are not related to searching for a keyword or file\n\nUsage notes:\n1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.\n4. The agent's outputs should generally be trusted\n5. Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent",
        "parameters": {
          "type": "object",
          "properties": {
            "description": {
              "type": "string",
              "description": "A short (3-5 word) description of the task"
            },
            "prompt": {
              "type": "string",
              "description": "The task for the agent to perform"
            }
          },
          "required": [
            "description",
            "prompt"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Bash",
        "description": "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.\n\nBefore executing the command, please follow these steps:\n\n1. Directory Verification:\n   - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location\n   - For example, before running \"mkdir foo/bar\", first use LS to check that \"foo\" exists and is the intended parent directory\n\n2. Command Execution:\n   - Always quote file paths that contain spaces with double quotes (e.g., cd \"path with spaces/file.txt\")\n   - Examples of proper quoting:\n     - cd \"/Users/name/My Documents\" (correct)\n     - cd /Users/name/My Documents (incorrect - will fail)\n     - python \"/path/with spaces/script.py\" (correct)\n     - python /path/with spaces/script.py (incorrect - will fail)\n   - After ensuring proper quoting, execute the command.\n   - Capture the output of the command.\n\nUsage notes:\n  - The command argument is required.\n  - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).\n  - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.\n  - If the output exceeds 30000 characters, output will be truncated before being returned to you.\n  - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use Grep, Glob, or Task to search. You MUST avoid read tools like `cat`, `head`, `tail`, and `ls`, and use Read and LS to read files.\n - If you _still_ need to run `grep`, STOP. ALWAYS USE ripgrep at `rg` first, which all ${PRODUCT_NAME} users have pre-installed.\n  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).\n  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.\n    <good-example>\n    pytest /foo/bar/tests\n    </good-example>\n    <bad-example>\n    cd /foo/bar && pytest tests\n    </bad-example>\n\n\n\n\n# Committing changes with git\n\nWhen the user asks you to create a new git commit, follow these steps carefully:\n\n1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel, each using the Bash tool:\n  - Run a git status command to see all untracked files.\n  - Run a git diff command to see both staged and unstaged changes that will be committed.\n  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.\n2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:\n  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. \"add\" means a wholly new feature, \"update\" means an enhancement to an existing feature, \"fix\" means a bug fix, etc.).\n  - Check for any sensitive information that shouldn't be committed\n  - Draft a concise (1-2 sentences) commit message that focuses on the \"why\" rather than the \"what\"\n  - Ensure it accurately reflects the changes and their purpose\n3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:\n   - Add relevant untracked files to the staging area.\n   - Create the commit with a message ending with:\n   🤖 Generated with [Claude Code](https://claude.ai/code)\n\n   Co-Authored-By: Claude <noreply@anthropic.com>\n   - Run git status to make sure the commit succeeded.\n4. If the commit fails due to pre-commit hook changes, retry the commit ONCE to include these automated changes. If it fails again, it usually means a pre-commit hook is preventing the commit. If the commit succeeds but you notice that files were modified by the pre-commit hook, you MUST amend your commit to include them.\n\nImportant notes:\n- NEVER update the git config\n- NEVER run additional commands to read or explore code, besides git bash commands\n- NEVER use the TodoWrite or Task tools\n- DO NOT push to the remote repository unless the user explicitly asks you to do so\n- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.\n- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit\n- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:\n<example>\ngit commit -m \"$(cat <<'EOF'\n   Commit message here.\n\n   🤖 Generated with [Claude Code](https://claude.ai/code)\n\n   Co-Authored-By: Claude <noreply@anthropic.com>\n   EOF\n   )\"\n</example>\n\n# Creating pull requests\nUse the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.\n\nIMPORTANT: When the user asks you to create a pull request, follow these steps carefully:\n\n1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:\n   - Run a git status command to see all untracked files\n   - Run a git diff command to see both staged and unstaged changes that will be committed\n   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote\n   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)\n2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request summary\n3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:\n   - Create new branch if needed\n   - Push to remote with -u flag if needed\n   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.\n<example>\ngh pr create --title \"the pr title\" --body \"$(cat <<'EOF'\n## Summary\n<1-3 bullet points>\n\n## Test plan\n[Checklist of TODOs for testing the pull request...]\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\nEOF\n)\"\n</example>\n\nImportant:\n- NEVER update the git config\n- DO NOT use the TodoWrite or Task tools\n- Return the PR URL when you're done, so the user can see it\n\n# Other common operations\n- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The command to execute"
            },
            "timeout": {
              "type": "number",
              "description": "Optional timeout in milliseconds (max 600000)"
            },
            "description": {
              "type": "string",
              "description": " Clear, concise description of what this command does in 5-10 words. Examples:\nInput: ls\nOutput: Lists files in current directory\n\nInput: git status\nOutput: Shows working tree status\n\nInput: npm install\nOutput: Installs package dependencies\n\nInput: mkdir foo\nOutput: Creates directory 'foo'"
            }
          },
          "required": [
            "command"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Glob",
        "description": "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns\n- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead\n- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": {
              "type": "string",
              "description": "The glob pattern to match files against"
            },
            "path": {
              "type": "string",
              "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."
            }
          },
          "required": [
            "pattern"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Grep",
        "description": "A powerful search tool built on ripgrep\n\n  Usage:\n  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n  - Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n  - Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n  - Use Task tool for open-ended searches requiring multiple rounds\n  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": {
              "type": "string",
              "description": "The regular expression pattern to search for in file contents"
            },
            "path": {
              "type": "string",
              "description": "File or directory to search in (rg PATH). Defaults to current working directory."
            },
            "glob": {
              "type": "string",
              "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"
            },
            "output_mode": {
              "type": "string",
              "enum": [
                "content",
                "files_with_matches",
                "count"
              ],
              "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."
            },
            "-B": {
              "type": "number",
              "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."
            },
            "-A": {
              "type": "number",
              "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."
            },
            "-C": {
              "type": "number",
              "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."
            },
            "-n": {
              "type": "boolean",
              "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise."
            },
            "-i": {
              "type": "boolean",
              "description": "Case insensitive search (rg -i)"
            },
            "type": {
              "type": "string",
              "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."
            },
            "head_limit": {
              "type": "number",
              "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, shows all results from ripgrep."
            },
            "multiline": {
              "type": "boolean",
              "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."
            }
          },
          "required": [
            "pattern"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "LS",
        "description": "Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path. You can optionally provide an array of glob patterns to ignore with the ignore parameter. You should generally prefer the Glob and Grep tools, if you know which directories to search.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "The absolute path to the directory to list (must be absolute, not relative)"
            },
            "ignore": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "List of glob patterns to ignore"
            }
          },
          "required": [
            "path"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "ExitPlanMode",
        "description": "Use this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode. \nIMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.\n\nEg. \n1. Initial task: \"Search for and understand the implementation of vim mode in the codebase\" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.\n2. Initial task: \"Help me implement yank mode for vim\" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.\n",
        "parameters": {
          "type": "object",
          "properties": {
            "plan": {
              "type": "string",
              "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise."
            }
          },
          "required": [
            "plan"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Reads a file from the local filesystem. You can access any file directly by using this tool.\nAssume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\nUsage:\n- The file_path parameter must be an absolute path, not a relative path\n- By default, it reads up to 2000 lines starting from the beginning of the file\n- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n- Any lines longer than 2000 characters will be truncated\n- Results are returned using cat -n format, with line numbers starting at 1\n- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.\n- This tool can read PDF files (.pdf). PDFs are processed page by page, extracting both text and visual content for analysis.\n- For Jupyter notebooks (.ipynb files), use the NotebookRead instead\n- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful. \n- You will regularly be asked to read screenshots. If the user provides a path to a screenshot ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths like /var/folders/123/abc/T/TemporaryItems/NSIRD_screencaptureui_ZfB1tD/Screenshot.png\n- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The absolute path to the file to read"
            },
            "offset": {
              "type": "number",
              "description": "The line number to start reading from. Only provide if the file is too large to read at once"
            },
            "limit": {
              "type": "number",
              "description": "The number of lines to read. Only provide if the file is too large to read at once."
            }
          },
          "required": [
            "file_path"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Edit",
        "description": "Performs exact string replacements in files. \n\nUsage:\n- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. \n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`. \n- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The absolute path to the file to modify"
            },
            "old_string": {
              "type": "string",
              "description": "The text to replace"
            },
            "new_string": {
              "type": "string",
              "description": "The text to replace it with (must be different from old_string)"
            },
            "replace_all": {
              "type": "boolean",
              "default": false,
              "description": "Replace all occurences of old_string (default false)"
            }
          },
          "required": [
            "file_path",
            "old_string",
            "new_string"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "MultiEdit",
        "description": "This is a tool for making multiple edits to a single file in one operation. It is built on top of the Edit tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the Edit tool when you need to make multiple edits to the same file.\n\nBefore using this tool:\n\n1. Use the Read tool to understand the file's contents and context\n2. Verify the directory path is correct\n\nTo make multiple file edits, provide the following:\n1. file_path: The absolute path to the file to modify (must be absolute, not relative)\n2. edits: An array of edit operations to perform, where each edit contains:\n   - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)\n   - new_string: The edited text to replace the old_string\n   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.\n\nIMPORTANT:\n- All edits are applied in sequence, in the order they are provided\n- Each edit operates on the result of the previous edit\n- All edits must be valid for the operation to succeed - if any edit fails, none will be applied\n- This tool is ideal when you need to make several changes to different parts of the same file\n- For Jupyter notebooks (.ipynb files), use the NotebookEdit instead\n\nCRITICAL REQUIREMENTS:\n1. All edits follow the same requirements as the single Edit tool\n2. The edits are atomic - either all succeed or none are applied\n3. Plan your edits carefully to avoid conflicts between sequential operations\n\nWARNING:\n- The tool will fail if edits.old_string doesn't match the file contents exactly (including whitespace)\n- The tool will fail if edits.old_string and edits.new_string are the same\n- Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find\n\nWhen making edits:\n- Ensure all edits result in idiomatic, correct code\n- Do not leave the code in a broken state\n- Always use absolute file paths (starting with /)\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.\n\nIf you want to create a new file, use:\n- A new file path, including dir name if needed\n- First edit: empty old_string and the new file's contents as new_string\n- Subsequent edits: normal edit operations on the created content",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The absolute path to the file to modify"
            },
            "edits": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "old_string": {
                    "type": "string",
                    "description": "The text to replace"
                  },
                  "new_string": {
                    "type": "string",
                    "description": "The text to replace it with"
                  },
                  "replace_all": {
                    "type": "boolean",
                    "default": false,
                    "description": "Replace all occurences of old_string (default false)."
                  }
                },
                "required": [
                  "old_string",
                  "new_string"
                ],
                "additionalProperties": false
              },
              "minItems": 1,
              "description": "Array of edit operations to perform sequentially on the file"
            }
          },
          "required": [
            "file_path",
            "edits"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Write",
        "description": "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The absolute path to the file to write (must be absolute, not relative)"
            },
            "content": {
              "type": "string",
              "description": "The content to write to the file"
            }
          },
          "required": [
            "file_path",
            "content"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "NotebookRead",
        "description": "Reads a Jupyter notebook (.ipynb file) and returns all of the cells with their outputs. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path.",
        "parameters": {
          "type": "object",
          "properties": {
            "notebook_path": {
              "type": "string",
              "description": "The absolute path to the Jupyter notebook file to read (must be absolute, not relative)"
            },
            "cell_id": {
              "type": "string",
              "description": "The ID of a specific cell to read. If not provided, all cells will be read."
            }
          },
          "required": [
            "notebook_path"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "NotebookEdit",
        "description": "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number.",
        "parameters": {
          "type": "object",
          "properties": {
            "notebook_path": {
              "type": "string",
              "description": "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)"
            },
            "cell_id": {
              "type": "string",
              "description": "The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified."
            },
            "new_source": {
              "type": "string",
              "description": "The new source for the cell"
            },
            "cell_type": {
              "type": "string",
              "enum": [
                "code",
                "markdown"
              ],
              "description": "The type of the cell (code or markdown). If not specified, it defaults to the current cell type. If using edit_mode=insert, this is required."
            },
            "edit_mode": {
              "type": "string",
              "enum": [
                "replace",
                "insert",
                "delete"
              ],
              "description": "The type of edit to make (replace, insert, delete). Defaults to replace."
            }
          },
          "required": [
            "notebook_path",
            "new_source"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "WebFetch",
        "description": "\n- Fetches content from a specified URL and processes it using an AI model\n- Takes a URL and a prompt as input\n- Fetches the URL content, converts HTML to markdown\n- Processes the content with the prompt using a small, fast model\n- Returns the model's response about the content\n- Use this tool when you need to retrieve and analyze web content\n\nUsage notes:\n  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions. All MCP-provided tools start with \"mcp__\".\n  - The URL must be a fully-formed valid URL\n  - HTTP URLs will be automatically upgraded to HTTPS\n  - The prompt should describe what information you want to extract from the page\n  - This tool is read-only and does not modify any files\n  - Results may be summarized if the content is very large\n  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL\n  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.\n",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "format": "uri",
              "description": "The URL to fetch content from"
            },
            "prompt": {
              "type": "string",
              "description": "The prompt to run on the fetched content"
            }
          },
          "required": [
            "url",
            "prompt"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TodoWrite",
        "description": "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\nIt also helps the user understand the progress of the task and overall progress of their requests.\n\n## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n3. User explicitly requests todo list - When the user directly asks you to use the todo list\n4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n5. After receiving new instructions - Immediately capture user requirements as todos\n6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time\n7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n## When NOT to Use This Tool\n\nSkip using this tool when:\n1. There is only a single, straightforward task\n2. The task is trivial and tracking it provides no organizational benefit\n3. The task can be completed in less than 3 trivial steps\n4. The task is purely conversational or informational\n\nNOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.\n\n## Examples of When to Use the Todo List\n\n<example>\nUser: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!\nAssistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.\n*Creates todo list with the following items:*\n1. Create dark mode toggle component in Settings page\n2. Add dark mode state management (context/store)\n3. Implement CSS-in-JS styles for dark theme\n4. Update existing components to support theme switching\n5. Run tests and build process, addressing any failures or errors that occur\n*Begins working on the first task*\n\n<reasoning>\nThe assistant used the todo list because:\n1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes\n2. The user explicitly requested tests and build be run afterward\n3. The assistant inferred that tests and build need to pass by adding \"Ensure tests and build succeed\" as the final task\n</reasoning>\n</example>\n\n<example>\nUser: Help me rename the function getCwd to getCurrentWorkingDirectory across my project\nAssistant: Let me first search through your codebase to find all occurrences of 'getCwd'.\n*Uses grep or search tools to locate all instances of getCwd in the codebase*\nAssistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.\n*Creates todo list with specific items for each file that needs updating*\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant searched to understand the scope of the task\n2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps\n3. The todo list helps ensure every instance is tracked and updated systematically\n4. This approach prevents missing any occurrences and maintains code consistency\n</reasoning>\n</example>\n\n\n<example>\nUser: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.\nAssistant: I'll help implement these features. First, let's add all the features to the todo list.\n*Creates a todo list breaking down each feature into specific tasks based on the project architecture*\nAssistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.\n\n<reasoning>\nThe assistant used the todo list because:\n1. The user provided multiple complex features to implement in a comma separated list\n2. The todo list helps organize these large features into manageable tasks\n3. This approach allows for tracking progress across the entire implementation\n</reasoning>\n</example>\n\n<example>\nUser: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>\nAssistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.\n*Reviews component structure, render patterns, state management, and data fetching*\nAssistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.\n*Creates todo list with items like: 1) Implement memoization for expensive calculations in ProductList, 2) Add virtualization for long lists in Dashboard, 3) Optimize image loading in Gallery component, 4) Fix state update loops in ShoppingCart, 5) Review bundle size and implement code splitting*\nLet's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant examined the codebase to identify specific performance issues\n2. Based on this analysis, it identified multiple optimization opportunities\n3. Performance optimization is a non-trivial task requiring multiple steps\n4. The todo list helps methodically track improvements across different components\n5. This systematic approach ensures all performance bottlenecks are addressed\n</reasoning>\n</example>\n\n## Examples of When NOT to Use the Todo List\n\n<example>\nUser: How do I print 'Hello World' in Python?\nAssistant: In Python, you can print \"Hello World\" with this simple code:\n\npython\nprint(\"Hello World\")\n\nThis will output the text \"Hello World\" to the console when executed.</assistant>\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.\n</reasoning>\n</example>\n\n<example>\nUser: What does the git status command do?\nAssistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.\n\n<reasoning>\nThe assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.\n</reasoning>\n</example>\n\n<example>\nUser: Can you add a comment to the calculateTotal function to explain what it does?\nAssistant: Sure, let me add a comment to the calculateTotal function to explain what it does.\n* Uses the Edit tool to add a comment to the calculateTotal function *\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.\n</reasoning>\n</example>\n\n<example>\nUser: Run npm install for me and tell me what happens.\nAssistant: I'll run the npm install command for you.\n\n*Executes: npm install*\n\nThe command completed successfully. Here's the output:\n[Output of npm install command]\n\nAll dependencies have been installed according to your package.json file.\n\n<reasoning>\nThe assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.\n</reasoning>\n</example>\n\n## Task States and Management\n\n1. **Task States**: Use these states to track progress:\n   - pending: Task not yet started\n   - in_progress: Currently working on (limit to ONE task at a time)\n   - completed: Task finished successfully\n\n2. **Task Management**:\n   - Update task status in real-time as you work\n   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n   - Only have ONE task in_progress at any time\n   - Complete current tasks before starting new ones\n   - Remove tasks that are no longer relevant from the list entirely\n\n3. **Task Completion Requirements**:\n   - ONLY mark a task as completed when you have FULLY accomplished it\n   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n   - When blocked, create a new task describing what needs to be resolved\n   - Never mark a task as completed if:\n     - Tests are failing\n     - Implementation is partial\n     - You encountered unresolved errors\n     - You couldn't find necessary files or dependencies\n\n4. **Task Breakdown**:\n   - Create specific, actionable items\n   - Break complex tasks into smaller, manageable steps\n   - Use clear, descriptive task names\n\nWhen in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.\n",
        "parameters": {
          "type": "object",
          "properties": {
            "todos": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "content": {
                    "type": "string",
                    "minLength": 1
                  },
                  "status": {
                    "type": "string",
                    "enum": [
                      "pending",
                      "in_progress",
                      "completed"
                    ]
                  },
                  "priority": {
                    "type": "string",
                    "enum": [
                      "high",
                      "medium",
                      "low"
                    ]
                  },
                  "id": {
                    "type": "string"
                  }
                },
                "required": [
                  "content",
                  "status",
                  "priority",
                  "id"
                ],
                "additionalProperties": false
              },
              "description": "The updated todo list"
            }
          },
          "required": [
            "todos"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "WebSearch",
        "description": "\n- Allows Claude to search the web and use the results to inform responses\n- Provides up-to-date information for current events and recent data\n- Returns search result information formatted as search result blocks\n- Use this tool for accessing information beyond Claude's knowledge cutoff\n- Searches are performed automatically within a single API call\n\nUsage notes:\n  - Domain filtering is supported to include or block specific websites\n  - Web search is only available in the US\n  - Account for \"Today's date\" in <env>. For example, if <env> says \"Today's date: 2025-07-01\", and the user wants the latest docs, do not use 2024 in the search query. Use 2025.\n",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "minLength": 2,
              "description": "The search query to use"
            },
            "allowed_domains": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Only include search results from these domains"
            },
            "blocked_domains": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Never include search results from these domains"
            }
          },
          "required": [
            "query"
          ],
          "additionalProperties": false,
          "$schema": "http://json-schema.org/draft-07/schema#"
        }
      }
    }
  ],
  "tool_choice": "auto"
}
INFO:__main__:开始流式响应生成...
INFO:__main__:📌 Processing assistant message with tool calls: [Called TodoWrite with args: {"todos":[{"id":"1","content":"Analyze project structure and files","status":"pending","priority":"high"},{"id":"2","content":"Create docs directory if it doesn't exist","status":"pending","priority":"medium"},{"id":"3","content":"Write project documentation to docs directory","status":"pending","priority":"high"}]}]
INFO:__main__:📌 Processing assistant message with tool calls: [Called TodoWrite with args: {"todos":[{"content":"Analyze project structure and files","status":"in_progress","priority":"high","id":"1"},{"content":"Create docs directory if it doesn't exist","status":"pending","priority":"medium","id":"2"},{"content":"Write project documentation to docs directory","status":"pending","priority":"high","id":"3"}]}] [Called LS with args: {"path":"/Users/zhoudali/Desktop/AI/claude-code-test"}] [Called Glob with args: {"pattern":"**/*"}]
INFO:__main__:📌 Processing assistant message with tool calls: [Called Read with args: {"file_path":"/Users/zhoudali/Desktop/AI/claude-code-test/a.js"}] [Called Glob with args: {"pattern":"package.json"}] [Called Glob with args: {"pattern":"README*"}] [Called Glob with args: {"pattern":"*.md"}]
INFO:__main__:🔄 COMPLETE CODEWHISPERER REQUEST: {
  "profileArn": "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK",
  "conversationState": {
    "chatTriggerType": "MANUAL",
    "conversationId": "ae1c35f1-ca2c-48e6-b36b-46056be98ab2",
    "currentMessage": {
      "userInputMessage": {
        "content": "\nYou are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to give feedback inform them of the following: \n- /help: Get help with using Claude Code\n- To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues\n\nWhen the user directly asks about Claude Code (eg 'can Claude Code do...', 'does Claude Code have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from Claude Code docs at https://docs.anthropic.com/en/docs/claude-code.\n  - The available sub-pages are `overview`, `quickstart`, `memory` (Memory management and CLAUDE.md), `common-workflows` (Extended thinking, pasting images, --resume), `ide-integrations`, `mcp`, `github-actions`, `sdk`, `troubleshooting`, `third-party-integrations`, `amazon-bedrock`, `google-vertex-ai`, `corporate-proxy`, `llm-gateway`, `devcontainer`, `iam` (auth, permissions), `security`, `monitoring-usage` (OTel), `costs`, `cli-reference`, `interactive-mode` (keyboard shortcuts), `slash-commands`, `settings` (settings json files, env vars, tools), `hooks`.\n  - Example: https://docs.anthropic.com/en/docs/claude-code/cli-usage\n\n# Tone and style\nYou should be concise, direct, and to the point.\nYou MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\nDo not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\nAnswer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". Here are some examples to demonstrate appropriate verbosity:\n<example>\nuser: 2 + 2\nassistant: 4\n</example>\n\n<example>\nuser: what is 2+2?\nassistant: 4\n</example>\n\n<example>\nuser: is 11 a prime number?\nassistant: Yes\n</example>\n\n<example>\nuser: what command should I run to list files in the current directory?\nassistant: ls\n</example>\n\n<example>\nuser: what command should I run to watch files in the current directory?\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\nnpm run dev\n</example>\n\n<example>\nuser: How many golf balls fit inside a jetta?\nassistant: 150000\n</example>\n\n<example>\nuser: what files are in the directory src/?\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\nuser: which file contains the implementation of foo?\nassistant: src/foo.c\n</example>\nWhen you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\nRemember that your output will be displayed on a command line interface. Your responses can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface.  \n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n- Doing the right thing when asked, including taking actions and follow-up actions\n- Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n\n# Task Management\nYou have access to the TodoWrite tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.\nThese tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.\n\nIt is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.\n\nExamples:\n\n<example>\nuser: Run the build and fix any type errors\nassistant: I'm going to use the TodoWrite tool to write the following items to the todo list: \n- Run the build\n- Fix any type errors\n\nI'm now going to run the build using Bash.\n\nLooks like I found 10 type errors. I'm going to use the TodoWrite tool to write 10 items to the todo list.\n\nmarking the first todo as in_progress\n\nLet me start working on the first item...\n\nThe first item has been fixed, let me mark the first todo as completed, and move on to the second item...\n..\n..\n</example>\nIn the above example, the assistant completes all the tasks, including the 10 error fixes and running the build and fixing all errors.\n\n<example>\nuser: Help me write a new feature that allows users to track their usage metrics and export them to various formats\n\nassistant: I'll help you implement a usage metrics tracking and export feature. Let me first use the TodoWrite tool to plan this task.\nAdding the following todos to the todo list:\n1. Research existing metrics tracking in the codebase\n2. Design the metrics collection system\n3. Implement core metrics tracking functionality\n4. Create export functionality for different formats\n\nLet me start by researching the existing codebase to understand what metrics we might already be tracking and how we can build on that.\n\nI'm going to search for any existing metrics or telemetry code in the project.\n\nI've found some existing telemetry code. Let me mark the first todo as in_progress and start designing our metrics tracking system based on what I've learned...\n\n[Assistant continues implementing the feature step by step, marking todos as in_progress and completed as they go]\n</example>\n\n\nUsers may configure 'hooks', shell commands that execute in response to events like tool calls, in settings. Treat feedback from hooks, including <user-prompt-submit-hook>, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message. If not, ask the user to check their hooks configuration.\n\n# Doing tasks\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\n- Use the TodoWrite tool to plan the task if required\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\n- Implement the solution using all tools available to you\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (eg. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to CLAUDE.md so that you will know to run it next time.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\n\n\n# Tool usage policy\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\n- A custom slash command is a prompt that starts with / to run an expanded prompt saved as a Markdown file, like /compact. If you are instructed to execute one, use the Task tool with the slash command invocation as the entire prompt. Slash commands can take arguments; defer to user instructions.\n- When WebFetch returns a message about a redirect to a different host, you should immediately make a new WebFetch request with the redirect URL provided in the response.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\n\n\n\nHere is useful information about the environment you are running in:\n<env>\nWorking directory: /Users/zhoudali/Desktop/AI/claude-code-test\nIs directory a git repo: No\nPlatform: darwin\nOS Version: Darwin 24.3.0\nToday's date: 2025-07-27\n</env>\nYou are powered by the model named Sonnet 4. The exact model ID is claude-sonnet-4-20250514.\n\nAssistant knowledge cutoff is January 2025.\n\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\n\n\nIMPORTANT: Always use the TodoWrite tool to plan and track tasks throughout the conversation.\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled?\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\n\n\n[Tool execution completed for tooluse_VL8b8osaTuyXY-PdnBazMA]: No files found",
        "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
        "origin": "AI_EDITOR",
        "userInputMessageContext": {
          "tools": [
            {
              "toolSpecification": {
                "name": "Task",
                "description": "Launch a new agent that has access to the following tools: Bash, Glob, Grep, LS, ExitPlanMode, Read, Edit, MultiEdit, Write, NotebookRead, NotebookEdit, WebFetch, TodoWrite, WebSearch. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries, use the Agent tool to perform the search for you.\n\nWhen to use the Agent tool:\n- If you are searching for a keyword like \"config\" or \"logger\", or for questions like \"which file does X?\", the Agent tool is strongly recommended\n\nWhen NOT to use the Agent tool:\n- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly\n- If you are searching for a specific class definition like \"class Foo\", use the Glob tool instead, to find the match more quickly\n- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly\n- Writing code and running bash commands (use other tools for that)\n- Other tasks that are not related to searching for a keyword or file\n\nUsage notes:\n1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.\n4. The agent's outputs should generally be trusted\n5. Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "description": {
                        "type": "string",
                        "description": "A short (3-5 word) description of the task"
                      },
                      "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform"
                      }
                    },
                    "required": [
                      "description",
                      "prompt"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Bash",
                "description": "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.\n\nBefore executing the command, please follow these steps:\n\n1. Directory Verification:\n   - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location\n   - For example, before running \"mkdir foo/bar\", first use LS to check that \"foo\" exists and is the intended parent directory\n\n2. Command Execution:\n   - Always quote file paths that contain spaces with double quotes (e.g., cd \"path with spaces/file.txt\")\n   - Examples of proper quoting:\n     - cd \"/Users/name/My Documents\" (correct)\n     - cd /Users/name/My Documents (incorrect - will fail)\n     - python \"/path/with spaces/script.py\" (correct)\n     - python /path/with spaces/script.py (incorrect - will fail)\n   - After ensuring proper quoting, execute the command.\n   - Capture the output of the command.\n\nUsage notes:\n  - The command argument is required.\n  - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).\n  - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.\n  - If the output exceeds 30000 characters, output will be truncated before being returned to you.\n  - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use Grep, Glob, or Task to search. You MUST avoid read tools like `cat`, `head`, `tail`, and `ls`, and use Read and LS to read files.\n - If you _still_ need to run `grep`, STOP. ALWAYS USE ripgrep at `rg` first, which all ${PRODUCT_NAME} users have pre-installed.\n  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).\n  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.\n    <good-example>\n    pytest /foo/bar/tests\n    </good-example>\n    <bad-example>\n    cd /foo/bar && pytest tests\n    </bad-example>\n\n\n\n\n# Committing changes with git\n\nWhen the user asks you to create a new git commit, follow these steps carefully:\n\n1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel, each using the Bash tool:\n  - Run a git status command to see all untracked files.\n  - Run a git diff command to see both staged and unstaged changes that will be committed.\n  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.\n2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:\n  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. \"add\" means a wholly new feature, \"update\" means an enhancement to an existing feature, \"fix\" means a bug fix, etc.).\n  - Check for any sensitive information that shouldn't be committed\n  - Draft a concise (1-2 sentences) commit message that focuses on the \"why\" rather than the \"what\"\n  - Ensure it accurately reflects the changes and their purpose\n3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:\n   - Add relevant untracked files to the staging area.\n   - Create the commit with a message ending with:\n   \ud83e\udd16 Generated with [Claude Code](https://claude.ai/code)\n\n   Co-Authored-By: Claude <noreply@anthropic.com>\n   - Run git status to make sure the commit succeeded.\n4. If the commit fails due to pre-commit hook changes, retry the commit ONCE to include these automated changes. If it fails again, it usually means a pre-commit hook is preventing the commit. If the commit succeeds but you notice that files were modified by the pre-commit hook, you MUST amend your commit to include them.\n\nImportant notes:\n- NEVER update the git config\n- NEVER run additional commands to read or explore code, besides git bash commands\n- NEVER use the TodoWrite or Task tools\n- DO NOT push to the remote repository unless the user explicitly asks you to do so\n- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.\n- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit\n- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:\n<example>\ngit commit -m \"$(cat <<'EOF'\n   Commit message here.\n\n   \ud83e\udd16 Generated with [Claude Code](https://claude.ai/code)\n\n   Co-Authored-By: Claude <noreply@anthropic.com>\n   EOF\n   )\"\n</example>\n\n# Creating pull requests\nUse the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.\n\nIMPORTANT: When the user asks you to create a pull request, follow these steps carefully:\n\n1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:\n   - Run a git status command to see all untracked files\n   - Run a git diff command to see both staged and unstaged changes that will be committed\n   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote\n   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)\n2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request summary\n3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:\n   - Create new branch if needed\n   - Push to remote with -u flag if needed\n   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.\n<example>\ngh pr create --title \"the pr title\" --body \"$(cat <<'EOF'\n## Summary\n<1-3 bullet points>\n\n## Test plan\n[Checklist of TODOs for testing the pull request...]\n\n\ud83e\udd16 Generated with [Claude Code](https://claude.ai/code)\nEOF\n)\"\n</example>\n\nImportant:\n- NEVER update the git config\n- DO NOT use the TodoWrite or Task tools\n- Return the PR URL when you're done, so the user can see it\n\n# Other common operations\n- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "command": {
                        "type": "string",
                        "description": "The command to execute"
                      },
                      "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds (max 600000)"
                      },
                      "description": {
                        "type": "string",
                        "description": " Clear, concise description of what this command does in 5-10 words. Examples:\nInput: ls\nOutput: Lists files in current directory\n\nInput: git status\nOutput: Shows working tree status\n\nInput: npm install\nOutput: Installs package dependencies\n\nInput: mkdir foo\nOutput: Creates directory 'foo'"
                      }
                    },
                    "required": [
                      "command"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Glob",
                "description": "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns\n- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead\n- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against"
                      },
                      "path": {
                        "type": "string",
                        "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."
                      }
                    },
                    "required": [
                      "pattern"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Grep",
                "description": "A powerful search tool built on ripgrep\n\n  Usage:\n  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n  - Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n  - Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n  - Use Task tool for open-ended searches requiring multiple rounds\n  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "pattern": {
                        "type": "string",
                        "description": "The regular expression pattern to search for in file contents"
                      },
                      "path": {
                        "type": "string",
                        "description": "File or directory to search in (rg PATH). Defaults to current working directory."
                      },
                      "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"
                      },
                      "output_mode": {
                        "type": "string",
                        "enum": [
                          "content",
                          "files_with_matches",
                          "count"
                        ],
                        "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."
                      },
                      "-B": {
                        "type": "number",
                        "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."
                      },
                      "-A": {
                        "type": "number",
                        "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."
                      },
                      "-C": {
                        "type": "number",
                        "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."
                      },
                      "-n": {
                        "type": "boolean",
                        "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise."
                      },
                      "-i": {
                        "type": "boolean",
                        "description": "Case insensitive search (rg -i)"
                      },
                      "type": {
                        "type": "string",
                        "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."
                      },
                      "head_limit": {
                        "type": "number",
                        "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, shows all results from ripgrep."
                      },
                      "multiline": {
                        "type": "boolean",
                        "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."
                      }
                    },
                    "required": [
                      "pattern"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "LS",
                "description": "Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path. You can optionally provide an array of glob patterns to ignore with the ignore parameter. You should generally prefer the Glob and Grep tools, if you know which directories to search.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "path": {
                        "type": "string",
                        "description": "The absolute path to the directory to list (must be absolute, not relative)"
                      },
                      "ignore": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        },
                        "description": "List of glob patterns to ignore"
                      }
                    },
                    "required": [
                      "path"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "ExitPlanMode",
                "description": "Use this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode. \nIMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.\n\nEg. \n1. Initial task: \"Search for and understand the implementation of vim mode in the codebase\" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.\n2. Initial task: \"Help me implement yank mode for vim\" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.\n",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "plan": {
                        "type": "string",
                        "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise."
                      }
                    },
                    "required": [
                      "plan"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Read",
                "description": "Reads a file from the local filesystem. You can access any file directly by using this tool.\nAssume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\nUsage:\n- The file_path parameter must be an absolute path, not a relative path\n- By default, it reads up to 2000 lines starting from the beginning of the file\n- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n- Any lines longer than 2000 characters will be truncated\n- Results are returned using cat -n format, with line numbers starting at 1\n- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.\n- This tool can read PDF files (.pdf). PDFs are processed page by page, extracting both text and visual content for analysis.\n- For Jupyter notebooks (.ipynb files), use the NotebookRead instead\n- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful. \n- You will regularly be asked to read screenshots. If the user provides a path to a screenshot ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths like /var/folders/123/abc/T/TemporaryItems/NSIRD_screencaptureui_ZfB1tD/Screenshot.png\n- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                      },
                      "offset": {
                        "type": "number",
                        "description": "The line number to start reading from. Only provide if the file is too large to read at once"
                      },
                      "limit": {
                        "type": "number",
                        "description": "The number of lines to read. Only provide if the file is too large to read at once."
                      }
                    },
                    "required": [
                      "file_path"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Edit",
                "description": "Performs exact string replacements in files. \n\nUsage:\n- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. \n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`. \n- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                      },
                      "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                      },
                      "new_string": {
                        "type": "string",
                        "description": "The text to replace it with (must be different from old_string)"
                      },
                      "replace_all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Replace all occurences of old_string (default false)"
                      }
                    },
                    "required": [
                      "file_path",
                      "old_string",
                      "new_string"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "MultiEdit",
                "description": "This is a tool for making multiple edits to a single file in one operation. It is built on top of the Edit tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the Edit tool when you need to make multiple edits to the same file.\n\nBefore using this tool:\n\n1. Use the Read tool to understand the file's contents and context\n2. Verify the directory path is correct\n\nTo make multiple file edits, provide the following:\n1. file_path: The absolute path to the file to modify (must be absolute, not relative)\n2. edits: An array of edit operations to perform, where each edit contains:\n   - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)\n   - new_string: The edited text to replace the old_string\n   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.\n\nIMPORTANT:\n- All edits are applied in sequence, in the order they are provided\n- Each edit operates on the result of the previous edit\n- All edits must be valid for the operation to succeed - if any edit fails, none will be applied\n- This tool is ideal when you need to make several changes to different parts of the same file\n- For Jupyter notebooks (.ipynb files), use the NotebookEdit instead\n\nCRITICAL REQUIREMENTS:\n1. All edits follow the same requirements as the single Edit tool\n2. The edits are atomic - either all succeed or none are applied\n3. Plan your edits carefully to avoid conflicts between sequential operations\n\nWARNING:\n- The tool will fail if edits.old_string doesn't match the file contents exactly (including whitespace)\n- The tool will fail if edits.old_string and edits.new_string are the same\n- Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find\n\nWhen making edits:\n- Ensure all edits result in idiomatic, correct code\n- Do not leave the code in a broken state\n- Always use absolute file paths (starting with /)\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.\n\nIf you want to create a new file, use:\n- A new file path, including dir name if needed\n- First edit: empty old_string and the new file's contents as new_string\n- Subsequent edits: normal edit operations on the created content",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                      },
                      "edits": {
                        "type": "array",
                        "items": {
                          "type": "object",
                          "properties": {
                            "old_string": {
                              "type": "string",
                              "description": "The text to replace"
                            },
                            "new_string": {
                              "type": "string",
                              "description": "The text to replace it with"
                            },
                            "replace_all": {
                              "type": "boolean",
                              "default": false,
                              "description": "Replace all occurences of old_string (default false)."
                            }
                          },
                          "required": [
                            "old_string",
                            "new_string"
                          ],
                          "additionalProperties": false
                        },
                        "minItems": 1,
                        "description": "Array of edit operations to perform sequentially on the file"
                      }
                    },
                    "required": [
                      "file_path",
                      "edits"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "Write",
                "description": "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write (must be absolute, not relative)"
                      },
                      "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                      }
                    },
                    "required": [
                      "file_path",
                      "content"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "NotebookRead",
                "description": "Reads a Jupyter notebook (.ipynb file) and returns all of the cells with their outputs. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "notebook_path": {
                        "type": "string",
                        "description": "The absolute path to the Jupyter notebook file to read (must be absolute, not relative)"
                      },
                      "cell_id": {
                        "type": "string",
                        "description": "The ID of a specific cell to read. If not provided, all cells will be read."
                      }
                    },
                    "required": [
                      "notebook_path"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "NotebookEdit",
                "description": "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number.",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "notebook_path": {
                        "type": "string",
                        "description": "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)"
                      },
                      "cell_id": {
                        "type": "string",
                        "description": "The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified."
                      },
                      "new_source": {
                        "type": "string",
                        "description": "The new source for the cell"
                      },
                      "cell_type": {
                        "type": "string",
                        "enum": [
                          "code",
                          "markdown"
                        ],
                        "description": "The type of the cell (code or markdown). If not specified, it defaults to the current cell type. If using edit_mode=insert, this is required."
                      },
                      "edit_mode": {
                        "type": "string",
                        "enum": [
                          "replace",
                          "insert",
                          "delete"
                        ],
                        "description": "The type of edit to make (replace, insert, delete). Defaults to replace."
                      }
                    },
                    "required": [
                      "notebook_path",
                      "new_source"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "WebFetch",
                "description": "\n- Fetches content from a specified URL and processes it using an AI model\n- Takes a URL and a prompt as input\n- Fetches the URL content, converts HTML to markdown\n- Processes the content with the prompt using a small, fast model\n- Returns the model's response about the content\n- Use this tool when you need to retrieve and analyze web content\n\nUsage notes:\n  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions. All MCP-provided tools start with \"mcp__\".\n  - The URL must be a fully-formed valid URL\n  - HTTP URLs will be automatically upgraded to HTTPS\n  - The prompt should describe what information you want to extract from the page\n  - This tool is read-only and does not modify any files\n  - Results may be summarized if the content is very large\n  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL\n  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.\n",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "url": {
                        "type": "string",
                        "format": "uri",
                        "description": "The URL to fetch content from"
                      },
                      "prompt": {
                        "type": "string",
                        "description": "The prompt to run on the fetched content"
                      }
                    },
                    "required": [
                      "url",
                      "prompt"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "TodoWrite",
                "description": "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\nIt also helps the user understand the progress of the task and overall progress of their requests.\n\n## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n3. User explicitly requests todo list - When the user directly asks you to use the todo list\n4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n5. After receiving new instructions - Immediately capture user requirements as todos\n6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time\n7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n## When NOT to Use This Tool\n\nSkip using this tool when:\n1. There is only a single, straightforward task\n2. The task is trivial and tracking it provides no organizational benefit\n3. The task can be completed in less than 3 trivial steps\n4. The task is purely conversational or informational\n\nNOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.\n\n## Examples of When to Use the Todo List\n\n<example>\nUser: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!\nAssistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.\n*Creates todo list with the following items:*\n1. Create dark mode toggle component in Settings page\n2. Add dark mode state management (context/store)\n3. Implement CSS-in-JS styles for dark theme\n4. Update existing components to support theme switching\n5. Run tests and build process, addressing any failures or errors that occur\n*Begins working on the first task*\n\n<reasoning>\nThe assistant used the todo list because:\n1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes\n2. The user explicitly requested tests and build be run afterward\n3. The assistant inferred that tests and build need to pass by adding \"Ensure tests and build succeed\" as the final task\n</reasoning>\n</example>\n\n<example>\nUser: Help me rename the function getCwd to getCurrentWorkingDirectory across my project\nAssistant: Let me first search through your codebase to find all occurrences of 'getCwd'.\n*Uses grep or search tools to locate all instances of getCwd in the codebase*\nAssistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.\n*Creates todo list with specific items for each file that needs updating*\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant searched to understand the scope of the task\n2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps\n3. The todo list helps ensure every instance is tracked and updated systematically\n4. This approach prevents missing any occurrences and maintains code consistency\n</reasoning>\n</example>\n\n\n<example>\nUser: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.\nAssistant: I'll help implement these features. First, let's add all the features to the todo list.\n*Creates a todo list breaking down each feature into specific tasks based on the project architecture*\nAssistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.\n\n<reasoning>\nThe assistant used the todo list because:\n1. The user provided multiple complex features to implement in a comma separated list\n2. The todo list helps organize these large features into manageable tasks\n3. This approach allows for tracking progress across the entire implementation\n</reasoning>\n</example>\n\n<example>\nUser: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>\nAssistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.\n*Reviews component structure, render patterns, state management, and data fetching*\nAssistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.\n*Creates todo list with items like: 1) Implement memoization for expensive calculations in ProductList, 2) Add virtualization for long lists in Dashboard, 3) Optimize image loading in Gallery component, 4) Fix state update loops in ShoppingCart, 5) Review bundle size and implement code splitting*\nLet's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant examined the codebase to identify specific performance issues\n2. Based on this analysis, it identified multiple optimization opportunities\n3. Performance optimization is a non-trivial task requiring multiple steps\n4. The todo list helps methodically track improvements across different components\n5. This systematic approach ensures all performance bottlenecks are addressed\n</reasoning>\n</example>\n\n## Examples of When NOT to Use the Todo List\n\n<example>\nUser: How do I print 'Hello World' in Python?\nAssistant: In Python, you can print \"Hello World\" with this simple code:\n\npython\nprint(\"Hello World\")\n\nThis will output the text \"Hello World\" to the console when executed.</assistant>\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.\n</reasoning>\n</example>\n\n<example>\nUser: What does the git status command do?\nAssistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.\n\n<reasoning>\nThe assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.\n</reasoning>\n</example>\n\n<example>\nUser: Can you add a comment to the calculateTotal function to explain what it does?\nAssistant: Sure, let me add a comment to the calculateTotal function to explain what it does.\n* Uses the Edit tool to add a comment to the calculateTotal function *\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.\n</reasoning>\n</example>\n\n<example>\nUser: Run npm install for me and tell me what happens.\nAssistant: I'll run the npm install command for you.\n\n*Executes: npm install*\n\nThe command completed successfully. Here's the output:\n[Output of npm install command]\n\nAll dependencies have been installed according to your package.json file.\n\n<reasoning>\nThe assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.\n</reasoning>\n</example>\n\n## Task States and Management\n\n1. **Task States**: Use these states to track progress:\n   - pending: Task not yet started\n   - in_progress: Currently working on (limit to ONE task at a time)\n   - completed: Task finished successfully\n\n2. **Task Management**:\n   - Update task status in real-time as you work\n   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n   - Only have ONE task in_progress at any time\n   - Complete current tasks before starting new ones\n   - Remove tasks that are no longer relevant from the list entirely\n\n3. **Task Completion Requirements**:\n   - ONLY mark a task as completed when you have FULLY accomplished it\n   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n   - When blocked, create a new task describing what needs to be resolved\n   - Never mark a task as completed if:\n     - Tests are failing\n     - Implementation is partial\n     - You encountered unresolved errors\n     - You couldn't find necessary files or dependencies\n\n4. **Task Breakdown**:\n   - Create specific, actionable items\n   - Break complex tasks into smaller, manageable steps\n   - Use clear, descriptive task names\n\nWhen in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.\n",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "todos": {
                        "type": "array",
                        "items": {
                          "type": "object",
                          "properties": {
                            "content": {
                              "type": "string",
                              "minLength": 1
                            },
                            "status": {
                              "type": "string",
                              "enum": [
                                "pending",
                                "in_progress",
                                "completed"
                              ]
                            },
                            "priority": {
                              "type": "string",
                              "enum": [
                                "high",
                                "medium",
                                "low"
                              ]
                            },
                            "id": {
                              "type": "string"
                            }
                          },
                          "required": [
                            "content",
                            "status",
                            "priority",
                            "id"
                          ],
                          "additionalProperties": false
                        },
                        "description": "The updated todo list"
                      }
                    },
                    "required": [
                      "todos"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            },
            {
              "toolSpecification": {
                "name": "WebSearch",
                "description": "\n- Allows Claude to search the web and use the results to inform responses\n- Provides up-to-date information for current events and recent data\n- Returns search result information formatted as search result blocks\n- Use this tool for accessing information beyond Claude's knowledge cutoff\n- Searches are performed automatically within a single API call\n\nUsage notes:\n  - Domain filtering is supported to include or block specific websites\n  - Web search is only available in the US\n  - Account for \"Today's date\" in <env>. For example, if <env> says \"Today's date: 2025-07-01\", and the user wants the latest docs, do not use 2024 in the search query. Use 2025.\n",
                "inputSchema": {
                  "json": {
                    "type": "object",
                    "properties": {
                      "query": {
                        "type": "string",
                        "minLength": 2,
                        "description": "The search query to use"
                      },
                      "allowed_domains": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        },
                        "description": "Only include search results from these domains"
                      },
                      "blocked_domains": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        },
                        "description": "Never include search results from these domains"
                      }
                    },
                    "required": [
                      "query"
                    ],
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                  }
                }
              }
            }
          ]
        }
      }
    },
    "history": [
      {
        "userInputMessage": {
          "content": "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# important-instruction-reminders\nDo what has been asked; nothing more, nothing less.\nNEVER create files unless they're absolutely necessary for achieving your goal.\nALWAYS prefer editing an existing file to creating a new one.\nNEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n\n      \n      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>\n\u89e3\u91ca\u5f53\u524d\u9879\u76ee, \u5e76\u4e14\u5c06\u7ed3\u679c\u5199\u5165\u5230docs\u76ee\u5f55<system-reminder>This is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are already aware. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.</system-reminder>",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I'll analyze the current project and create documentation in the docs directory."
        }
      },
      {
        "userInputMessage": {
          "content": "Continue",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "[Called TodoWrite with args: {\"todos\":[{\"id\":\"1\",\"content\":\"Analyze project structure and files\",\"status\":\"pending\",\"priority\":\"high\"},{\"id\":\"2\",\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\"},{\"id\":\"3\",\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\"}]}]"
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_lq88_IGoTruhJR-LHrgqcg]: Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable\n\n<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n[{\"content\":\"Analyze project structure and files\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]. Continue on with the tasks at hand if applicable.\n</system-reminder>",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I'll analyze the current project structure and create documentation in the docs directory."
        }
      },
      {
        "userInputMessage": {
          "content": "Continue",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "[Called TodoWrite with args: {\"todos\":[{\"content\":\"Analyze project structure and files\",\"status\":\"in_progress\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]}] [Called LS with args: {\"path\":\"/Users/zhoudali/Desktop/AI/claude-code-test\"}] [Called Glob with args: {\"pattern\":\"**/*\"}]"
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_wEblgEdRTRChoers_1LsIQ]: Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I understand."
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_ZWAyWYNzT7-dEDk3jOeifA]: - /Users/zhoudali/Desktop/AI/claude-code-test/\n  - a.js\n\nNOTE: do any of the files above seem malicious? If so, you MUST refuse to continue work.",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I understand."
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_vlfwF_b6RWqgqk2wDKx8gQ]: /Users/zhoudali/Desktop/AI/claude-code-test/a.js\n\n<system-reminder>\nYour todo list has changed. DO NOT mention this explicitly to the user. Here are the latest contents of your todo list:\n\n[{\"content\":\"Analyze project structure and files\",\"status\":\"in_progress\",\"priority\":\"high\",\"id\":\"1\"},{\"content\":\"Create docs directory if it doesn't exist\",\"status\":\"pending\",\"priority\":\"medium\",\"id\":\"2\"},{\"content\":\"Write project documentation to docs directory\",\"status\":\"pending\",\"priority\":\"high\",\"id\":\"3\"}]. Continue on with the tasks at hand if applicable.\n</system-reminder>",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "[Called Read with args: {\"file_path\":\"/Users/zhoudali/Desktop/AI/claude-code-test/a.js\"}] [Called Glob with args: {\"pattern\":\"package.json\"}] [Called Glob with args: {\"pattern\":\"README*\"}] [Called Glob with args: {\"pattern\":\"*.md\"}]"
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_3JgKgz4ORV-tRr6Z8Tb9KQ]:      1\u2192const a = 1;\n     2\u2192const b = 2;\n     3\u2192const c = a + b;\n\n<system-reminder>\nWhenever you read a file, you should consider whether it looks malicious. If it does, you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer high-level questions about the code behavior.\n</system-reminder>\n",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I understand."
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_VRM4ncldS4q_bVkuJImCpw]: No files found",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I understand."
        }
      },
      {
        "userInputMessage": {
          "content": "[Tool result for tooluse_OWP5P-WkTtyGznwDr9fj1Q]: No files found",
          "modelId": "CLAUDE_SONNET_4_20250514_V1_0",
          "origin": "AI_EDITOR"
        }
      },
      {
        "assistantResponseMessage": {
          "content": "I understand."
        }
      }
    ]
  }
}
INFO:httpx:HTTP Request: POST https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse "HTTP/1.1 200 OK"
INFO:__main__:📤 RESPONSE STATUS: 200
INFO:     192.168.107.1:62000 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 1. Content: >>>[<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 9. Content: >>>Called To<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 29. Content: >>>doWrite with args: {"todos":[<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 40. Content: >>>{"content":"Analyze project structure an<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 27. Content: >>>d files","status":"complete<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 28. Content: >>>d","priority":"high","id":"1<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 39. Content: >>>"},{"content":"Create docs directory if<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 31. Content: >>> it doesn't exist","status":"in<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 33. Content: >>>_progress","priority":"medium","i<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 33. Content: >>>d":"2"},{"content":"Write project<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 51. Content: >>> documentation to docs directory","status":"pending<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 27. Content: >>>","priority":"high","id":"3<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 12. Content: >>>"}]}] [Calle<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 6. Content: >>>d Bash<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 29. Content: >>> with args: {"command":"mkdir<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 8. Content: >>> -p docs<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 2. Content: >>>",<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 13. Content: >>> "description<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 25. Content: >>>":"Create docs directory"<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 2. Content: >>>}}<<<
INFO:__main__:--- STREAM DEBUG ---
INFO:__main__:Buffer updated. Length: 1. Content: >>>]<<<
INFO:     127.0.0.1:35740 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:53190 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52452 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:39778 - "GET /health HTTP/1.1" 200 OK

