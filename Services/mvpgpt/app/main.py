import os
import json
import httpx
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# --- Configuration ---
WORKER_URL = "https://thirdmvpchatapworker.ainh.workers.dev"
SESSION_ID = "global_chat_v1"
API_KEY = os.getenv('EXTERNAL_API','2343432423432')
MODEL_NAME = "gemini-2.5-flash"

# Initialize Client
client = genai.Client(api_key=API_KEY)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Client UI (HTML/JS) ---
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Chat</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
            #chat-box { border: 1px solid #ddd; background: #fff; height: 65vh; overflow-y: auto; padding: 20px; margin-bottom: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            .msg { margin: 10px 0; padding: 12px 16px; border-radius: 12px; max-width: 80%; word-wrap: break-word; }
            .user { background: #0084ff; color: white; margin-left: auto; border-bottom-right-radius: 4px; }
            .assistant { background: #e4e6eb; color: #050505; margin-right: auto; border-bottom-left-radius: 4px; }
            .controls { display: flex; gap: 10px; }
            input { flex-grow: 1; padding: 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
            button { padding: 0 24px; cursor: pointer; border: none; border-radius: 8px; font-weight: 600; font-size: 15px; }
            button#send { background: #0084ff; color: white; }
            button.reset { background: #ff3b30; color: white; }
            button:active { opacity: 0.8; }
        </style>
    </head>
    <body>
        <h2>Gemini 2.5 Flash Chat</h2>
        <div id="chat-box"></div>
        <div class="controls">
            <input type="text" id="messageText" autocomplete="off" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendMessage()"/>
            <button id="send" onclick="sendMessage()">Send</button>
            <button class="reset" onclick="resetChat()">Reset</button>
        </div>
        <script>
            var ws = new WebSocket((window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host + "/ws");
            var chatBox = document.getElementById('chat-box');
            var currentStreamDiv = null;

            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === 'clear') {
                    chatBox.innerHTML = '';
                    currentStreamDiv = null;
                } 
                else if (data.role === 'user') {
                    addMessage('user', data.content);
                } 
                else if (data.role === 'assistant') {
                    if (data.type === 'stream') {
                        if (!currentStreamDiv) {
                            currentStreamDiv = document.createElement('div');
                            currentStreamDiv.className = 'msg assistant';
                            chatBox.appendChild(currentStreamDiv);
                        }
                        currentStreamDiv.textContent += data.chunk;
                    } else if (data.type === 'full') {
                         addMessage('assistant', data.content);
                    }
                }
                
                if (data.type === 'stream_end') {
                    currentStreamDiv = null;
                }
                scrollToBottom();
            };

            function addMessage(role, text) {
                var div = document.createElement('div');
                div.className = 'msg ' + role;
                div.textContent = text;
                chatBox.appendChild(div);
                scrollToBottom();
            }

            function scrollToBottom() {
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                var input = document.getElementById("messageText");
                if (!input.value.trim()) return;
                var content = input.value;
                input.value = '';
                await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({role: "user", content: content})
                });
            }

            async function resetChat() {
                if(confirm("Delete history?")) await fetch("/reset", { method: "DELETE" });
            }
        </script>
    </body>
</html>
"""

# --- Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        encoded = json.dumps(message)
        for connection in self.active_connections[:]:
            try: await connection.send_text(encoded)
            except: pass

manager = ConnectionManager()
http_client = httpx.AsyncClient()

# --- Helper: Universal History Parser ---
def parse_worker_history(data: Any) -> List[Dict]:
    """
    Handles both [msg, msg] and {'history': [msg, msg]}
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("history", [])
    return []

# --- Helpers ---
async def save_message(role: str, content: str):
    try:
        await http_client.post(
            f"{WORKER_URL}/set?session_id={SESSION_ID}", 
            json={"role": role, "content": content},
            timeout=5.0
        )
    except Exception as e:
        print(f"Error saving to worker: {e}")

async def generate_ai_response(user_content: str):
    print(f"DEBUG: AI Generation triggered for: {user_content}")
    
    # 1. Fetch History
    raw_history = []
    try:
        resp = await http_client.get(f"{WORKER_URL}/get?session_id={SESSION_ID}", timeout=10.0)
        if resp.status_code == 200:
            try:
                # --- FIXED: Use Helper Parser ---
                json_data = resp.json()
                raw_history = parse_worker_history(json_data)
            except json.JSONDecodeError:
                print("Worker returned invalid JSON. Using empty history.")
                raw_history = []
        else:
            print(f"Worker returned status {resp.status_code}")
    except Exception as e:
        print(f"Warning: Failed to fetch history: {e}")
        raw_history = []

    # 2. Format for Gemini
    gemini_contents = []
    for msg in raw_history:
        role = "model" if msg.get("role") in ["assistant", "model"] else "user"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg.get("content", ""))]
            )
        )
    
    # Append CURRENT message
    gemini_contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_content)]
        )
    )

    # 3. Stream
    full_response = ""
    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=gemini_contents
        )

        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                await manager.broadcast({"role": "assistant", "chunk": chunk.text, "type": "stream"})
        
        await save_message("assistant", full_response)
        await manager.broadcast({"type": "stream_end"})

    except Exception as e:
        err_msg = f"Error generating response: {str(e)}"
        print(err_msg)
        await manager.broadcast({"role": "assistant", "chunk": f" [{err_msg}]", "type": "stream"})
        await manager.broadcast({"type": "stream_end"})

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_client():
    return html

@app.post("/chat")
async def chat_endpoint(background_tasks: BackgroundTasks, payload: Dict = Body(...)):
    user_content = payload.get("content", "")
    background_tasks.add_task(save_message, "user", user_content)
    await manager.broadcast({"role": "user", "content": user_content})
    background_tasks.add_task(generate_ai_response, user_content)
    return {"status": "ok"}

@app.delete("/reset")
async def reset_history():
    try:
        await http_client.delete(f"{WORKER_URL}/delete?session_id={SESSION_ID}")
        await manager.broadcast({"type": "clear"})
    except Exception as e:
        print(f"Error resetting: {e}")
    return {"status": "reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # --- Sync History (Fixed) ---
    try:
        resp = await http_client.get(f"{WORKER_URL}/get?session_id={SESSION_ID}")
        if resp.status_code == 200 and resp.text.strip():
            # --- FIXED: Use Helper Parser ---
            json_data = resp.json()
            history = parse_worker_history(json_data)
            
            for msg in history:
                if isinstance(msg, dict): # Safety check
                    await websocket.send_text(json.dumps({
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "type": "full"
                    }))
    except Exception as e:
        print(f"History sync error: {e}")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()