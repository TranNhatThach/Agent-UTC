from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import hashlib
import os

# Import các cấu hình và module phụ trợ
from config import UPLOAD_FOLDER, redis_client, has_db
from rate_limiter import rate_limiter
from agent import run_ollama_agent
from rag_utils import process_and_embed_file, get_qdrant_client

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = []

@app.post("/api/upload", dependencies=[Depends(rate_limiter)])
async def upload_file(file: UploadFile = File(...)):
    global has_db
    if not has_db:
        # Thử kết nối lại DB động
        try:
            get_qdrant_client()
            has_db = True
        except Exception:
            raise HTTPException(status_code=500, detail="Hệ thống DB chưa sẵn sàng.")
        
    # Check file extension
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['txt', 'pdf', 'docx']:
        raise HTTPException(status_code=400, detail="Định dạng file không được hỗ trợ")
        
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {e}")
        
    success, message = process_and_embed_file(file_path)
    if success:
        return {"success": True, "message": f"Đã nạp file {file.filename} thành công! {message}"}
    else:
        raise HTTPException(status_code=500, detail=f"Lỗi khi nạp file: {message}")

@app.post("/api/chat", dependencies=[Depends(rate_limiter)])
async def chat_endpoint(payload: ChatRequest):
    message = payload.message
    history = payload.history

    if not message:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp tin nhắn")

    # Kiểm tra Redis Cache trước
    cache_key = None
    if redis_client:
        try:
            # Tạo key dựa vào message và history
            history_str = "".join([f"{msg.role}:{msg.content}" for msg in history])
            hash_input = f"local:{history_str}:{message.strip().lower()}"
            cache_key = f"chat_cache:{hashlib.md5(hash_input.encode('utf-8')).hexdigest()}"
            
            cached_reply = redis_client.get(cache_key)
            if cached_reply:
                print(f"[Redis Cache] Đánh trúng cache cho câu hỏi: '{message}'")
                return {"reply": cached_reply, "cached": True}
        except Exception as cache_err:
            print(f"[Redis Warning] Lỗi đọc cache: {cache_err}")

    try:
        # Chuyển đổi history Pydantic model thành list dict
        history_list = [{"role": h.role, "content": h.content} for h in history]
        
        # Chạy Agent Local Ollama
        local_model_name = os.environ.get('LOCAL_MODEL_NAME', 'tuvan_thpt')
        reply = run_ollama_agent(message, local_model_name, history_list)

        # Lưu vào Redis Cache nếu thành công (Cache trong 1 giờ)
        if redis_client and cache_key and reply:
            try:
                redis_client.set(cache_key, reply, ex=3600)
                print("[Redis Cache] Đã lưu kết quả mới vào cache.")
            except Exception as cache_err:
                print(f"[Redis Warning] Lỗi lưu cache: {cache_err}")

        return {"reply": reply}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    print("========================================")
    print("Khởi động AGENT AI Server tại http://127.0.0.1:5000")
    print("Chế độ Model: Local Model (Ollama)")
    print("========================================")
    
    try:
        from rag_utils import get_embedding_model
        print("Đang tải trước model Embedding BGE-m3 trong Main Thread...")
        get_embedding_model()
        print("✅ Đã tải xong model Embedding.")
    except Exception as model_err:
        print(f"⚠️ Cảnh báo: Lỗi tải trước model Embedding: {model_err}")

    uvicorn.run("server:app", host="127.0.0.1", port=5000)
