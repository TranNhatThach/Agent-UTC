from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import hashlib
import os

# Import configurations and services
from config import UPLOAD_FOLDER, redis_client, has_db
from rate_limiter import rate_limiter
from services.llm_service import run_ollama_agent
from services.document_worker import process_and_embed_file
from services.vector_store import get_qdrant_client, lookup_majors_by_score

app = FastAPI()

# CORS configuration
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

@app.post("/api/analyze")
async def analyze_scores(scores: List[float]):
    """
    Nhận một danh sách điểm số môn thi và gọi Agent phân tích kết quả tư vấn tuyển sinh ngành học tại Giao thông Vận tải.
    """
    if not scores:
        raise HTTPException(status_code=400, detail="Danh sách điểm không được rỗng")
    try:
        # Check if only a single total score is provided
        if len(scores) == 1:
            total_score = scores[0]
            prompt = (
                f"Học sinh chỉ cung cấp tổng điểm xét tuyển THPT là {total_score} điểm (chưa rõ tổ hợp môn cụ thể). "
                f"Hãy tư vấn chi tiết các ngành, khoa của trường Đại học Giao thông Vận tải mà học sinh này có thể đỗ (điểm chuẩn từ {total_score} trở xuống), "
                f"đồng thời liệt kê các tổ hợp môn tương ứng của từng ngành đó để học sinh tham khảo."
            )
            response_payload = {
                "success": True,
                "total_score": total_score,
                "input_type": "total_score_only"
            }
        else:
            math = scores[0] if len(scores) > 0 else 0.0
            physics = scores[1] if len(scores) > 1 else 0.0
            chemistry = scores[2] if len(scores) > 2 else 0.0
            english = scores[3] if len(scores) > 3 else 0.0
            literature = scores[4] if len(scores) > 4 else 0.0
            
            prompt = (
                f"Phân tích điểm thi THPT của học sinh: "
                f"Toán: {math}, Lý: {physics}, Hóa: {chemistry}, Anh: {english}, Văn: {literature}. "
                f"Hãy tư vấn chi tiết các tổ hợp xét tuyển (A00, A01, D01, D07) và các ngành học sinh này có thể đỗ tại trường."
            )
            response_payload = {
                "success": True,
                "math": math,
                "physics": physics,
                "chemistry": chemistry,
                "english": english,
                "literature": literature,
                "input_type": "subject_scores"
            }
        
        local_model_name = os.environ.get('LOCAL_MODEL_NAME', 'tuvan_thpt')
        reply = run_ollama_agent(prompt, local_model_name)
        response_payload["recommendations"] = reply
        return response_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích điểm bằng Agent: {e}")

@app.post("/api/upload", dependencies=[Depends(rate_limiter)])
async def upload_file(file: UploadFile = File(...)):
    global has_db
    if not has_db:
        # Try to reconnect dynamically
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

    try:
        # Convert history Pydantic model to list of dicts
        history_list = [{"role": h.role, "content": h.content} for h in history]
        
        # Run local agent
        local_model_name = os.environ.get('LOCAL_MODEL_NAME', 'tuvan_thpt')
        reply = run_ollama_agent(message, local_model_name, history_list)

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
        from services.vector_store import get_embedding_model
        print("Đang tải trước model Embedding BGE-m3 trong Main Thread...")
        get_embedding_model()
        print("✅ Đã tải xong model Embedding.")
    except Exception as model_err:
        print(f"⚠️ Cảnh báo: Lỗi tải trước model Embedding: {model_err}")

    uvicorn.run("main:app", host="127.0.0.1", port=5000)
