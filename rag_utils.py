import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import uuid
import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import PyPDF2
from docx import Document

DATA_DIR = "data"
DB_DIR = "qdrant_db"
COLLECTION_NAME = "admission_info"

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Đang tải thư viện AI (PyTorch/Transformers)...")
        from sentence_transformers import SentenceTransformer
        print("Đang tải model Embedding BGE-m3 (khoảng 2.3GB)...")
        embedding_model = SentenceTransformer("BAAI/bge-m3")
    return embedding_model

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        client = QdrantClient(path=DB_DIR)
        collections = [col.name for col in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print("Tạo Collection mới trên Qdrant...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        _qdrant_client = client
    return _qdrant_client

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
    return text

def chunk_text(text, max_chunk_size=500, overlap=100):
    # Chuẩn hóa khoảng trắng
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        chunk = text[start:end]
        # Thử lùi lại một chút để tìm khoảng trắng (tránh cắt giữa chừng một từ)
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > max_chunk_size - overlap:
                end = start + last_space
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (len(chunk) - overlap) if (len(chunk) - overlap) > 0 else len(chunk)
    return [c for c in chunks if len(c) > 20]

def process_and_embed_file(file_path):
    filename = os.path.basename(file_path)
    print(f"Đang xử lý file: {filename}")
    text = extract_text_from_file(file_path)
    if not text:
        return False, "Không trích xuất được văn bản từ file."
        
    chunks = chunk_text(text)
    if not chunks:
        return False, "Không tìm thấy đoạn văn bản hợp lệ nào."
        
    client = get_qdrant_client()
    model = get_embedding_model()
    
    points = []
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector,
            payload={"source": filename, "text": chunk}
        ))
        
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return True, f"Đã nhúng {len(chunks)} đoạn thành công."
    except Exception as e:
        return False, str(e)


# ===================== TOOLS CHO AGENT =====================

def search_documents(query: str) -> str:
    """Hàm tìm kiếm thông tin về quy chế, mô tả khoa ngành, học phí, định hướng nghề nghiệp từ tài liệu của trường."""
    try:
        client = get_qdrant_client()
        model = get_embedding_model()
        
        query_vector = model.encode(query).tolist()
        
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        if not search_result:
            return "Không tìm thấy thông tin nào liên quan trong tài liệu của trường."
            
        results = [hit.payload['text'] for hit in search_result]
        return "\n---\n".join(results)
    except Exception as e:
        return f"Lỗi khi tìm kiếm dữ liệu: {str(e)}"

def lookup_majors_by_score(score: float) -> str:
    """Hàm lọc ra danh sách TẤT CẢ các ngành, khoa và tổ hợp môn mà học sinh có thể đỗ dựa trên điểm số (tổng điểm xét tuyển học bạ)."""
    file_path = os.path.join(DATA_DIR, 'diem_chuan.json')
    if not os.path.exists(file_path):
        return "Lỗi: Không tìm thấy cơ sở dữ liệu điểm chuẩn của trường."
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        eligible_majors = []
        for item in data:
            if score >= item['diem_chuan']:
                eligible_majors.append(
                    f"- {item['ten_nganh']} (Mã: {item['ma_nganh']}, Khoa: {item['khoa']}) | Tổ hợp: {item['to_hop']} | Điểm chuẩn năm trước: {item['diem_chuan']}"
                )
                
        if not eligible_majors:
            return f"Rất tiếc, với mức điểm {score}, hiện tại không có ngành nào có điểm chuẩn năm ngoái thấp hơn hoặc bằng mức này."
            
        result = f"[DANH SÁCH CÁC NGÀNH CÓ THỂ ĐỖ VỚI ĐIỂM {score}]:\n"
        result += "\n".join(eligible_majors)
        return result
    except Exception as e:
        return f"Lỗi khi đọc file điểm chuẩn: {str(e)}"
