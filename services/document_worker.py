import os
import uuid
import PyPDF2
from docx import Document
from qdrant_client.models import PointStruct
from services.vector_store import get_qdrant_client, get_embedding_model, COLLECTION_NAME

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
