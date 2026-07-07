import os
import json
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Configuration constants
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

def search_documents(query: str) -> str:
    """Hàm tìm kiếm thông tin chi tiết về quy chế tuyển sinh, học phí, định hướng ngành nghề và thông tin đào tạo từ tài liệu chính thức của Đại học Giao thông Vận tải. Chỉ gọi khi người dùng hỏi các câu hỏi chuyên môn cần dữ liệu nội bộ."""
    from config import redis_client
    # Check cache for Vector DB search result to avoid cache poisoning on LLM answer
    cache_key = f"db_search:{hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()}" if redis_client else None
    if redis_client and cache_key:
        try:
            cached_res = redis_client.get(cache_key)
            if cached_res:
                print(f"[Redis Cache] Đánh trúng cache dữ liệu Vector DB cho query: '{query}'")
                return cached_res
        except Exception as cache_err:
            print(f"[Redis Warning] Lỗi đọc cache tài liệu: {cache_err}")

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
        result_str = "\n---\n".join(results)

        # Cache the retrieved database context in Redis (expires in 1 hour)
        if redis_client and cache_key and results:
            try:
                redis_client.set(cache_key, result_str, ex=3600)
                print(f"[Redis Cache] Đã lưu kết quả tìm kiếm Vector DB mới vào cache.")
            except Exception as cache_err:
                print(f"[Redis Warning] Lỗi ghi cache tài liệu: {cache_err}")

        return result_str
    except Exception as e:
        return f"Lỗi khi tìm kiếm dữ liệu: {str(e)}"

def lookup_majors_by_score(score: float) -> str:
    """Hàm lọc ra danh sách các ngành học sinh có thể đỗ dựa trên điểm số thi hoặc học bạ THPT. Chỉ gọi khi người dùng cung cấp điểm số tổng thể cụ thể (ví dụ: 'em được 24 điểm') để tra cứu ngành đỗ."""
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

def analyze_subject_scores(math: float, physics: float = 0.0, chemistry: float = 0.0, english: float = 0.0, literature: float = 0.0) -> str:
    """Hàm phân tích điểm số chi tiết theo từng môn học (Toán, Lý, Hóa, Anh, Văn) để tính điểm xét tuyển theo các tổ hợp A00, A01, D01, D07 và đề xuất ngành đỗ. Chỉ gọi khi người dùng cung cấp điểm số của các môn học riêng lẻ."""
    file_path = os.path.join(DATA_DIR, 'diem_chuan.json')
    if not os.path.exists(file_path):
        return "Lỗi: Không tìm thấy cơ sở dữ liệu điểm chuẩn của trường."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        combinations = {
            "A00": {"name": "A00 (Toán, Lý, Hóa)", "score": math + physics + chemistry},
            "A01": {"name": "A01 (Toán, Lý, Anh)", "score": math + physics + english},
            "D01": {"name": "D01 (Toán, Văn, Anh)", "score": math + literature + english},
            "D07": {"name": "D07 (Toán, Hóa, Anh)", "score": math + chemistry + english}
        }

        analysis_result = []
        for code, comb in combinations.items():
            total_score = comb["score"]
            eligible = []
            for item in data:
                # Kiểm tra xem tổ hợp này có được ngành đó chấp nhận không
                allowed_combs = [c.strip() for c in item['to_hop'].split(',')]
                if code in allowed_combs and total_score >= item['diem_chuan']:
                    eligible.append(
                        f"  + {item['ten_nganh']} ({item['ma_nganh']}) - Điểm chuẩn: {item['diem_chuan']}"
                    )
            
            analysis_result.append(f"Tổ hợp {comb['name']}: {round(total_score, 2)} điểm")
            if eligible:
                analysis_result.extend(eligible)
            else:
                analysis_result.append("  (Không có ngành nào phù hợp hoặc điểm chuẩn năm trước cao hơn)")
            analysis_result.append("")

        return "\n".join(analysis_result)
    except Exception as e:
        return f"Lỗi trong quá trình phân tích điểm tổ hợp: {str(e)}"
