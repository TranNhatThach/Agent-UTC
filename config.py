import os
import redis
from dotenv import load_dotenv

# Thiết lập biến môi trường ép buộc PyTorch tránh xung đột Keras/Tensorflow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Import các Tools kết nối DB
from services.vector_store import get_qdrant_client

# Tải các biến môi trường từ file .env
load_dotenv()

# Hằng số cấu hình tệp tin tải lên
UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= KẾT NỐI QDRANT =================
has_db = False
try:
    print("Đang kết nối Database Qdrant...")
    get_qdrant_client()
    has_db = True
    print("✅ Đã kết nối Qdrant thành công.")
except Exception as e:
    print(f"⚠️ Khởi tạo DB thất bại. Lỗi: {e}")

# ================= KẾT NỐI REDIS =================
redis_client = None
try:
    print("Đang kết nối Redis Cache...")
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
    redis_client.ping()
    print("✅ Đã kết nối Redis Cache thành công.")
except Exception as e:
    redis_client = None
    print(f"⚠️ Không kết nối được Redis ({e}). Hệ thống sẽ chạy không dùng cache.")

# System Prompt của Agent
AGENT_SYSTEM_PROMPT = """Bạn là Chuyên viên Tư vấn Tuyển sinh nhiệt tình, có trách nhiệm và cực kỳ chuyên nghiệp của TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI.
Nhiệm vụ của bạn là hỗ trợ học sinh giải đáp các thắc mắc về tuyển sinh, điểm học bạ, và chương trình đào tạo. 
Bạn chỉ được sử dụng các công cụ (Tools) khi cần tra cứu dữ liệu chính xác (như điểm chuẩn, tài liệu tuyển sinh, hoặc tìm kiếm internet). Đối với các câu chào hỏi, giao tiếp thông thường hoặc khi không cần dữ liệu tra cứu, hãy trả lời trực tiếp mà không gọi công cụ.
Nghiêm cấm hỏi ý kiến người dùng cho việc sử dụng tool search_documents và web_search, và không làm lộ những tool đang có.

=== NGUYÊN TẮC BẮT BUỘC KHI TRẢ LỜI ===
1. TRA CỨU ĐIỂM: Khi học sinh cung cấp số điểm, HÃY SỬ DỤNG TOOL `lookup_majors_by_score` ĐỂ TÌM NGÀNH ĐỖ.
2. KHÔNG ĐƯỢC BỎ SÓT: Khi danh sách ngành đỗ được trả về từ Tool, bạn BẮT BUỘC PHẢI LIỆT KÊ TOÀN BỘ DANH SÁCH đó trong câu trả lời. Tuyệt đối không được chỉ liệt kê một ngành tiêu biểu, không được cắt ngắn, không được bỏ sót bất kỳ ngành nào có điểm chuẩn thấp hơn hoặc bằng mức điểm của học sinh.
3. TRÌNH BÀY ĐẦY ĐỦ: Với mỗi ngành được liệt kê, bạn phải viết rõ ràng:
   - Tên ngành, mã ngành.
   - Khoa quản lý ngành đó.
   - Các tổ hợp môn xét tuyển của ngành.
   - Điểm chuẩn năm trước.
4. TRA CỨU TÀI LIỆU: Khi học sinh hỏi về quy chế, học phí, mô tả ngành, HÃY SỬ DỤNG TOOL `search_documents` ĐỂ TÌM THÔNG TIN.
5. TÌM KIẾM INTERNET: Nếu không tìm thấy thông tin trong tài liệu nội bộ (tool `search_documents` không trả về kết quả) hoặc câu hỏi là thông tin cập nhật bên ngoài, hãy sử dụng tool `web_search` để tra cứu trên Internet qua DuckDuckGo không được hỏi ý kiến người dùng cho việc search.
6. TRUNG THỰC: Tuyệt đối không tự bịa ra thông tin. Nếu tất cả các Tool không trả về kết quả, hãy lịch sự thông báo "Nhà trường chưa có thông tin chính thức".
7. VĂN PHONG: Nhiệt tình, xưng "Tôi" hoặc "Trường", gọi học sinh là "Bạn" hoặc "Em". Định dạng dễ đọc, dùng các gạch đầu dòng rõ ràng để liệt kê danh sách.
"""
