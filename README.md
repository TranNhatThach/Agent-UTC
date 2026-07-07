# UTC Admission AI Agent (Thuần Local RAG System)

Hệ thống Chatbot tư vấn tuyển sinh và tra cứu điểm học bạ thông minh hoạt động thuần local (offline) dành cho **Trường Đại học Giao thông Vận tải (UTC)**. Dự án kết hợp công nghệ **FastAPI**, **RAG (Retrieval-Augmented Generation)** thông qua **Qdrant Vector Database**, mô hình nhúng **BGE-M3** và LLM chạy trên **Ollama**.

---

## 🚀 Các Tính Năng Nổi Bật

1. **Local RAG (Hỏi đáp tài liệu)**: Vector hóa các tài liệu nội bộ tuyển sinh (`PDF`, `DOCX`, `TXT`) bằng mô hình nhúng tiên tiến `BAAI/bge-m3` và cơ sở dữ liệu vector `Qdrant` chạy cục bộ.
2. **Tra cứu điểm & phân tích tổ hợp học bạ**:
   - Lọc tự động các ngành có cơ hội đỗ dựa trên điểm xét tuyển từ cơ sở dữ liệu điểm chuẩn các năm.
   - Hỗ trợ phân tích điểm theo từng môn học để tư vấn cơ hội đỗ theo các tổ hợp khối ngành tương ứng (`A00`, `A01`, `D01`, `D07`).
3. **Tìm kiếm Internet dự phòng (DuckDuckGo)**: Tự động kích hoạt công cụ tra cứu thông tin trên Internet qua DuckDuckGo khi tài liệu nội bộ không có dữ liệu cần thiết.
4. **Bảo vệ Cache chống Nhiễm độc (Cache Poisoning)**: Caching ở cấp độ dữ liệu Vector DB thay vì lưu trực tiếp câu trả lời của LLM. Giúp hệ thống hoạt động nhanh và an toàn, tránh lặp lại câu trả lời bị ảo giác (hallucination) của LLM.
5. **Rate Limiting chống spam**: Giới hạn tần suất gọi API (tối đa 15 request/phút/IP) sử dụng **Redis pipeline** để bảo vệ tài nguyên server.
6. **Session-based Chat History**: Lịch sử chat được lưu trữ tạm thời trong RAM của client, tự động reset hoàn toàn khi tải lại trang (F5) hoặc tắt tab.

---

## 🛠️ Yêu Cầu Hệ Thống

- **Python 3.10+**
- **Docker Desktop** (Để chạy Redis làm Rate Limiter và Cache)
- **Ollama** (Ứng dụng chạy mô hình ngôn ngữ lớn LLM local)

---

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Chuẩn bị Môi trường
Tải mã nguồn về máy, mở Terminal di chuyển vào thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 2. Chạy dịch vụ Redis (Docker)
```bash
docker run --name redis-cache -p 6379:6379 -d redis
```

### 3. Tạo mô hình local Ollama
1. Tải và cài đặt ứng dụng [Ollama](https://ollama.com).
2. Chuẩn bị file `Modelfile` và chạy lệnh sau để build mô hình local `utc-score-tuvan`:
```bash
ollama create utc-score-tuvan -f Modelfile
```

### 4. Cấu hình biến môi trường
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```

### 5. Nạp tài liệu tuyển sinh (RAG Setup)
Đặt các tài liệu tuyển sinh định dạng `.pdf`, `.docx`, `.txt` vào thư mục `data/`, sau đó khởi chạy script nạp dữ liệu:
```bash
python rag_setup.py
```

### 6. Khởi động Server Backend
```bash
python main.py
```
Server FastAPI sẽ được khởi chạy tại cổng `5000`: `http://127.0.0.1:5000`

### 7. Trải nghiệm giao diện
Mở trực tiếp file `index.html` bằng trình duyệt của bạn hoặc đưa lên một hosting tĩnh để sử dụng giao diện chat client kết nối trực tiếp đến backend.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
utc-rag-project/
│
├── .env                  # Lưu biến cấu hình cục bộ (API URL, model name, v.v.)
├── main.py               # File chạy chính của FastAPI & định nghĩa các API endpoint
├── requirements.txt      # Danh sách thư viện Python cần cài đặt
├── config.py             # Quản lý cấu hình chung, System Prompt và kết nối DB/Redis
├── rate_limiter.py       # Logic giới hạn tần suất gọi API bằng Redis
├── rag_setup.py          # Script tự động quét thư mục data và nạp vector vào Qdrant
├── index.html            # Giao diện chat client (HTML, CSS và Javascript thuần)
│
├── services/             # Thư mục chứa các module nghiệp vụ tách biệt
│   ├── document_worker.py# Xử lý đọc file PDF/Docx/Txt và Chunking văn bản
│   ├── vector_store.py   # Kết nối Qdrant, nhúng vector, tìm kiếm ngữ cảnh & tra cứu điểm
│   ├── llm_service.py    # Quản lý kết nối Ollama Agent, kiểm soát tool calling & hội thoại
│   └── web_search.py     # Tìm kiếm bổ sung bằng DuckDuckGo khi cơ sở dữ liệu nội bộ thiếu thông tin
│
└── data/                 # Thư mục chứa tài liệu tuyển sinh đầu vào và diem_chuan.json
```
