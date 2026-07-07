import os
import glob
from services.document_worker import process_and_embed_file

DATA_DIR = "data"

def setup_rag():
    print("Bắt đầu nạp toàn bộ thư mục data vào Qdrant...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Đã tạo thư mục {DATA_DIR}. Vui lòng thêm các file vào đây.")
        return

    files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    supported_files = [f for f in files if f.endswith(('.txt', '.pdf', '.docx'))]
    
    if not supported_files:
        print(f"Không tìm thấy file nào hợp lệ (.txt, .pdf, .docx) trong {DATA_DIR}.")
        return

    print(f"Tìm thấy {len(supported_files)} file hợp lệ.")
    
    try:
        for file_path in supported_files:
            process_and_embed_file(file_path)

        print("✅ Đã nạp xong toàn bộ dữ liệu vào Qdrant thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    setup_rag()
