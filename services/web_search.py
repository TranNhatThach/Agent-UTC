from duckduckgo_search import DDGS

def web_search(query: str) -> str:
    """
    Tìm kiếm thông tin trên internet bằng DuckDuckGo. Chỉ sử dụng làm giải pháp dự phòng cuối cùng khi search_documents không trả về kết quả hoặc khi người dùng hỏi về tin tức thời sự bên ngoài không có trong tài liệu của trường Đại học Giao thông Vận tải.
    
    Args:
        query: Từ khóa hoặc câu hỏi cần tìm kiếm bằng tiếng Việt.
    """
    try:
        print(f"[Tool: Web Search] Đang tìm kiếm: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "Không tìm thấy thông tin trên internet."
        
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Tiêu đề: {r.get('title')}\nĐoạn trích: {r.get('body')}\nLink: {r.get('href')}\n"
            )
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Lỗi khi thực hiện tìm kiếm web: {e}"
