import ollama
from config import AGENT_SYSTEM_PROMPT
from rag_utils import search_documents, lookup_majors_by_score
from duckduckgo_search import DDGS

# Bộ chuyển đổi sửa lỗi chính tả/font chữ Unicode từ model local
def clean_vietnamese_query(text: str) -> str:
    if not isinstance(text, str):
        return text
    corrections = {
        "Trðng": "Trường",
        "Đƻ": "Đại",
        "Hóc": "Học",
        "tái": "tải",
        "Ęnh ých": "chính thức",
        "Ęnh": "định",
        "ých": "hướng"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def web_search(query: str) -> str:
    """
    Tìm kiếm thông tin trên internet bằng DuckDuckGo khi cơ sở dữ liệu tài liệu nội bộ không có thông tin cần thiết.
    Chỉ sử dụng khi người dùng hỏi các tin tức, quy chế tuyển sinh chung, hoặc các thông tin thực tế bên ngoài LƯU Ý CHỈ VỀ THÔNG TIN VỀ TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI THƯƠNG HIỆU HÀ NỘI.
    
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

# Đăng ký các công cụ Python cho LLM Agent
AGENT_TOOLS = [search_documents, lookup_majors_by_score, web_search]

def run_ollama_agent(message: str, model_name: str, history: list = None):
    if history is None:
        history = []
        
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": message})
    
    # Gửi tin nhắn kèm tools cho Ollama
    response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=AGENT_TOOLS
    )
    
    # Kiểm tra xem Model có quyết định gọi tool nào không
    if response.message.tool_calls:
        # Thêm câu trả lời (có chứa lệnh gọi tool) của assistant vào lịch sử
        if hasattr(response.message, 'model_dump'):
            messages.append(response.message.model_dump())
        else:
            messages.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": getattr(response.message, 'tool_calls', None)
            })
        
        # Chạy từng tool mà AI yêu cầu
        for tool_call in response.message.tool_calls:
            func_name = tool_call.function.name
            args = tool_call.function.arguments
            
            print(f"[Agent] Gọi tool: {func_name} với args: {args}")
            
            result_str = ""
            if func_name == 'search_documents':
                cleaned_query = clean_vietnamese_query(args.get('query', ''))
                result_str = search_documents(query=cleaned_query)
            elif func_name == 'lookup_majors_by_score':
                score_val = float(args.get('score', 0))
                result_str = lookup_majors_by_score(score=score_val)
            elif func_name == 'web_search':
                cleaned_query = clean_vietnamese_query(args.get('query', ''))
                result_str = web_search(query=cleaned_query)
                
            messages.append({
                "role": "tool",
                "name": func_name,
                "content": result_str
            })
            
        # Gửi lại toàn bộ lịch sử (bao gồm kết quả tool) để LLM tổng hợp câu trả lời
        final_response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return final_response.message.content
    else:
        return response.message.content
