import ollama
import re
from config import AGENT_SYSTEM_PROMPT
from services.vector_store import search_documents, lookup_majors_by_score, analyze_subject_scores
from services.web_search import web_search

def is_pure_greeting(message: str) -> bool:
    cleaned = message.strip().lower()
    pattern = r'^(hello|hi|xin chào|chào bạn|chào|helo|halô|alo|lô|hế lô)[!\.\?\s~]*$'
    return bool(re.match(pattern, cleaned))

def correct_vietnamese_query_via_llm(text: str, model_name: str) -> str:
    if not text or len(text.strip()) < 3:
        return text
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Bạn là chuyên gia sửa lỗi font, bảng mã Unicode dựng sẵn/tổ hợp và chính tả tiếng Việt. Hãy chuyển đổi câu truy vấn lỗi hoặc viết sai của người dùng thành câu truy vấn tiếng Việt chuẩn chỉnh, tự nhiên. Chỉ trả về kết quả đã sửa, không giải thích, không thêm dấu nháy."
                },
                {"role": "user", "content": text}
            ]
        )
        corrected = response.message.content.strip().strip('"').strip("'")
        print(f"[Corrector] Gốc: '{text}' -> Sửa: '{corrected}'")
        return corrected
    except Exception as e:
        print(f"[Corrector Warning] Lỗi sửa từ khóa: {e}")
        return text

# Register the Python tools for the LLM Agent
AGENT_TOOLS = [search_documents, lookup_majors_by_score, web_search, analyze_subject_scores]

def run_ollama_agent(message: str, model_name: str, history: list = None):
    if history is None:
        history = []
        
    # Check if the user message is a pure greeting
    if is_pure_greeting(message):
        print(f"[Greeting Check] Phát hiện chào hỏi thuần túy. Phản hồi trực tiếp.")
        greeting_messages = [
            {"role": "system", "content": "Bạn là chuyên viên tư vấn tuyển sinh Đại học Giao thông Vận tải. Hãy chào lại học sinh một cách ngắn gọn, thân thiện và nhiệt tình, sẵn sàng hỗ trợ."}
        ]
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            greeting_messages.append({"role": role, "content": msg.get("content", "")})
        greeting_messages.append({"role": "user", "content": message})
        
        response = ollama.chat(
            model=model_name,
            messages=greeting_messages
        )
        return response.message.content

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": message})
    
    # Send messages along with tool signatures to Ollama
    response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=AGENT_TOOLS
    )
    
    # Check if the model decided to call any tools
    if response.message.tool_calls:
        # Add assistant's response (with tool calls) to messages
        if hasattr(response.message, 'model_dump'):
            messages.append(response.message.model_dump())
        else:
            messages.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": getattr(response.message, 'tool_calls', None)
            })
        
        # Run each requested tool
        for tool_call in response.message.tool_calls:
            func_name = tool_call.function.name
            args = tool_call.function.arguments
            
            print(f"[Agent] Gọi tool: {func_name} với args: {args}")
            
            result_str = ""
            if func_name == 'search_documents':
                raw_query = args.get('query', '')
                result_str = search_documents(query=raw_query)
            elif func_name == 'lookup_majors_by_score':
                score_val = float(args.get('score', 0))
                result_str = lookup_majors_by_score(score=score_val)
            elif func_name == 'web_search':
                raw_query = args.get('query', '')
                result_str = web_search(query=raw_query)
            elif func_name == 'analyze_subject_scores':
                math_val = float(args.get('math', 0))
                physics_val = float(args.get('physics', 0))
                chemistry_val = float(args.get('chemistry', 0))
                english_val = float(args.get('english', 0))
                literature_val = float(args.get('literature', 0))
                result_str = analyze_subject_scores(
                    math=math_val,
                    physics=physics_val,
                    chemistry=chemistry_val,
                    english=english_val,
                    literature=literature_val
                )
                
            messages.append({
                "role": "tool",
                "name": func_name,
                "content": result_str
            })
            
        # Re-send all history (including tool results) to summarize the final response
        final_response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return final_response.message.content
    else:
        return response.message.content

