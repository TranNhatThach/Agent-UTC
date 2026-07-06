from fastapi import Request, HTTPException
from config import redis_client

# Middleware Rate Limiter kiểm soát số lượng request bằng Redis
async def rate_limiter(request: Request):
    if not redis_client:
        return
        
    try:
        client_ip = request.client.host if request.client else "127.0.0.1"
        key = f"rate_limit:{client_ip}"
        
        # Giới hạn: Tối đa 15 request trong vòng 60 giây cho mỗi IP
        limit = 15
        window = 60
        
        current_requests = redis_client.get(key)
        if current_requests and int(current_requests) >= limit:
            raise HTTPException(
                status_code=429, 
                detail="Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau 1 phút."
            )
            
        pipe = redis_client.pipeline()
        pipe.incr(key)
        if not current_requests:
            pipe.expire(key, window)
        pipe.execute()
    except HTTPException:
        raise
    except Exception as limit_err:
        print(f"[Rate Limit Warning] Lỗi kiểm tra giới hạn: {limit_err}")
