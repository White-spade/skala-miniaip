from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from step01_monitoring import get_monitoring_status


app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"]
)

# 요청 데이터 형식
class MonitorRequest(BaseModel):
    system_name: str


# 모니터링 API
@app.post("/monitoring")
def monitoring(request: MonitorRequest):
    data = get_monitoring_status(request.system_name)

    if data is None:
        raise HTTPException(status_code=404, detail="서비스 없음")

    return {
        "system_name": request.system_name,
        "data": data
    }