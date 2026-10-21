import random
from datetime import datetime

from fastmcp import FastMCP
from step01_monitoring import get_monitoring_status as load_status


# MCP 서버 생성
mcp = FastMCP("ai-minip-ops")


# 모니터링 Tool 등록
@mcp.tool()
def get_monitoring_status(system_name: str) -> dict:
    """서비스의 모니터링 상태를 조회합니다."""

    data = load_status(system_name)

    if data is None:
        return {
            "system_name": system_name,
            "status": "unknown",
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "last_checked": datetime.now().isoformat(),
            "alerts": ["등록되지 않은 서비스입니다."]
        }

    # 측정값에 랜덤 변동 적용
    jitter = random.uniform(-3.0, 8.0)

    return {
        "system_name": system_name,
        "status": data["status"],
        "cpu_percent": round(data["cpu_base"] + jitter, 1),
        "memory_percent": round(data["memory_base"] + jitter / 2, 1),
        "last_checked": datetime.now().isoformat(),
        "alerts": data["alerts"]
    }


# stdio 서버 실행
if __name__ == "__main__":
    mcp.run(transport="stdio")