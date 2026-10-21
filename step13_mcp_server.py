from step12_mcp_server import mcp


# 장애 분석 프롬프트 등록
@mcp.prompt()
def incident_analysis_prompt(system_name: str, symptom: str) -> str:
    """장애 진단 표준 절차 프롬프트"""

    return f"""다음 장애에 대해 표준 진단 절차를 수행하세요.

## 장애 정보
- 시스템: {system_name}
- 증상: {symptom}

## 진단 절차
1. get_monitoring_status Tool로 시스템 상태를 조회하세요.
2. 관련 Runbook을 확인하세요.
3. 가능한 원인과 권장 조치를 제시하세요.
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")