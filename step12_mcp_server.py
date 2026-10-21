from pathlib import Path

from step09_mcp_server import mcp


# Runbook 파일 목록
RUNBOOKS = {
    "cpu-spike": "step12_cpu_spike.md",
    "network-timeout": "step12_network_timeout.md"
}


# Runbook Resource 등록
@mcp.resource("runbook://incident-response/{incident_type}")
def get_incident_runbook(incident_type: str) -> str:
    """장애 유형에 해당하는 Runbook을 반환합니다."""

    filename = RUNBOOKS.get(incident_type)

    if filename is None:
        return "등록되지 않은 Runbook입니다."

    path = Path(__file__).with_name(filename)

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")