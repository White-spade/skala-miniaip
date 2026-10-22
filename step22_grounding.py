import asyncio
import json
import re

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from step11_write import writer_llm
from step20_clarification_hybrid import (
    AgentState as BaseAgentState,
    prepare_node, route_node, target_check_node, plan_node,
    gate_node, research_node, replan_node, analysis_node, chat_node,
    mentioned_services, unknown_services,
)


class AgentState(BaseAgentState, total=False):
    sources: list[dict]
    validation_count: int
    validation_passed: bool
    validation_feedback: str
    validation_route: str


FIELDS = ("status", "cpu_percent", "memory_percent")
MAX_REWRITES = 3


# Replan이 끝난 시점의 MCP 수집 자료에 출처 ID 부여
def source_node(state: AgentState):
    sources = []
    for item in state.get("collected", []):
        if item.get("source") not in ("monitoring", "runbook"):
            continue
        sources.append({
            "id": f"S{len(sources) + 1}",
            "kind": item["source"],
            "target": item.get("target", "unknown"),
            "incident_type": item.get("incident_type", ""),
            "data": item.get("data"),
        })
    return {"sources": sources}


def runbook_text(data) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(runbook_text(part) for part in data)
    if isinstance(data, dict):
        if "text" in data:
            return runbook_text(data["text"])
        if "content" in data:
            return runbook_text(data["content"])
        return json.dumps(data, ensure_ascii=False)
    return str(data or "")


def runbook_steps(data) -> list[str]:
    return [match.group(1).strip() for line in runbook_text(data).splitlines()
            if (match := re.match(r"^\s*\d+[.)]\s+(.+)$", line))]


# 보고서는 자유 서술이 아니라 출처를 선택하는 작은 JSON으로 생성
async def write_node(state: AgentState):
    sources = state.get("sources", [])
    feedback = state.get("validation_feedback", "")
    messages = [
        ("system",
         "당신은 AIOps 보고서의 근거 선택기입니다. 반드시 JSON 객체만 출력하세요. "
         '형식: {"facts": [{"source_id": "S1", "field": "cpu_percent", "value": 98.8}], '
         '"steps": [{"source_id": "S2", "quote": "런북의 번호를 제외한 원문 한 줄"}]}. '
         "각 monitoring 출처에서 존재하는 status, cpu_percent, memory_percent를 "
         "모두 facts에 넣고 원본 자료형과 값을 정확히 복사하세요. "
         "steps는 runbook의 번호가 붙은 절차 중 관련된 내용을 최대 3개 그대로 복사하세요. "
         "런북 절차가 있으면 최소 1개를 선택하세요. 없는 경우 빈 배열로 두세요. "
         "다른 키, 새로운 수치, 임의의 인용, 원인 단정, 마크다운은 넣지 마세요."),
        ("human", json.dumps({
            "request": state["user_input"],
            "sources": sources,
            "analysis_for_selection_only": state.get("analysis_result", {}),
            "validation_feedback": feedback,
            "previous_draft": state.get("result") if feedback else None,
        }, ensure_ascii=False)),
    ]
    response = await writer_llm.ainvoke(messages)
    return {"result": response.content}


def parse_draft(text: str) -> dict:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    draft = json.loads(text)
    if not isinstance(draft, dict) or not isinstance(draft.get("facts"), list) or not isinstance(draft.get("steps"), list):
        raise ValueError("facts와 steps 배열이 필요합니다.")
    return draft


# LLM의 숫자·출처·런북 인용을 MCP 원본과 직접 대조
def verify_draft(draft: dict, sources: list[dict]) -> list[str]:
    errors = []
    if set(draft) != {"facts", "steps"}:
        errors.append("허용되지 않은 JSON 필드")

    lookup = {source["id"]: source for source in sources}
    seen = set()
    for fact in draft["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"source_id", "field", "value"}:
            errors.append("사실 항목 형식 오류")
            continue
        sid, field, value = fact["source_id"], fact["field"], fact["value"]
        if not isinstance(sid, str) or not isinstance(field, str):
            errors.append("출처 ID 또는 필드명 오류")
            continue
        source = lookup.get(sid)
        if not source or source["kind"] != "monitoring" or field not in FIELDS:
            errors.append(f"잘못된 출처 또는 필드: {sid}/{field}")
            continue
        data = source["data"]
        if not isinstance(data, dict) or field not in data:
            errors.append(f"원본에 없는 필드: {sid}/{field}")
            continue
        expected = data[field]
        same_type = type(value) is type(expected) or (type(value) in (int, float) and type(expected) in (int, float))
        if not same_type or value != expected:
            errors.append(f"원본 값 불일치: {sid}/{field} (LLM={value}, MCP={expected})")
        if (sid, field) in seen:
            errors.append(f"중복 사실: {sid}/{field}")
        seen.add((sid, field))

    for source in sources:
        if source["kind"] == "monitoring" and isinstance(source["data"], dict):
            for field in FIELDS:
                if field in source["data"] and (source["id"], field) not in seen:
                    errors.append(f"필수 사실 누락: {source['id']}/{field}")

    if len(draft["steps"]) > 3:
        errors.append("런북 절차는 최대 3개까지 선택할 수 있습니다.")
    seen_steps = set()
    for step in draft["steps"]:
        if not isinstance(step, dict) or set(step) != {"source_id", "quote"}:
            errors.append("런북 항목 형식 오류")
            continue
        sid, quote = step["source_id"], step["quote"]
        if not isinstance(sid, str) or not isinstance(quote, str):
            errors.append("런북 출처 또는 인용문 형식 오류")
            continue
        source = lookup.get(sid)
        if not source or source["kind"] != "runbook" or quote not in runbook_steps(source["data"]):
            errors.append(f"런북 원문 불일치: {sid}")
        if (sid, quote) in seen_steps:
            errors.append(f"런북 절차 중복: {sid}")
        seen_steps.add((sid, quote))

    if any(source["kind"] == "runbook" and runbook_steps(source["data"]) for source in sources) and not draft["steps"]:
        errors.append("런북 절차가 있는데 선택하지 않았습니다.")
    if not any(source["kind"] == "monitoring" and isinstance(source["data"], dict) for source in sources):
        errors.append("검증 가능한 모니터링 근거가 없습니다.")
    return errors


def render_report(draft: dict, sources: list[dict], fallback: bool = False) -> str:
    lines = ["# AIOps 조사 결과"]
    if fallback:
        lines.append("3회 재작성 후에도 검증 실패: LLM 초안을 폐기하고 MCP 원본 자료만 표시합니다.")
    lines.append("\n## 모니터링 (검증된 관측값)")
    for source in sources:
        if source["kind"] != "monitoring" or not isinstance(source["data"], dict):
            continue
        data, sid = source["data"], source["id"]
        fields = [f"상태 {data['status']}"] if "status" in data else []
        fields += [f"CPU {data['cpu_percent']}%"] if "cpu_percent" in data else []
        fields += [f"Memory {data['memory_percent']}%"] if "memory_percent" in data else []
        lines.append(f"- {source['target']}: {', '.join(fields)} [{sid}]")
        for alert in data.get("alerts", []):
            lines.append(f"  - 알림 원문: {alert} [{sid}]")
        if data.get("timestamp"):
            lines.append(f"  - 조회 시각: {data['timestamp']} [{sid}]")

    lines.append("\n## Runbook 권장 절차 (실행하지 않음)")
    if draft["steps"]:
        for step in draft["steps"]:
            lines.append(f"- {step['quote']} [{step['source_id']}]")
    else:
        lines.append("- 확인 가능한 절차 없음")

    lines.append("\n## 출처")
    for source in sources:
        suffix = f" / {source['incident_type']}" if source["incident_type"] else ""
        lines.append(f"- [{source['id']}] MCP {source['kind']} / {source['target']}{suffix}")
    lines.append("\n※ 관측값과 인용의 원문 일치만 검증했습니다. 장애 원인과 조치 효과는 확정하지 않았습니다.")
    return "\n".join(lines)


def conservative_report(sources: list[dict]) -> str:
    steps = []
    for source in sources:
        if source["kind"] == "runbook":
            steps.extend({"source_id": source["id"], "quote": quote}
                         for quote in runbook_steps(source["data"])[:2])
    return render_report({"facts": [], "steps": steps[:3]}, sources, fallback=True)


# LLM 검증 대신 Python으로 직접 검증; 실패하면 Write만 재실행
# 최초 작성 1회 + 재작성 최대 3회 (검증은 최대 4회)
def validate_node(state: AgentState):
    count = state.get("validation_count", 0) + 1
    sources = state.get("sources", [])
    try:
        draft = parse_draft(state["result"])
        errors = verify_draft(draft, sources)
    except (ValueError, TypeError) as exc:
        draft, errors = None, [f"JSON 파싱 실패: {exc}"]

    passed = not errors
    if passed:
        route = "end"
        result = render_report(draft, sources)
    elif count <= MAX_REWRITES:
        route = "write"
        result = state["result"]
    else:
        route = "end"
        result = conservative_report(sources)

    return {
        "validation_count": count,
        "validation_passed": passed,
        "validation_feedback": "; ".join(errors)[:800],
        "validation_route": route,
        "result": result,
    }


builder = StateGraph(AgentState)
for name, fn in (
    ("prepare", prepare_node), ("route", route_node),
    ("target_check", target_check_node), ("plan", plan_node),
    ("gate", gate_node), ("research", research_node),
    ("replan", replan_node), ("sources", source_node),
    ("analysis", analysis_node), ("write", write_node),
    ("validate", validate_node), ("chat", chat_node),
):
    builder.add_node(name, fn)

builder.add_edge(START, "prepare")
builder.add_conditional_edges("prepare", lambda s: s["next_step"], {
    "route": "route", "target_check": "target_check", "end": END,
})
builder.add_conditional_edges("route", lambda s: s["intent"], {
    "diagnosis": "target_check", "chat": "chat",
})
builder.add_conditional_edges("target_check", lambda s: s["next_step"], {
    "plan": "plan", "end": END,
})
builder.add_edge("plan", "gate")
builder.add_conditional_edges("gate", lambda s: "research" if s["tasks"] else "end", {
    "research": "research", "end": END,
})
builder.add_edge("research", "replan")
builder.add_conditional_edges("replan", lambda s: s["replan_route"], {
    "research": "research", "analysis": "sources", "end": END,
})
builder.add_edge("sources", "analysis")
builder.add_edge("analysis", "write")
builder.add_edge("write", "validate")
builder.add_conditional_edges("validate", lambda s: s["validation_route"], {
    "write": "write", "end": END,
})
builder.add_edge("chat", END)

graph = builder.compile()


async def main():
    query = input("요청: ").strip()
    result = await graph.ainvoke({"user_input": query})
    print("\n" + result.get("result", "응답 없음"))
    print("\n검증 횟수:", result.get("validation_count", 0))
    print("검증 통과:", result.get("validation_passed", "해당 없음"))
    if result.get("validation_feedback"):
        print("검증 피드백:", result["validation_feedback"])


if __name__ == "__main__":
    asyncio.run(main())
