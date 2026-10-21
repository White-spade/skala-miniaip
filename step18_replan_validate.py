import asyncio
import json
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from step05_llm_routing import llm
from step11_write import writer_llm
from step15_analysis import (
    AgentState as BaseAgentState,
    route_node, plan_node, gate_node, analysis_node,
    write_node as original_write_node, chat_node,
)
from step16_react import research_node as original_research_node


# 기존 State 확장
class AgentState(BaseAgentState, total=False):
    replan_count: int
    replan_route: str
    validation_count: int
    validation_route: str
    validation_passed: bool
    validation_feedback: str


# 새 Task만 실행하고 기존 조사 결과 유지
async def research_node(state: AgentState):
    pending = [
        t for t in state["tasks"]
        if t["agent"] == "research" and t["status"] == "pending"
    ]
    if not pending:
        return {}

    response = await original_research_node({**state, "tasks": pending})
    updated = {t["id"]: t for t in response["tasks"]}
    tasks = [updated.get(t["id"], t) for t in state["tasks"]]

    collected = state.get("collected", []) + response.get("collected", [])
    unique = {}
    for item in collected:
        key = (item.get("source"), item.get("target"), item.get("incident_type"))
        unique[key] = item

    return {"tasks": tasks, "collected": list(unique.values())}


# 수집 결과를 보고 추가 조사 여부 결정
async def replan_node(state: AgentState):
    count = state.get("replan_count", 0)
    tasks = [t.copy() for t in state["tasks"]]
    monitoring = [
        item for item in state.get("collected", [])
        if item.get("source") == "monitoring"
    ]

    # 조사 결과가 전혀 없으면 실패한 작업을 한 번만 재시도
    if not monitoring:
        failed = [t for t in tasks if t["agent"] == "research" and t["status"] == "failed"]
        if failed and count == 0:
            for task in failed:
                task["status"] = "pending"
            return {"tasks": tasks, "replan_count": 1, "replan_route": "research"}
        return {"replan_route": "end", "result": "모니터링 데이터를 수집하지 못했습니다."}

    # 추가 조사는 최대 한 번
    if count >= 1:
        return {"replan_route": "analysis"}

    services = json.loads(Path(__file__).with_name("monitoring.json").read_text(encoding="utf-8"))
    observed = {item["target"] for item in monitoring}
    planned = {t["target"] for t in tasks if t["agent"] == "research"}
    alerts = " ".join(str(alert) for item in monitoring for alert in item["data"].get("alerts", []))

    # 관측된 Alert에 등장한 등록 서비스만 추가 후보로 허용
    candidates = [name for name in services if name in alerts and name not in observed and name not in planned]
    if not candidates:
        return {"replan_route": "analysis"}

    messages = [
        ("system",
         "당신은 AIOps 재조사 판단자입니다. "
         "현재 자료만으로 사용자 요청에 답하기 부족하면 추가 서비스를 선택하세요. "
         "추가 조사가 불필요하면 next_target은 null입니다. "
         "반드시 후보 목록의 서비스만 선택하세요. "
         'JSON {"next_target": "서비스명 또는 null"} 형식으로만 답하세요.'),
        ("human", json.dumps({
            "request": state["user_input"],
            "monitoring": monitoring,
            "candidates": candidates,
        }, ensure_ascii=False))
    ]

    response = await llm.ainvoke(messages)
    try:
        target = json.loads(response.content).get("next_target")
    except (ValueError, TypeError):
        target = None

    if target not in candidates:
        return {"replan_route": "analysis"}

    tasks.append({
        "id": f"task-{len(tasks) + 1}",
        "agent": "research",
        "kind": "monitoring",
        "target": target,
        "status": "pending"
    })
    return {"tasks": tasks, "replan_count": count + 1, "replan_route": "research"}


# 검증 실패 시 직전 보고서와 피드백을 반영해 재작성
async def write_node(state: AgentState):
    feedback = state.get("validation_feedback")
    if not feedback:
        return await original_write_node(state)

    messages = [
        ("system",
         "당신은 AIOps 보고서 작성자입니다. "
         "직전 보고서의 오류를 수정하세요. 관측 사실과 원인 가설을 구분하고, "
         "측정되지 않은 수치나 시각을 만들어내지 마세요."),
        ("human", json.dumps({
            "user_request": state["user_input"],
            "collected": state.get("collected", []),
            "analysis": state.get("analysis_result", {}),
            "previous_report": state["result"],
            "feedback": feedback,
        }, ensure_ascii=False))
    ]
    response = await writer_llm.ainvoke(messages)
    return {"result": response.content}


# 검증 실패 시 관측 사실만 반환하는 최소 대체 응답
def safe_report(state: AgentState):
    lines = ["보고서 검증에 실패해 관측 데이터만 제공합니다."]
    for item in state.get("collected", []):
        if item.get("source") != "monitoring":
            continue
        data = item["data"]
        lines.append(
            f"{item['target']}: {data.get('status', 'unknown')}, "
            f"CPU {data.get('cpu_percent', '-')}%, "
            f"Memory {data.get('memory_percent', '-')}%"
        )
        lines.extend(f"- {alert}" for alert in data.get("alerts", []))
    return "\n".join(lines)


# 보고서의 근거 및 과도한 단정 여부 검증
async def validate_node(state: AgentState):
    messages = [
        ("system",
         "당신은 AIOps 보고서 검증자입니다. "
         "원본 자료에 없는 수치나 장애 발생 시각이 사실처럼 쓰였거나, "
         "추론한 원인을 확정 사실처럼 단정하면 실패로 판단하세요. "
         "가설로 명확히 표현한 추론은 허용하세요. "
         '반드시 JSON {"valid": true 또는 false, "feedback": "검증 의견"}로 답하세요.'),
        ("human", json.dumps({
            "collected": state.get("collected", []),
            "report": state["result"],
        }, ensure_ascii=False))
    ]
    response = await llm.ainvoke(messages)
    try:
        decision = json.loads(response.content)
    except (ValueError, TypeError):
        decision = {}

    passed = decision.get("valid") is True
    feedback = str(decision.get("feedback") or "검증 결과를 확인할 수 없습니다.")
    count = state.get("validation_count", 0) + 1

    if passed:
        route = "end"
        result = state["result"]
    elif count < 2:
        route = "write"
        result = state["result"]
    else:
        route = "end"
        result = safe_report(state)

    return {
        "validation_passed": passed,
        "validation_feedback": "" if passed else feedback,
        "validation_count": count,
        "validation_route": route,
        "result": result,
    }


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("gate", gate_node)
builder.add_node("research", research_node)
builder.add_node("replan", replan_node)
builder.add_node("analysis", analysis_node)
builder.add_node("write", write_node)
builder.add_node("validate", validate_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route", lambda s: "plan" if s["intent"] == "diagnosis" else "chat",
    {"plan": "plan", "chat": "chat"}
)
builder.add_edge("plan", "gate")
builder.add_conditional_edges(
    "gate", lambda s: "research" if s["tasks"] else "end",
    {"research": "research", "end": END}
)
builder.add_edge("research", "replan")
builder.add_conditional_edges(
    "replan", lambda s: s["replan_route"],
    {"research": "research", "analysis": "analysis", "end": END}
)
builder.add_edge("analysis", "write")
builder.add_edge("write", "validate")
builder.add_conditional_edges(
    "validate", lambda s: s["validation_route"],
    {"write": "write", "end": END}
)
builder.add_edge("chat", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="step18_graph.png")


async def main():
    user_input = input("요청: ")
    result = await graph.ainvoke({"user_input": user_input})

    print("추가 조사 횟수:", result.get("replan_count", 0))
    print("보고서 검증 횟수:", result.get("validation_count", 0))
    print("검증 통과:", result.get("validation_passed", "검증 없음"))
    print("\n=== 최종 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())
