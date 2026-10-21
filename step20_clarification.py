import asyncio
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from step18_replan_validate import (
    AgentState as BaseAgentState,
    route_node,
    plan_node as original_plan_node,
    gate_node,
    research_node,
    replan_node,
    analysis_node,
    write_node,
    validate_node,
    chat_node,
)


# 기존 State 확장
class AgentState(BaseAgentState, total=False):
    pending_query: str
    clarification_attempts: int
    next_step: str
    approved_targets: list[str]


# JSON에 등록된 서비스 목록
SERVICES = list(json.loads(
    Path(__file__).with_name("monitoring.json").read_text(encoding="utf-8")
))


def mentioned_services(text: str) -> list[str]:
    return [name for name in SERVICES if re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", text
    )]


def unknown_services(text: str) -> list[str]:
    # 서비스명처럼 보이지만 목록에는 없는 이름 탐지
    pattern = r"(?<![A-Za-z0-9_-])(?:[A-Za-z][A-Za-z0-9-]*-service|web-\d+)(?![A-Za-z0-9_-])"
    return [name for name in set(re.findall(pattern, text)) if name not in SERVICES]


def choices() -> str:
    return "\n".join(f"{i}. {name}" for i, name in enumerate(SERVICES, 1))


def select_reply(reply: str) -> str | None:
    # 번호 또는 등록된 서비스명으로 답변할 수 있음
    match = re.fullmatch(r"(\d+)\s*번?", reply.strip())
    if match:
        index = int(match.group(1)) - 1
        return SERVICES[index] if 0 <= index < len(SERVICES) else None

    names = mentioned_services(reply)
    return names[0] if len(names) == 1 and not unknown_services(reply) else None


# 확인 대기 중인 요청에 후속 답변 결합
def prepare_node(state: AgentState):
    pending = state.get("pending_query", "")
    if not pending:
        return {"next_step": "route"}

    selected = select_reply(state["user_input"])
    if not selected:
        attempts = state.get("clarification_attempts", 0) + 1
        if attempts >= 3:
            return {
                "pending_query": "",
                "clarification_attempts": 0,
                "next_step": "end",
                "result": "서비스 확인에 3회 실패하여 요청을 취소했습니다.",
            }
        return {
            "clarification_attempts": attempts,
            "next_step": "end",
            "result": f"서비스를 확인할 수 없습니다. ({attempts}/3)\n{choices()}",
        }

    # 미등록 이름은 선택한 서비스로 교체
    query = pending
    missing = unknown_services(query)
    if missing:
        for name in missing:
            query = query.replace(name, selected)
    else:
        query += f"\n조사 대상 서비스: {selected}"

    return {
        "user_input": query,
        "intent": "diagnosis",
        "pending_query": "",
        "clarification_attempts": 0,
        "next_step": "validate",
    }


# 등록된 서비스만 Plan으로 전달
def target_check_node(state: AgentState):
    targets = mentioned_services(state["user_input"])
    missing = unknown_services(state["user_input"])

    if missing or not targets:
        detail = f"미등록 서비스: {', '.join(missing)}\n" if missing else "조사할 서비스가 불명확합니다.\n"
        return {
            "approved_targets": [],
            "pending_query": state["user_input"],
            "clarification_attempts": 0,
            "next_step": "end",
            "result": detail + "번호 또는 정확한 서비스명을 입력하세요.\n" + choices(),
        }

    return {"approved_targets": targets, "next_step": "plan"}


# LLM 계획을 검증된 서비스 목록으로 제한
def plan_node(state: AgentState):
    result = original_plan_node(state)
    allowed = state["approved_targets"]
    tasks = [t for t in result["tasks"]
             if t.get("agent") != "research" or t.get("target") in allowed]

    planned = {t["target"] for t in tasks if t.get("agent") == "research"}
    next_id = len(result["tasks"]) + 1
    for name in allowed:
        if name not in planned:
            tasks.append({
                "id": f"task-{next_id}",
                "agent": "research",
                "kind": "monitoring",
                "target": name,
                "status": "pending",
            })
            next_id += 1
    return {**result, "tasks": tasks}


# 기존 Agent 그래프에 확인 노드 두 개 추가
builder = StateGraph(AgentState)

builder.add_node("prepare", prepare_node)
builder.add_node("route", route_node)
builder.add_node("target_check", target_check_node)
builder.add_node("plan", plan_node)
builder.add_node("gate", gate_node)
builder.add_node("research", research_node)
builder.add_node("replan", replan_node)
builder.add_node("analysis", analysis_node)
builder.add_node("write", write_node)
builder.add_node("validate", validate_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "prepare")
builder.add_conditional_edges("prepare", lambda s: s["next_step"], {
    "route": "route", "validate": "target_check", "end": END,
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
    "research": "research", "analysis": "analysis", "end": END,
})
builder.add_edge("analysis", "write")
builder.add_edge("write", "validate")
builder.add_conditional_edges("validate", lambda s: s["validation_route"], {
    "write": "write", "end": END,
})
builder.add_edge("chat", END)

graph = builder.compile()


async def main():
    pending_query = ""
    attempts = 0

    while True:
        user_input = input("\n요청 (종료: /exit): ").strip()
        if user_input == "/exit":
            break

        result = await graph.ainvoke({
            "user_input": user_input,
            "pending_query": pending_query,
            "clarification_attempts": attempts,
        })
        print("\n" + result["result"])
        pending_query = result.get("pending_query", "")
        attempts = result.get("clarification_attempts", 0)


if __name__ == "__main__":
    asyncio.run(main())
