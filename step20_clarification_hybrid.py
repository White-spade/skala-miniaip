import asyncio
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from step05_llm_routing import llm
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


# 기존 State에 재질문 관련 값 추가
class AgentState(BaseAgentState, total=False):
    pending_query: str
    pending_candidate: str
    clarification_attempts: int
    llm_clarification_used: bool
    approved_targets: list[str]
    next_step: str


# 등록 서비스 목록
SERVICES = list(json.loads(
    Path(__file__).with_name("monitoring.json").read_text(encoding="utf-8")
))


def mentioned_services(text: str) -> list[str]:
    return [name for name in SERVICES if re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", text
    )]


def unknown_services(text: str) -> list[str]:
    # 오타 예시: payment-services, unknown-service, web-99
    pattern = r"(?<![A-Za-z0-9_-])(?:[A-Za-z][A-Za-z0-9-]*-services?|web-\d+)(?![A-Za-z0-9_-])"
    names = re.findall(pattern, text)
    return list(dict.fromkeys(name for name in names if name not in SERVICES))


def choices() -> str:
    return "\n".join(f"{i}. {name}" for i, name in enumerate(SERVICES, 1))


def select_reply(reply: str) -> str | None:
    # 번호나 등록 서비스명으로 선택
    match = re.fullmatch(r"(\d+)\s*번?", reply.strip())
    if match:
        index = int(match.group(1)) - 1
        return SERVICES[index] if 0 <= index < len(SERVICES) else None

    names = mentioned_services(reply)
    return names[0] if len(names) == 1 and not unknown_services(reply) else None


# 서비스 확인 중이면 후속 답변을 기존 요청에 결합

def prepare_node(state: AgentState):
    pending = state.get("pending_query", "")
    if not pending:
        return {"next_step": "route"}

    reply = state["user_input"].strip()
    candidate = state.get("pending_candidate", "")
    yes = {"네", "예", "응", "맞아", "맞아요", "그거", "그거야", "yes", "y"}
    no = {"아니", "아니요", "아뇨", "아니야", "no", "n"}

    # LLM 후보를 거절하면 고정 목록으로 다시 안내
    if candidate and reply.lower() in no:
        return {
            "pending_candidate": "",
            "next_step": "end",
            "result": "알겠습니다. 정확한 서비스명을 선택하세요.\n" + choices(),
        }

    selected = candidate if candidate and reply.lower() in yes else select_reply(reply)
    if not selected:
        attempts = state.get("clarification_attempts", 0) + 1
        if attempts >= 3:
            return {
                "pending_query": "",
                "pending_candidate": "",
                "clarification_attempts": 0,
                "llm_clarification_used": False,
                "next_step": "end",
                "result": "서비스 확인에 3회 실패하여 요청을 취소했습니다.",
            }
        return {
            "clarification_attempts": attempts,
            "next_step": "end",
            "result": f"서비스를 확인할 수 없습니다. ({attempts}/3)\n" + choices(),
        }

    # 미등록 이름 하나를 확정 서비스로 교체
    missing = unknown_services(pending)
    if missing:
        query = pending.replace(missing[0], selected, 1)
    else:
        query = pending + f"\n조사 대상 서비스: {selected}"

    return {
        "user_input": query,
        "intent": "diagnosis",
        "pending_query": "",
        "pending_candidate": "",
        "clarification_attempts": 0,
        "next_step": "target_check",
    }


# 규칙으로 판별 불가할 때만 LLM을 정확히 한 번 호출
async def clarification_fallback(query: str, missing: list[str]) -> tuple[str, str]:
    messages = [
        ("system",
         "당신은 IT 운영 서비스명 확인 질문 작성자입니다. "
         "등록된 서비스 목록 외의 서비스를 새로 만들어내지 마세요. "
         "입력에 오타처럼 보이는 서비스명이 있으면 가장 가능성 높은 등록 서비스 하나를 candidate로 제안하세요. "
         "서비스명을 짐작할 근거가 없다면 candidate는 빈 문자열로 반환하세요. "
         "질문은 한국어로 한 문장만 작성하고, 추측한 대상을 확정하지 마세요. "
         'JSON 객체 {"candidate": "등록 서비스명 또는 빈 문자열", "question": "사용자 확인 질문"}만 반환하세요.'),
        ("human", json.dumps({
            "query": query,
            "unknown_names": missing,
            "registered_services": SERVICES,
        }, ensure_ascii=False)),
    ]

    try:
        response = await llm.ainvoke(messages)
        data = json.loads(response.content)
        candidate = data.get("candidate", "")
        question = data.get("question", "")
    except (ValueError, TypeError, AttributeError):
        return "", "어떤 서비스를 조사할까요?"

    # LLM이 제안해도 등록 목록에 없거나 근거가 없으면 폐기
    if candidate not in SERVICES or len(missing) != 1:
        candidate = ""
        question = "어떤 서비스를 조사할까요?"

    if not isinstance(question, str) or not question.strip() or len(question) > 150:
        question = "어떤 서비스를 조사할까요?"
    else:
        question = question.strip()

    # 대상 확인 질문만 허용하고 실제 조사에는 사용하지 않음
    if candidate and candidate not in question:
        question = f"'{candidate}'를 말씀하셨나요?"

    return candidate, question


# 등록된 서비스만 Plan으로 통과
async def target_check_node(state: AgentState):
    query = state["user_input"]
    targets = mentioned_services(query)
    missing = unknown_services(query)

    if targets and not missing:
        return {
            "approved_targets": targets,
            "next_step": "plan",
        }

    candidate = ""
    question = "어떤 서비스를 조사할까요?"
    used = state.get("llm_clarification_used", False)

    if not used:
        print("[Clarification LLM 호출: 1회]")
        candidate, question = await clarification_fallback(query, missing)
        used = True

    prefix = f"미등록 서비스: {', '.join(missing)}\n" if missing else "조사 대상이 불명확합니다.\n"
    hint = " (네/아니요 또는 서비스명으로 답변)" if candidate else ""

    return {
        "approved_targets": [],
        "pending_query": query,
        "pending_candidate": candidate,
        "llm_clarification_used": used,
        "next_step": "end",
        "result": prefix + question + hint + "\n" + choices(),
    }


# LLM이 생성한 Research Task도 최종 검증

def plan_node(state: AgentState):
    result = original_plan_node(state)
    allowed = state["approved_targets"]

    tasks = [
        task for task in result["tasks"]
        if task.get("agent") != "research" or task.get("target") in allowed
    ]

    planned = {task["target"] for task in tasks if task.get("agent") == "research"}
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


# 기존 그래프에 Prepare / Target Check 추가
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
    session = {
        "pending_query": "",
        "pending_candidate": "",
        "clarification_attempts": 0,
        "llm_clarification_used": False,
    }

    while True:
        user_input = input("\n요청 (종료: /exit): ").strip()
        if user_input == "/exit":
            break

        result = await graph.ainvoke({**session, "user_input": user_input})
        print("\n" + result["result"])

        # 대기 중인 재질문 맥락만 다음 턴까지 유지
        for key in session:
            session[key] = result.get(key, session[key])

        # 하나의 요청이 끝나면 다음 요청에서는 LLM을 다시 사용할 수 있음
        if not session["pending_query"]:
            session = dict.fromkeys(session, "")
            session["clarification_attempts"] = 0
            session["llm_clarification_used"] = False


if __name__ == "__main__":
    asyncio.run(main())
