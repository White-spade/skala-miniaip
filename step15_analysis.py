import asyncio
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from step05_llm_routing import llm
from step06_planning import route_node, plan_node as basic_plan_node, chat_node
from step12_runbook import research_node as basic_research_node
from step14_gate import gate_node
from step11_write import writer_llm


# 공유 상태
class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    system_name: str
    needs_analysis: bool
    tasks: list[dict]
    collected: list[dict]
    analysis_result: dict
    gate_removed: int
    result: str


# 조사 계획 및 추가 분석 여부 결정
def plan_node(state: AgentState):
    plan = basic_plan_node(state)
    tasks = [task.copy() for task in plan["tasks"]]

    if not tasks:
        return {**plan, "needs_analysis": False}

    messages = [
        ("system",
         "당신은 AIOps 조사 계획 수립자입니다. "
         "사용자 요청에 추가적인 원인 분석이나 서비스 간 연관성 분석이 필요한지 판단하세요. "
         "단순 상태 확인이면 false, 원인 분석이나 서비스 간 관계 분석이 필요하면 true입니다. "
         '반드시 {"needs_analysis": true} 또는 {"needs_analysis": false} JSON만 반환하세요.'),
        ("human", state["user_input"])
    ]

    response = llm.invoke(messages)
    decision = json.loads(response.content)

    needs_analysis = decision.get("needs_analysis") is True

    # 분석 작업 추가
    if needs_analysis:
        tasks.append({
            "id": f"task-{len(tasks) + 1}",
            "agent": "analysis",
            "kind": "correlation",
            "target": "collected-data",
            "status": "pending"
        })

    return {
        "tasks": tasks,
        "needs_analysis": needs_analysis,
        "result": plan["result"]
    }


# Research Task만 MCP로 전달
async def research_node(state: AgentState):
    research_tasks = [
        task for task in state["tasks"]
        if task["agent"] == "research"
    ]

    research_state = {**state, "tasks": research_tasks}
    response = await basic_research_node(research_state)

    analysis_tasks = [
        task.copy() for task in state["tasks"]
        if task["agent"] == "analysis"
    ]

    return {
        "tasks": response["tasks"] + analysis_tasks,
        "collected": response["collected"]
    }


# 장애 원인 분석 노드
async def analysis_node(state: AgentState):
    pending = [
        task for task in state["tasks"]
        if task["agent"] == "analysis" and task["status"] == "pending"
    ]

    if not pending:
        return {}

    # 모니터링 결과만 추출
    monitoring = [
        item for item in state.get("collected", [])
        if item["source"] == "monitoring"
    ]

    if monitoring:
        messages = [
            ("system",
             "당신은 AIOps 장애 분석가입니다. "
             "모니터링 결과에서 관측 사실, 원인 가설, 추가 확인 사항을 구분하세요. "
             "서비스 간 연관성을 분석하되 인과관계를 사실처럼 단정하지 마세요. "
             '반드시 {"observations": [], "hypotheses": [], "checks": []} '
             "형식의 JSON으로 반환하고, 각 배열에는 한국어 문장을 넣으세요."),
            ("human", json.dumps(monitoring, ensure_ascii=False))
        ]

        response = await llm.ainvoke(messages)
        analysis = json.loads(response.content)
        status = "done"

    else:
        analysis = {
            "observations": [],
            "hypotheses": [],
            "checks": ["모니터링 데이터를 확보하지 못했습니다."]
        }
        status = "failed"

    # Analysis Task 상태 갱신
    tasks = [task.copy() for task in state["tasks"]]

    for task in tasks:
        if task["agent"] == "analysis" and task["status"] == "pending":
            task["status"] = status

    return {
        "tasks": tasks,
        "analysis_result": analysis
    }


# 분석 결과를 이용한 보고서 작성
async def write_node(state: AgentState):
    messages = [
        ("system",
         "당신은 AIOps 보고서 작성자입니다. "
         "모니터링 데이터와 Runbook, 사전 분석 결과를 종합해 한국어 보고서를 작성하세요. "
         "관측 사실과 예상 원인을 구분하고, 확인되지 않은 원인을 단정하지 마세요. "
         "Runbook의 권장 조치를 이미 수행했다고 표현하지 마세요.\n"
         "보고서 구성:\n"
         "1. 서비스 상태\n"
         "2. 주요 이상 징후\n"
         "3. 예상 원인 및 연관성\n"
         "4. 권장 조치"),
        ("human",
         f"사용자 요청: {state['user_input']}\n\n"
         f"수집 자료:\n{json.dumps(state.get('collected', []), ensure_ascii=False)}\n\n"
         f"분석 결과:\n{json.dumps(state.get('analysis_result', {}), ensure_ascii=False)}")
    ]

    response = await writer_llm.ainvoke(messages)

    return {"result": response.content}


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("gate", gate_node)
builder.add_node("research", research_node)
builder.add_node("analysis", analysis_node)
builder.add_node("write", write_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "route")

builder.add_conditional_edges(
    "route",
    lambda state: "plan" if state["intent"] == "diagnosis" else "chat",
    {"plan": "plan", "chat": "chat"}
)

builder.add_edge("plan", "gate")

builder.add_conditional_edges(
    "gate",
    lambda state: "research" if state["tasks"] else "end",
    {"research": "research", "end": END}
)

builder.add_edge("research", "analysis")
builder.add_edge("analysis", "write")
builder.add_edge("write", END)
builder.add_edge("chat", END)

graph = builder.compile()


# 실행
async def main():
    user_input = input("요청: ")

    result = await graph.ainvoke({"user_input": user_input})

    print("\n추가 분석 필요:", result.get("needs_analysis", False))
    print("Gate 제거 작업:", result.get("gate_removed", 0))

    for task in result.get("tasks", []):
        print(f"{task['agent']}: {task['target']} [{task['status']}]")

    if result.get("analysis_result"):
        print("\n=== Analysis 결과 ===")
        print(json.dumps(result["analysis_result"], ensure_ascii=False, indent=2))

    print("\n=== 최종 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())