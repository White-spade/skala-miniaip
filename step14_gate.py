import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from step06_planning import route_node, plan_node, chat_node
from step12_runbook import research_node, write_node


# 공유 상태
class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    system_name: str
    needs_analysis: bool
    tasks: list[dict]
    collected: list[dict]
    gate_removed: int
    result: str


# 작업 검증 노드
def gate_node(state: AgentState):
    tasks = state["tasks"]

    # 추가 분석이 불필요하면 Analysis Task 제거
    if not state.get("needs_analysis", False):
        kept = [t for t in tasks if t["agent"] != "analysis"]
    else:
        kept = tasks

    return {
        "tasks": kept,
        "gate_removed": len(tasks) - len(kept)
    }


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("gate", gate_node)
builder.add_node("research", research_node)
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

builder.add_edge("research", "write")
builder.add_edge("write", END)
builder.add_edge("chat", END)

graph = builder.compile()


# 실행
async def main():
    user_input = input("요청: ")

    result = await graph.ainvoke({"user_input": user_input})

    print(f"\nGate 제거 작업: {result.get('gate_removed', 0)}개")

    for task in result.get("tasks", []):
        print(task)

    print("\n=== 진단 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())