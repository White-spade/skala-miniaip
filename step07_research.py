from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from step01_monitoring import get_monitoring_status
from step06_planning import route_node, plan_node, chat_node


# 공유 상태
class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    system_name: str
    tasks: list[dict]
    collected: list[dict]
    result: str


# 모니터링 조사 노드
def research_node(state: AgentState):
    tasks = [task.copy() for task in state["tasks"]]
    collected = []

    for task in tasks:
        # 실행 대기 중인 모니터링 작업만 선택
        if task["status"] != "pending" or task["kind"] != "monitoring":
            continue

        # 모니터링 조회
        data = get_monitoring_status(task["target"])

        if data is None:
            task["status"] = "failed"
            continue

        # 조회 결과 저장
        collected.append({
            "source": "monitoring",
            "target": task["target"],
            "data": data
        })

        task["status"] = "done"

    return {
        "tasks": tasks,
        "collected": collected,
        "result": "모니터링 조회 완료"
    }


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("research", research_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "route")

builder.add_conditional_edges(
    "route",
    lambda state: "plan" if state["intent"] == "diagnosis" else "chat",
    {
        "plan": "plan",
        "chat": "chat"
    }
)

builder.add_conditional_edges(
    "plan",
    lambda state: "research" if state["tasks"] else "end",
    {
        "research": "research",
        "end": END
    }
)

builder.add_edge("research", END)
builder.add_edge("chat", END)

graph = builder.compile()
#graph.get_graph().draw_mermaid_png(output_file_path="step07_graph.png")

# 실행
if __name__ == "__main__":
    user_input = input("요청: ")

    result = graph.invoke({"user_input": user_input})

    print("\n결과:", result["result"])

    for task in result.get("tasks", []):
        print(f"{task['id']}: {task['target']} [{task['status']}]")

    for item in result.get("collected", []):
        data = item["data"]

        print(f"\n서비스: {item['target']}")
        print(f"상태: {data['status']}")
        print(f"CPU: {data['cpu_base']}%")
        print(f"Memory: {data['memory_base']}%")

        for alert in data["alerts"]:
            print(f"- {alert}")