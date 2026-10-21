from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from step02_diagnosis import diagnose_service


# 공유 상태
class AgentState(TypedDict):
    user_input: str
    intent: str
    result: str


# 요청 분류 노드
def route_node(state: AgentState):
    if state["user_input"].startswith("진단 "):
        return {"intent": "diagnosis"}

    return {"intent": "chat"}


# 실행 경로 선택
def select_route(state: AgentState):
    return state["intent"]


# 장애 진단 노드
def diagnosis_node(state: AgentState):
    system_name = state["user_input"].removeprefix("진단 ").strip()
    result = diagnose_service(system_name)
    return {"result": result}


# 일반 대화 노드
def chat_node(state: AgentState):
    return {"result": "일반 대화는 아직 지원하지 않습니다."}


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("diagnosis", diagnosis_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "route")

builder.add_conditional_edges(
    "route",
    select_route,
    {
        "diagnosis": "diagnosis",
        "chat": "chat"
    }
)

builder.add_edge("diagnosis", END)
builder.add_edge("chat", END)

graph = builder.compile()

# 그래프 이미지 저장
graph.get_graph().draw_mermaid_png(output_file_path="step04_graph.png")


if __name__ == "__main__":
    user_input = input("요청: ")
    result = graph.invoke({"user_input": user_input})
    print(result["result"])