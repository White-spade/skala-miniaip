from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from step02_diagnosis import diagnose_service


# 공유 상태 정의
class AgentState(TypedDict):
    system_name: str
    diagnosis: str


# 진단 노드
def diagnosis_node(state: AgentState):
    result = diagnose_service(state["system_name"])
    return {"diagnosis": result}


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("diagnosis", diagnosis_node)

builder.add_edge(START, "diagnosis")
builder.add_edge("diagnosis", END)

graph = builder.compile()


# 실행
if __name__ == "__main__":
    system_name = input("진단할 서비스 이름: ")

    result = graph.invoke({"system_name": system_name})

    print(result["diagnosis"])