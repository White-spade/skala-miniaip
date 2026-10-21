import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from step05_llm_routing import llm, route_node


# 공유 상태
class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    system_name: str
    tasks: list[dict]
    result: str


# 조사 계획 생성 노드
def plan_node(state: AgentState):
    # 등록된 서비스 목록
    path = Path(__file__).with_name("monitoring.json")

    with path.open(encoding="utf-8") as f:
        services = list(json.load(f))

    # LLM에 계획 생성 요청
    messages = [
        ("system",
         "당신은 AIOps 장애 조사 계획 수립자입니다. "
         "사용자가 명시적으로 언급한 등록 서비스들을 모두 조사 대상으로 선정하세요. "
         f"등록 서비스: {', '.join(services)}. "
         "사용자가 언급하지 않은 서비스를 추가하지 마세요. "
         "서비스가 명시되지 않았다면 빈 배열을 반환하세요. "
         '반드시 {"targets": ["서비스명"]} 형식의 JSON만 반환하세요.'),
        ("human", state["user_input"])
    ]

    response = llm.invoke(messages)
    data = json.loads(response.content)

    targets = data.get("targets", [])

    if not isinstance(targets, list):
        raise ValueError("조사 대상이 목록 형태가 아닙니다.")

    # LLM이 선택한 대상을 작업으로 변환
    tasks = []

    for target in targets:
        if target not in services or target not in state["user_input"]:
            continue

        if any(task["target"] == target for task in tasks):
            continue

        tasks.append({
            "id": f"task-{len(tasks) + 1}",
            "agent": "research",
            "kind": "monitoring",
            "target": target,
            "status": "pending"
        })

    if not tasks:
        return {
            "tasks": [],
            "result": "조사할 서비스 이름을 알려주세요."
        }

    return {
        "tasks": tasks,
        "result": "조사 계획을 생성했습니다."
    }


# 일반 대화 노드
def chat_node(state: AgentState):
    return {"result": "일반 대화는 아직 지원하지 않습니다."}


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
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

builder.add_edge("plan", END)
builder.add_edge("chat", END)

graph = builder.compile()


# 실행
if __name__ == "__main__":
    user_input = input("요청: ")

    result = graph.invoke({"user_input": user_input})

    print("분류:", result["intent"])
    print(result["result"])

    for task in result.get("tasks", []):
        print(task)