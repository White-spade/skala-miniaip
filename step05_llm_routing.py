import json
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from step02_diagnosis import diagnose_service


# LLM 생성
llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    format="json",
    reasoning=False
)


# 공유 상태
class AgentState(TypedDict):
    user_input: str
    intent: str
    system_name: str
    result: str


# LLM 요청 분류 노드
def route_node(state: AgentState):
    messages = [
        ("system",
        "당신은 IT 서비스 운영 요청 분류기입니다.\n"
        "사용자 요청을 diagnosis 또는 chat으로 분류하세요.\n"
        "diagnosis: 장애 의심, 성능 저하, 응답 지연, 모니터링, 상태 확인\n"
        "chat: 인사, 잡담, IT 운영과 무관한 질문\n"
        "서비스 이름이 없어도 장애나 성능 문제를 언급하면 diagnosis입니다.\n"
        "예시:\n"
        "'서버가 느린 것 같아' → diagnosis\n"
        "'CPU가 너무 높아' → diagnosis\n"
        "'payments-service 확인해줘' → diagnosis\n"
        "'안녕하세요' → chat\n"
        "사용자가 명시한 서비스 이름만 추출하세요.\n"
        "서비스 이름이 없으면 빈 문자열을 사용하세요.\n"
        '반드시 {"intent": "...", "system_name": "..."} 형식의 JSON만 반환하세요.'),
        ("human", state["user_input"])
    ]

    response = llm.invoke(messages)
    data = json.loads(response.content)

    # LLM 분류 결과 검증
    if data.get("intent") not in ("diagnosis", "chat"):
        raise ValueError("LLM이 올바르지 않은 분류값을 반환했습니다.")

    system_name = data.get("system_name", "")

    if not isinstance(system_name, str) or system_name not in state["user_input"]:
        system_name = ""

    return {
        "intent": data["intent"],
        "system_name": system_name
    }


# 장애 진단 노드
def diagnosis_node(state: AgentState):
    if not state["system_name"]:
        return {"result": "진단할 서비스 이름을 알려주세요."}

    result = diagnose_service(state["system_name"])
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
    lambda state: state["intent"],
    {
        "diagnosis": "diagnosis",
        "chat": "chat"
    }
)

builder.add_edge("diagnosis", END)
builder.add_edge("chat", END)

graph = builder.compile()


# 실행
if __name__ == "__main__":
    user_input = input("요청: ")

    result = graph.invoke({"user_input": user_input})

    print(f"분류: {result['intent']}")
    print(f"대상: {result['system_name']}")
    print(result["result"])