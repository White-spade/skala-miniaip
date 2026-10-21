import json

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from step06_planning import route_node, plan_node, chat_node
from step10_mcp_research import AgentState, research_node


# 보고서 작성용 LLM
writer_llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    reasoning=False
)#route나 plan과 다르게 format = "json" 매개변수가 없다. 자연어처리를 할것이기 때문


# 보고서 작성 노드
async def write_node(state: AgentState):
    # 수집 데이터 확인
    collected = state.get("collected", [])

    if not collected:
        return {"result": "모니터링 데이터를 수집하지 못했습니다."}

    # LLM 입력 구성
    messages = [
        ("system",
         "당신은 AIOps 장애 분석가입니다. "
         "제공된 모니터링 데이터만 근거로 진단 보고서를 작성하세요. "
         "상태, CPU, Memory, Alert를 분석하세요. "
         "추측을 사실처럼 단정하지 마세요. "
         "확인되지 않은 장애 원인은 가능성으로만 표현하세요. "
         "한국어로 작성하세요.\n"
         "보고서 구성:\n"
         "1. 서비스 상태\n"
         "2. 주요 이상 징후\n"
         "3. 예상 원인\n"
         "4. 권장 조치"),
        ("human",
         f"사용자 요청: {state['user_input']}\n\n"
         f"조사 작업:\n{json.dumps(state['tasks'], ensure_ascii=False)}\n\n"
         f"수집 데이터:\n{json.dumps(collected, ensure_ascii=False)}")
    ]

    # LLM 보고서 생성
    response = await writer_llm.ainvoke(messages)

    return {"result": response.content}


# 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("research", research_node)
builder.add_node("write", write_node)
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

builder.add_edge("research", "write")
builder.add_edge("write", END)
builder.add_edge("chat", END)

graph = builder.compile()


# 실행
if __name__ == "__main__":
    import asyncio

    async def main():
        user_input = input("요청: ")

        result = await graph.ainvoke({"user_input": user_input})

        print("\n=== 최종 보고서 ===")
        print(result["result"])

    asyncio.run(main())