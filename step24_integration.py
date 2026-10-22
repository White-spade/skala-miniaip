"""STEP 24: STEP 22 Grounding + optional STEP 18 LLM Validation."""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END  # noqa: E402
from langsmith import traceable  # noqa: E402
from step22_grounding import AgentState as BaseAgentState, graph as grounding_graph  # noqa: E402
from step18_replan_validate import validate_node as llm_validate_node  # noqa: E402


class AgentState(BaseAgentState, total=False):
    semantic_validation_ran: bool
    semantic_validation_passed: bool
    semantic_validation_feedback: str


# Grounding 성공 시에만 선택적으로 추가 LLM 평가
# Grounding Fallback이나 일반 대화에는 LLM 평가를 실행하지 않음
def semantic_route(state: AgentState) -> str:
    enabled = os.getenv("ENABLE_SEMANTIC_VALIDATION", "false").lower() in ("true", "1", "yes")
    return "semantic" if enabled and state.get("validation_passed") is True else "end"


# STEP 18의 기존 검증 함수를 1회만 실행하고 STEP 22 보고서는 유지
async def semantic_validation_node(state: AgentState):
    try:
        reviewed = await llm_validate_node({
            **state,
            "validation_count": 0,       # STEP 18의 독립적인 첫 검증으로 실행
            "validation_feedback": "",
        })
        passed = reviewed.get("validation_passed") is True
        feedback = reviewed.get("validation_feedback", "")
    except Exception:
        logging.exception("LLM Semantic Validation 실패")
        passed = False
        feedback = "LLM 검증기를 실행하지 못했습니다."

    status = "PASS (참고 평가)" if passed else "검토 필요 (Grounding 보고서는 유지)"
    return {
        "semantic_validation_ran": True,
        "semantic_validation_passed": passed,
        "semantic_validation_feedback": feedback,
        "result": state["result"] + f"\n\n※ LLM 의미 검증: {status}",
    }


# 이미 완성된 STEP 22 그래프를 서브그래프로 재사용
builder = StateGraph(AgentState)
builder.add_node("grounding_agent", grounding_graph)
builder.add_node("semantic_validation", semantic_validation_node)
builder.add_edge(START, "grounding_agent")
builder.add_conditional_edges("grounding_agent", semantic_route, {
    "semantic": "semantic_validation", "end": END,
})
builder.add_edge("semantic_validation", END)

graph = builder.compile()


@traceable(name="aiops_integrated_request")
async def run_agent(query: str):
    return await graph.ainvoke({"user_input": query})


async def main():
    print("\n=== LangGraph 구조 (Mermaid) ===")
    print(graph.get_graph(xray=True).draw_mermaid())

    query = input("요청: ").strip()
    result = await run_agent(query)
    print("\n" + result.get("result", "응답 없음"))
    print("Grounding 통과:", result.get("validation_passed", "해당 없음"))
    print("LLM 의미 검증:", result.get("semantic_validation_passed", "실행 안 함"))


if __name__ == "__main__":
    asyncio.run(main())
