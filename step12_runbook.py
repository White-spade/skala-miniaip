import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END

from step06_planning import route_node, plan_node, chat_node
from step10_mcp_research import AgentState
from step11_write import writer_llm


# 모니터링 결과로 Runbook 선택
def select_runbook(data):
    alerts = " ".join(data["alerts"]).lower()

    if "cpu " in alerts:
        return "cpu-spike"

    if "timeout" in alerts or "packet loss" in alerts:
        return "network-timeout"

    return None


# MCP 조사 노드
async def research_node(state: AgentState):
    tasks = [task.copy() for task in state["tasks"]]
    collected = []

    server_path = Path(__file__).with_name("step12_mcp_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for task in tasks:
                if task["status"] != "pending":
                    continue

                # Tool로 모니터링 조회
                response = await session.call_tool(
                    "get_monitoring_status",
                    arguments={"system_name": task["target"]}
                )

                if response.is_error:
                    task["status"] = "failed"
                    continue

                data = json.loads(response.content[0].text)

                if data["status"] == "unknown":
                    task["status"] = "failed"
                    continue

                collected.append({
                    "source": "monitoring",
                    "target": task["target"],
                    "data": data
                })

                task["status"] = "done"

                # 모니터링 결과로 장애 유형 선택
                incident_type = select_runbook(data)

                if incident_type is None:
                    continue

                # Resource로 Runbook 조회
                uri = f"runbook://incident-response/{incident_type}"
                resource = await session.read_resource(uri)

                collected.append({
                    "source": "runbook",
                    "target": task["target"],
                    "incident_type": incident_type,
                    "text": resource.contents[0].text
                })

    return {
        "tasks": tasks,
        "collected": collected
    }


# 보고서 작성 노드
async def write_node(state: AgentState):
    messages = [
        ("system",
         "당신은 AIOps 장애 분석가입니다. "
         "모니터링 데이터와 Runbook을 참고해 한국어 진단 보고서를 작성하세요. "
         "모니터링 데이터는 관측 사실이고 Runbook은 대응 지침입니다. "
         "확인되지 않은 원인을 단정하지 마세요. "
         "Runbook의 절차를 이미 수행했다고 표현하지 마세요.\n"
         "보고서 구성:\n"
         "1. 서비스 상태\n"
         "2. 주요 이상 징후\n"
         "3. 예상 원인\n"
         "4. Runbook 기반 권장 조치"),
        ("human",
         f"사용자 요청: {state['user_input']}\n\n"
         f"수집 자료:\n{json.dumps(state['collected'], ensure_ascii=False)}")
    ]

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
    {"plan": "plan", "chat": "chat"}
)

builder.add_conditional_edges(
    "plan",
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

    for item in result.get("collected", []):
        print(f"수집: {item['source']} / {item['target']}")

    print("\n=== 진단 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())