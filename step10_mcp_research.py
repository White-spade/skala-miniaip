import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END

from step06_planning import route_node, plan_node, chat_node


# 공유 상태
class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    system_name: str
    tasks: list[dict]
    collected: list[dict]
    result: str


# MCP 기반 모니터링 조사 노드
async def research_node(state: AgentState):
    tasks = [task.copy() for task in state["tasks"]]
    collected = []

    # MCP 서버 설정
    server_path = Path(__file__).with_name("step09_mcp_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    # MCP 서버 연결
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for task in tasks:
                if task["status"] != "pending" or task["kind"] != "monitoring":
                    continue

                # MCP Tool 호출
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

                # 조사 결과 저장
                collected.append({
                    "source": "monitoring",
                    "target": task["target"],
                    "data": data
                })

                task["status"] = "done"

    return {
        "tasks": tasks,
        "collected": collected,
        "result": "MCP 모니터링 조회 완료"
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
graph.get_graph().draw_mermaid_png(output_file_path="step10_graph.png")


# 실행
async def main():
    user_input = input("요청: ")

    result = await graph.ainvoke({"user_input": user_input})

    print("\n결과:", result["result"])

    for task in result.get("tasks", []):
        print(f"{task['id']}: {task['target']} [{task['status']}]")

    for item in result.get("collected", []):
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())