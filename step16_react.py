import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from step11_write import writer_llm
from step15_analysis import (
    AgentState, route_node, plan_node, gate_node,
    analysis_node, write_node, chat_node
)


# ReAct 기반 조사 노드
async def research_node(state: AgentState):
    tasks = [task.copy() for task in state["tasks"]]

    # 등록된 서비스 확인
    path = Path(__file__).with_name("monitoring.json")
    services = json.loads(path.read_text(encoding="utf-8"))

    # MCP 서버 설정
    server_path = Path(__file__).with_name("step12_mcp_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # MCP Tool 호출 함수
            @tool
            async def get_monitoring_status(system_name: str) -> str:
                """등록된 IT 서비스의 CPU, 메모리, 상태, 알림을 조회합니다."""

                if system_name not in services:
                    return json.dumps({"error": "등록되지 않은 서비스"})

                response = await session.call_tool(
                    "get_monitoring_status",
                    arguments={"system_name": system_name}
                )

                if response.is_error:
                    return json.dumps({"error": "모니터링 조회 실패"})

                data = json.loads(response.content[0].text)

                return json.dumps({
                    "source": "monitoring",
                    "target": system_name,
                    "data": data
                }, ensure_ascii=False)

            # MCP Resource 조회 함수
            @tool
            async def get_runbook(system_name: str, incident_type: str) -> str:
                """장애 대응 문서를 조회합니다. 유형은 cpu-spike 또는 network-timeout입니다."""

                if system_name not in services:
                    return json.dumps({"error": "등록되지 않은 서비스"})

                if incident_type not in ("cpu-spike", "network-timeout"):
                    return json.dumps({"error": "지원하지 않는 장애 유형"})

                uri = f"runbook://incident-response/{incident_type}"
                response = await session.read_resource(uri)

                return json.dumps({
                    "source": "runbook",
                    "target": system_name,
                    "incident_type": incident_type,
                    "text": response.contents[0].text
                }, ensure_ascii=False)

            # LLM에 Tool 등록
            tools = [get_monitoring_status, get_runbook]
            llm_with_tools = writer_llm.bind_tools(tools)

            # ReAct 판단 노드
            async def react_node(react_state: MessagesState):
                messages = [
                    SystemMessage(content=
                        "당신은 AIOps 조사 Agent입니다. "
                        "등록 서비스: " + ", ".join(services) + ". "
                        "조사 대상의 모니터링 상태를 반드시 Tool로 확인하세요. "
                        "결과를 관찰하고 관련 Runbook이 필요하면 조회하세요. "
                        "추가로 확인할 등록 서비스가 있다면 근거에 따라 조회하세요. "
                        "필요한 조사가 끝나면 Tool 호출을 중단하세요. "
                        "존재하지 않는 서비스나 측정값을 만들어내지 마세요."
                    )
                ] + react_state["messages"]

                response = await llm_with_tools.ainvoke(messages)

                for call in response.tool_calls:
                    print(f"ReAct 선택: {call['name']} {call['args']}")

                return {"messages": [response]}

            # ReAct 그래프 구성
            react_builder = StateGraph(MessagesState)

            react_builder.add_node("react", react_node)
            react_builder.add_node("tools", ToolNode(tools))

            react_builder.add_edge(START, "react")
            react_builder.add_conditional_edges(
                "react",
                tools_condition,
                {"tools": "tools", END: END}
            )
            react_builder.add_edge("tools", "react")

            react_graph = react_builder.compile()

            # ReAct 실행
            targets = [
                task["target"] for task in tasks
                if task["agent"] == "research"
            ]

            result = await react_graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=
                            f"사용자 요청: {state['user_input']}\n"
                            f"계획된 조사 대상: {targets}"
                        )
                    ]
                },
                config={"recursion_limit": 16}
            )

    # Tool 실행 결과 수집
    collected = []

    for message in result["messages"]:
        if not isinstance(message, ToolMessage):
            continue

        try:
            item = json.loads(message.content)
        except (TypeError, ValueError):
            continue

        if item.get("source") in ("monitoring", "runbook"):
            collected.append(item)

    # Research Task 상태 갱신
    for task in tasks:
        if task["agent"] != "research":
            continue

        success = any(
            item["source"] == "monitoring"
            and item["target"] == task["target"]
            for item in collected
        )

        task["status"] = "done" if success else "failed"

    return {
        "tasks": tasks,
        "collected": collected,
        "result": (
            "ReAct 조사 완료" if any(
                item["source"] == "monitoring" for item in collected
            ) else "모니터링 데이터를 수집하지 못했습니다."
        )
    }


# 전체 그래프 구성
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("plan", plan_node)
builder.add_node("gate", gate_node)
builder.add_node("research", research_node)
builder.add_node("analysis", analysis_node)
builder.add_node("write", write_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "route")

builder.add_conditional_edges(
    "route",
    lambda s: "plan" if s["intent"] == "diagnosis" else "chat",
    {"plan": "plan", "chat": "chat"}
)

builder.add_edge("plan", "gate")

builder.add_conditional_edges(
    "gate",
    lambda s: "research" if s["tasks"] else "end",
    {"research": "research", "end": END}
)

builder.add_conditional_edges(
    "research",
    lambda s: "analysis" if any(
        item["source"] == "monitoring"
        for item in s.get("collected", [])
    ) else "end",
    {"analysis": "analysis", "end": END}
)

builder.add_edge("analysis", "write")
builder.add_edge("write", END)
builder.add_edge("chat", END)

graph = builder.compile()
#graph.get_graph().draw_mermaid_png(output_file_path="step16_graph.png")

# 실행
async def main():
    user_input = input("요청: ")

    result = await graph.ainvoke({"user_input": user_input})

    print("\n=== Task 상태 ===")
    for task in result.get("tasks", []):
        print(f"{task['agent']}: {task['target']} [{task['status']}]")

    print("\n=== 수집 결과 ===")
    for item in result.get("collected", []):
        print(f"{item['source']}: {item['target']}")

    print("\n=== 최종 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())