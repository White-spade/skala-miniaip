import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# MCP 클라이언트 실행
async def main():
    server_path = Path(__file__).with_name("step09_mcp_server.py")

    # 서버 실행 설정
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    # MCP 서버 연결
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 등록된 Tool 확인
            tools = await session.list_tools()
            print("사용 가능한 Tool:", [tool.name for tool in tools.tools])

            # Tool 호출
            system_name = input("조회할 서비스 이름: ")

            response = await session.call_tool(
                "get_monitoring_status",
                arguments={"system_name": system_name}
            )

            if response.is_error:
                print("Tool 실행 오류:", response.content)
                return

            # 응답 출력
            data = json.loads(response.content[0].text)
            print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())