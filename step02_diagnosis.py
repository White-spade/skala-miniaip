from step01_monitoring import get_monitoring_status


def diagnose_service(system_name):
    # 모니터링 데이터 조회
    data = get_monitoring_status(system_name)

    if data is None:
        return "등록되지 않은 서비스입니다."

    # 정상 상태 확인
    if data["status"] == "healthy":
        return f"{system_name}: 정상입니다."

    # 이상 상태 및 알림 정리
    result = f"{system_name}: 이상 상태({data['status']})입니다."

    for alert in data["alerts"]:
        result += f"\n- {alert}"

    return result


if __name__ == "__main__":
    system_name = input("진단할 서비스 이름: ")
    diagnosis = diagnose_service(system_name)
    print(diagnosis)