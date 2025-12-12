#!/usr/bin/env python3
"""
한국관광공사 KorService2 MCP Server + HTTP API
- MCP 클라이언트용: stdio 또는 SSE
- HTTP 클라이언트용: REST API (/api/tools/*)
"""
import os
import httpx
import uvicorn
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

# 환경 변수
KORSERVICE_URL = os.getenv("KORSERVICE_URL", "https://apis.data.go.kr/B551011/KorService2")
TOUR_API_KEY = os.getenv("TOUR_API_KEY")

# FastAPI 앱 생성
app = FastAPI(title="Tour API MCP Server")

# 공통 파라미터
COMMON_PARAMS = {
    "serviceKey": TOUR_API_KEY,
    "MobileOS": "ETC",
    "MobileApp": "TravelMCP",
    "_type": "json",
}


async def call_api(endpoint: str, params: dict) -> dict:
    """API 호출 헬퍼"""
    all_params = {**COMMON_PARAMS, **params}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{KORSERVICE_URL}/{endpoint}", params=all_params)
        data = response.json()

        if data["response"]["header"]["resultCode"] != "0000":
            return {"error": data["response"]["header"]["resultMsg"]}

        items = data["response"]["body"].get("items", {})
        if not items:
            return {"items": [], "totalCount": 0}

        item_list = items.get("item", [])
        if isinstance(item_list, dict):
            item_list = [item_list]

        return {
            "items": item_list,
            "totalCount": data["response"]["body"].get("totalCount", len(item_list))
        }


# ========== Request Models ==========
class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


# ========== HTTP API 엔드포인트 ==========
@app.get("/health")
async def health():
    return {"status": "ok", "service": "tour-mcp-server"}


@app.get("/api/tools")
async def list_tools():
    """사용 가능한 도구 목록"""
    return {
        "tools": [
            {"name": "get_area_codes", "description": "지역코드/시군구코드 조회"},
            {"name": "get_category_codes", "description": "서비스 분류코드 조회"},
            {"name": "search_by_area", "description": "지역기반 관광정보 검색"},
            {"name": "search_by_location", "description": "GPS 위치기반 검색"},
            {"name": "search_by_keyword", "description": "키워드 검색"},
            {"name": "search_festivals", "description": "축제/행사 검색"},
            {"name": "search_stays", "description": "숙박 검색"},
            {"name": "get_detail_common", "description": "상세정보 (공통)"},
            {"name": "get_detail_intro", "description": "상세정보 (소개)"},
            {"name": "get_detail_info", "description": "상세정보 (반복)"},
            {"name": "get_detail_images", "description": "이미지 목록"},
            {"name": "get_pet_tour_info", "description": "반려동물 여행정보"},
        ]
    }


@app.post("/api/call_tool")
async def call_tool(request: ToolCallRequest):
    """도구 호출 엔드포인트"""
    tool_name = request.name
    args = request.arguments

    # 도구별 함수 매핑
    tool_handlers = {
        "get_area_codes": get_area_codes,
        "get_category_codes": get_category_codes,
        "search_by_area": search_by_area,
        "search_by_location": search_by_location,
        "search_by_keyword": search_by_keyword,
        "search_festivals": search_festivals,
        "search_stays": search_stays,
        "get_detail_common": get_detail_common,
        "get_detail_intro": get_detail_intro,
        "get_detail_info": get_detail_info,
        "get_detail_images": get_detail_images,
        "get_pet_tour_info": get_pet_tour_info,
    }

    handler = tool_handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = await handler(**args)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== 도구 함수들 ==========
async def get_area_codes(area_code: Optional[str] = None) -> dict:
    """지역코드 조회"""
    params = {"numOfRows": 100}
    if area_code:
        params["areaCode"] = area_code
    return await call_api("areaCode2", params)


async def get_category_codes(
    content_type_id: Optional[str] = None,
    cat1: Optional[str] = None,
    cat2: Optional[str] = None
) -> dict:
    """서비스 분류코드 조회"""
    params = {"numOfRows": 100}
    if content_type_id:
        params["contentTypeId"] = content_type_id
    if cat1:
        params["cat1"] = cat1
    if cat2:
        params["cat2"] = cat2
    return await call_api("categoryCode2", params)


async def search_by_area(
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    content_type_id: Optional[str] = None,
    cat1: Optional[str] = None,
    cat2: Optional[str] = None,
    cat3: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """지역기반 관광정보 조회"""
    params = {"numOfRows": num_of_rows, "arrange": arrange}
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    if content_type_id:
        params["contentTypeId"] = content_type_id
    if cat1:
        params["cat1"] = cat1
    if cat2:
        params["cat2"] = cat2
    if cat3:
        params["cat3"] = cat3
    return await call_api("areaBasedList2", params)


async def search_by_location(
    map_x: float,
    map_y: float,
    radius: int = 5000,
    content_type_id: Optional[str] = None,
    arrange: str = "E",
    num_of_rows: int = 20
) -> dict:
    """위치기반 관광정보 조회"""
    params = {
        "mapX": map_x,
        "mapY": map_y,
        "radius": radius,
        "numOfRows": num_of_rows,
        "arrange": arrange
    }
    if content_type_id:
        params["contentTypeId"] = content_type_id
    return await call_api("locationBasedList2", params)


async def search_by_keyword(
    keyword: str,
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    content_type_id: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """키워드 검색 조회"""
    params = {
        "keyword": keyword,
        "numOfRows": num_of_rows,
        "arrange": arrange
    }
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    if content_type_id:
        params["contentTypeId"] = content_type_id
    return await call_api("searchKeyword2", params)


async def search_festivals(
    event_start_date: str,
    event_end_date: Optional[str] = None,
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """행사/축제 정보 조회"""
    params = {
        "eventStartDate": event_start_date,
        "numOfRows": num_of_rows,
        "arrange": arrange
    }
    if event_end_date:
        params["eventEndDate"] = event_end_date
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    return await call_api("searchFestival2", params)


async def search_stays(
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """숙박정보 조회"""
    params = {
        "numOfRows": num_of_rows,
        "arrange": arrange
    }
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    return await call_api("searchStay2", params)


async def get_detail_common(
    content_id: str,
    content_type_id: str,
    default_yn: str = "Y",
    first_image_yn: str = "Y",
    addr_info_yn: str = "Y",
    map_info_yn: str = "Y",
    overview_yn: str = "Y"
) -> dict:
    """상세조회 - 공통정보"""
    params = {
        "contentId": content_id,
        "contentTypeId": content_type_id,
        "defaultYN": default_yn,
        "firstImageYN": first_image_yn,
        "addrinfoYN": addr_info_yn,
        "mapinfoYN": map_info_yn,
        "overviewYN": overview_yn
    }
    return await call_api("detailCommon2", params)


async def get_detail_intro(content_id: str, content_type_id: str) -> dict:
    """상세조회 - 소개정보"""
    params = {
        "contentId": content_id,
        "contentTypeId": content_type_id
    }
    return await call_api("detailIntro2", params)


async def get_detail_info(content_id: str, content_type_id: str) -> dict:
    """상세조회 - 반복정보"""
    params = {
        "contentId": content_id,
        "contentTypeId": content_type_id
    }
    return await call_api("detailInfo2", params)


async def get_detail_images(content_id: str) -> dict:
    """상세조회 - 이미지 목록"""
    params = {
        "contentId": content_id,
        "imageYN": "Y",
        "subImageYN": "Y"
    }
    return await call_api("detailImage2", params)


async def get_pet_tour_info(content_id: str) -> dict:
    """반려동물 여행정보 조회"""
    params = {"contentId": content_id}
    return await call_api("detailPetTour2", params)


# ========== 실행 ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Tour MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
