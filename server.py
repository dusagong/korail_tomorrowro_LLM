#!/usr/bin/env python3
"""
한국관광공사 KorService2 MCP Server
"""
import os
import httpx
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# 환경 변수
KORSERVICE_URL = os.getenv("KORSERVICE_URL", "https://apis.data.go.kr/B551011/KorService2")
TOUR_API_KEY = os.getenv("TOUR_API_KEY")

# MCP 서버 생성
mcp = FastMCP("KorService2 관광정보")

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


# ========== 1. 지역코드 조회 ==========
@mcp.tool()
async def get_area_codes(area_code: Optional[str] = None) -> dict:
    """
    지역코드 조회 (areaCode2)

    Args:
        area_code: 지역코드 (없으면 시도 목록, 있으면 해당 지역의 시군구 목록)

    Returns:
        지역코드 목록 (code, name, rnum)
    """
    params = {"numOfRows": 100}
    if area_code:
        params["areaCode"] = area_code
    return await call_api("areaCode2", params)


# ========== 2. 서비스 분류코드 조회 ==========
@mcp.tool()
async def get_category_codes(
    content_type_id: Optional[str] = None,
    cat1: Optional[str] = None,
    cat2: Optional[str] = None
) -> dict:
    """
    서비스 분류코드 조회 (categoryCode2)

    Args:
        content_type_id: 관광타입 ID (12:관광지, 14:문화시설, 15:축제, 25:여행코스, 28:레포츠, 32:숙박, 38:쇼핑, 39:음식점)
        cat1: 대분류 코드
        cat2: 중분류 코드

    Returns:
        분류코드 목록 (code, name)
    """
    params = {"numOfRows": 100}
    if content_type_id:
        params["contentTypeId"] = content_type_id
    if cat1:
        params["cat1"] = cat1
    if cat2:
        params["cat2"] = cat2
    return await call_api("categoryCode2", params)


# ========== 3. 지역기반 관광정보 조회 ==========
@mcp.tool()
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
    """
    지역기반 관광정보 조회 (areaBasedList2)

    Args:
        area_code: 지역코드 (1:서울, 6:부산, 32:강원, 39:제주 등)
        sigungu_code: 시군구코드
        content_type_id: 관광타입 (12:관광지, 14:문화시설, 15:축제, 25:여행코스, 28:레포츠, 32:숙박, 38:쇼핑, 39:음식점)
        cat1: 대분류
        cat2: 중분류
        cat3: 소분류
        arrange: 정렬 (A:제목순, C:수정일순, D:등록일순)
        num_of_rows: 결과 개수

    Returns:
        관광지 목록 (contentid, title, addr1, firstimage, mapx, mapy 등)
    """
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


# ========== 4. 위치기반 관광정보 조회 ==========
@mcp.tool()
async def search_by_location(
    map_x: float,
    map_y: float,
    radius: int = 5000,
    content_type_id: Optional[str] = None,
    arrange: str = "E",
    num_of_rows: int = 20
) -> dict:
    """
    위치기반 관광정보 조회 (locationBasedList2)

    Args:
        map_x: 경도 (longitude)
        map_y: 위도 (latitude)
        radius: 반경 (미터, 기본 5000m)
        content_type_id: 관광타입 (12:관광지, 39:음식점 등)
        arrange: 정렬 (A:제목순, C:수정일순, D:등록일순, E:거리순)
        num_of_rows: 결과 개수

    Returns:
        주변 관광지 목록 (dist: 거리 포함)
    """
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


# ========== 5. 키워드 검색 조회 ==========
@mcp.tool()
async def search_by_keyword(
    keyword: str,
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    content_type_id: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """
    키워드 검색 조회 (searchKeyword2)

    Args:
        keyword: 검색 키워드
        area_code: 지역코드
        sigungu_code: 시군구코드
        content_type_id: 관광타입 (12:관광지, 39:음식점 등)
        arrange: 정렬 (A:제목순, C:수정일순, D:등록일순)
        num_of_rows: 결과 개수

    Returns:
        검색 결과 목록
    """
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


# ========== 6. 행사정보 조회 ==========
@mcp.tool()
async def search_festivals(
    event_start_date: str,
    event_end_date: Optional[str] = None,
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """
    행사/축제 정보 조회 (searchFestival2)

    Args:
        event_start_date: 행사 시작일 (YYYYMMDD)
        event_end_date: 행사 종료일 (YYYYMMDD)
        area_code: 지역코드
        sigungu_code: 시군구코드
        arrange: 정렬
        num_of_rows: 결과 개수

    Returns:
        축제/행사 목록
    """
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


# ========== 7. 숙박정보 조회 ==========
@mcp.tool()
async def search_stays(
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    arrange: str = "A",
    num_of_rows: int = 20
) -> dict:
    """
    숙박정보 조회 (searchStay2)

    Args:
        area_code: 지역코드
        sigungu_code: 시군구코드
        arrange: 정렬
        num_of_rows: 결과 개수

    Returns:
        숙박 목록
    """
    params = {
        "numOfRows": num_of_rows,
        "arrange": arrange
    }
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    return await call_api("searchStay2", params)


# ========== 8. 상세조회 - 공통정보 ==========
@mcp.tool()
async def get_detail_common(
    content_id: str,
    content_type_id: str,
    default_yn: str = "Y",
    first_image_yn: str = "Y",
    addr_info_yn: str = "Y",
    map_info_yn: str = "Y",
    overview_yn: str = "Y"
) -> dict:
    """
    상세조회 - 공통정보 (detailCommon2)

    Args:
        content_id: 콘텐츠 ID
        content_type_id: 관광타입 ID
        default_yn: 기본정보 조회 여부
        first_image_yn: 대표이미지 조회 여부
        addr_info_yn: 주소정보 조회 여부
        map_info_yn: 좌표정보 조회 여부
        overview_yn: 개요정보 조회 여부

    Returns:
        공통 상세정보
    """
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


# ========== 9. 상세조회 - 소개정보 ==========
@mcp.tool()
async def get_detail_intro(content_id: str, content_type_id: str) -> dict:
    """
    상세조회 - 소개정보 (detailIntro2)
    운영시간, 입장료, 연령제한 등 타입별 상세속성

    Args:
        content_id: 콘텐츠 ID
        content_type_id: 관광타입 ID

    Returns:
        소개 상세정보 (관광타입별 속성 다름)
    """
    params = {
        "contentId": content_id,
        "contentTypeId": content_type_id
    }
    return await call_api("detailIntro2", params)


# ========== 10. 상세조회 - 반복정보 ==========
@mcp.tool()
async def get_detail_info(content_id: str, content_type_id: str) -> dict:
    """
    상세조회 - 반복정보 (detailInfo2)
    코스정보, 방정보 등 반복되는 정보

    Args:
        content_id: 콘텐츠 ID
        content_type_id: 관광타입 ID

    Returns:
        반복 상세정보
    """
    params = {
        "contentId": content_id,
        "contentTypeId": content_type_id
    }
    return await call_api("detailInfo2", params)


# ========== 11. 상세조회 - 이미지 ==========
@mcp.tool()
async def get_detail_images(content_id: str) -> dict:
    """
    상세조회 - 이미지 목록 (detailImage2)

    Args:
        content_id: 콘텐츠 ID

    Returns:
        이미지 목록 (originimgurl, smallimageurl)
    """
    params = {
        "contentId": content_id,
        "imageYN": "Y",
        "subImageYN": "Y"
    }
    return await call_api("detailImage2", params)


# ========== 12. 반려동물 여행정보 ==========
@mcp.tool()
async def get_pet_tour_info(content_id: str) -> dict:
    """
    반려동물 여행정보 조회 (detailPetTour2)

    Args:
        content_id: 콘텐츠 ID

    Returns:
        반려동물 동반 가능 정보
    """
    params = {"contentId": content_id}
    return await call_api("detailPetTour2", params)


# ========== 실행 ==========
if __name__ == "__main__":
    mcp.run()
