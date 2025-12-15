"""
EXAONE-3.5-32B Server with MCP Host functionality
LLM이 MCP 도구를 선택하고 호출하는 기능 포함
"""
import logging
import sys
import time
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import httpx
import json
import asyncio
import os
import re
import math
from typing import Optional

# ========== 로깅 설정 ==========
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 로거 생성
logger = logging.getLogger("LLM_MCP")
logger.setLevel(logging.DEBUG)

# 외부 라이브러리 로그 레벨 조절
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger.info("=" * 70)
logger.info(f"LLM MCP 서버 시작: {datetime.now().isoformat()}")
logger.info("=" * 70)

app = FastAPI(title="EXAONE-3.5-32B Server + MCP Host")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 계산 (km) - Haversine 공식"""
    R = 6371  # 지구 반지름 (km)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_nearby_places(places: list) -> list:
    """각 장소에 대해 가까운 장소 정보 추가"""
    for i, place in enumerate(places):
        try:
            lat1 = float(place.get("mapy", 0))
            lon1 = float(place.get("mapx", 0))
            if lat1 == 0 or lon1 == 0:
                continue

            nearby = []
            for j, other in enumerate(places):
                if i == j:
                    continue
                try:
                    lat2 = float(other.get("mapy", 0))
                    lon2 = float(other.get("mapx", 0))
                    if lat2 == 0 or lon2 == 0:
                        continue

                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    if dist < 5:  # 5km 이내
                        nearby.append(f"{other.get('title')}({dist:.1f}km)")
                except:
                    continue

            place["nearby"] = nearby[:3]  # 가까운 장소 최대 3개
        except:
            continue

    return places


def filter_by_geographic_cluster(places: list, max_radius_km: float = 10.0) -> list:
    """
    지리적 클러스터링을 통해 너무 멀리 떨어진 장소 필터링

    전략:
    1. 좌표가 있는 장소들의 중심점 계산
    2. 중심점에서 max_radius_km 이내의 장소만 선택
    3. 좌표 없는 장소는 유지 (제외하면 결과가 너무 적을 수 있음)
    """
    # 좌표가 있는 장소들 분리
    places_with_coords = []
    places_without_coords = []

    for place in places:
        try:
            lat = float(place.get("mapy", 0))
            lon = float(place.get("mapx", 0))
            if lat != 0 and lon != 0:
                places_with_coords.append((place, lat, lon))
            else:
                places_without_coords.append(place)
        except:
            places_without_coords.append(place)

    if len(places_with_coords) < 2:
        print(f"[CLUSTER] Not enough coords ({len(places_with_coords)}), returning all {len(places)} places")
        return places

    # 중심점 계산 (단순 평균)
    avg_lat = sum(p[1] for p in places_with_coords) / len(places_with_coords)
    avg_lon = sum(p[2] for p in places_with_coords) / len(places_with_coords)
    print(f"[CLUSTER] Center point: ({avg_lat:.6f}, {avg_lon:.6f})")

    # 중심점에서 가까운 장소만 선택
    filtered_with_coords = []
    excluded = []
    for place, lat, lon in places_with_coords:
        dist = haversine_distance(avg_lat, avg_lon, lat, lon)
        if dist <= max_radius_km:
            filtered_with_coords.append(place)
        else:
            excluded.append(f"{place.get('title', 'Unknown')}({dist:.1f}km)")

    if excluded:
        print(f"[CLUSTER] Excluded {len(excluded)} places beyond {max_radius_km}km: {excluded[:5]}")

    # 결과 합치기 (좌표 있는 것 + 좌표 없는 것)
    result = filtered_with_coords + places_without_coords
    print(f"[CLUSTER] Filtered: {len(places)} → {len(result)} places (radius={max_radius_km}km)")

    return result


def optimize_route_order(places: list) -> list:
    """
    Greedy 알고리즘으로 동선 최적화 (가까운 순서로 정렬)

    시작점: 첫 번째 장소
    다음 장소: 현재 위치에서 가장 가까운 미방문 장소
    """
    if len(places) < 2:
        return places

    # 좌표가 있는 장소만 최적화 대상
    coords = []
    for i, place in enumerate(places):
        try:
            lat = float(place.get("mapy", 0))
            lon = float(place.get("mapx", 0))
            if lat != 0 and lon != 0:
                coords.append((i, lat, lon))
        except:
            pass

    if len(coords) < 2:
        return places

    # Greedy TSP
    visited = set()
    route = [coords[0][0]]  # 첫 번째 장소부터 시작
    visited.add(coords[0][0])
    current_lat, current_lon = coords[0][1], coords[0][2]

    while len(visited) < len(coords):
        nearest = None
        nearest_dist = float('inf')

        for idx, lat, lon in coords:
            if idx not in visited:
                dist = haversine_distance(current_lat, current_lon, lat, lon)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (idx, lat, lon)

        if nearest:
            route.append(nearest[0])
            visited.add(nearest[0])
            current_lat, current_lon = nearest[1], nearest[2]

    # 좌표 없는 장소들 인덱스
    no_coord_indices = [i for i in range(len(places)) if i not in visited]

    # 최적화된 순서로 재배열
    optimized = [places[i] for i in route] + [places[i] for i in no_coord_indices]

    print(f"[ROUTE] Optimized route order: {[places[i].get('title', '')[:10] for i in route[:5]]}...")

    return optimized


# ========== MCP Server 설정 ==========
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# ========== 모델 로딩 ==========
print("Loading model...")
model_id = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_memory={0: "100GiB", "cpu": "50GiB"},
    offload_buffers=True,
)

device = next(model.parameters()).device
print(f"Model loaded! Device: {device}")

# ========== MCP 도구 정의 (tour-mcp-server 기준) ==========
MCP_TOOLS = [
    {
        "name": "get_area_codes",
        "description": "지역코드/시군구코드 목록 조회",
        "parameters": {"area_code": "선택 - 지역코드"}
    },
    {
        "name": "search_by_area",
        "description": "지역기반 관광정보 검색. 특정 지역의 관광지, 음식점, 숙박 등 조회",
        "parameters": {
            "area_code": "필수 - 지역코드",
            "content_type_id": "선택 - 콘텐츠타입",
            "sigungu_code": "선택 - 시군구코드",
            "num_of_rows": "선택 - 결과개수"
        }
    },
    {
        "name": "search_by_keyword",
        "description": "키워드로 관광정보 검색. 가장 유연한 검색 방법",
        "parameters": {
            "keyword": "필수 - 검색키워드",
            "area_code": "선택 - 지역코드",
            "content_type_id": "선택 - 콘텐츠타입",
            "num_of_rows": "선택 - 결과개수"
        }
    },
    {
        "name": "search_by_location",
        "description": "GPS 위치 기반 주변 관광정보 검색",
        "parameters": {
            "map_x": "필수 - 경도",
            "map_y": "필수 - 위도",
            "radius": "선택 - 반경(미터)",
            "content_type_id": "선택 - 콘텐츠타입"
        }
    },
    {
        "name": "search_festivals",
        "description": "축제/행사 정보 검색",
        "parameters": {
            "event_start_date": "필수 - 시작일(YYYYMMDD)",
            "event_end_date": "선택 - 종료일(YYYYMMDD)",
            "area_code": "선택 - 지역코드"
        }
    },
    {
        "name": "search_stays",
        "description": "숙박 정보 검색",
        "parameters": {
            "area_code": "선택 - 지역코드",
            "sigungu_code": "선택 - 시군구코드"
        }
    },
    {
        "name": "get_detail_common",
        "description": "관광지 상세정보 조회 (주소, 이미지, 개요 등)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_detail_intro",
        "description": "관광지 소개정보 조회 (운영시간, 입장료 등)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_detail_images",
        "description": "관광지 이미지 목록 조회",
        "parameters": {"content_id": "필수 - 콘텐츠ID"}
    },
    {
        "name": "get_category_codes",
        "description": "서비스 분류코드 조회",
        "parameters": {
            "content_type_id": "선택 - 콘텐츠타입",
            "cat1": "선택 - 대분류",
            "cat2": "선택 - 중분류"
        }
    },
    {
        "name": "get_detail_info",
        "description": "소개정보 조회 (반복정보)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_pet_tour_info",
        "description": "반려동물 여행정보 조회",
        "parameters": {
            "area_code": "선택 - 지역코드",
            "sigungu_code": "선택 - 시군구코드"
        }
    }
]

# 지역코드 매핑
AREA_CODES = {
    "서울": "1", "인천": "2", "대전": "3", "대구": "4", "광주": "5",
    "부산": "6", "울산": "7", "세종": "8", "경기": "31", "강원": "32",
    "충북": "33", "충남": "34", "경북": "35", "경남": "36",
    "전북": "37", "전남": "38", "제주": "39"
}

CONTENT_TYPES = {
    "관광지": "12", "문화시설": "14", "축제": "15", "여행코스": "25",
    "레포츠": "28", "숙박": "32", "쇼핑": "38", "음식점": "39", "카페": "39"
}

# Need 타입 → Content Type 매핑
NEED_TO_CONTENT_TYPE = {
    "food": "39",      # 음식점
    "cafe": "39",      # 카페 (음식점과 동일)
    "spot": "12",      # 관광지
    "stay": "32",      # 숙박
    "culture": "14",   # 문화시설
    "shopping": "38",  # 쇼핑
    "festival": "15",  # 축제
}

# 최소 결과 개수 기준
MIN_RESULTS_THRESHOLD = 3

# 🔴 LLM 키워드 정규화 캐시 (동일 키워드 반복 요청 방지)
KEYWORD_CACHE = {}


def normalize_keywords_batch(user_keywords: list[str]) -> dict[str, list[str]]:
    """
    LLM을 사용해 여러 사용자 키워드를 한번에 API 검색 키워드로 변환 (배치 처리)

    Input: ["고깃집", "횟집", "일식"]
    Output: {
        "고깃집": ["고기", "삼겹살", "갈비"],
        "횟집": ["횟집", "회", "해물"],
        "일식": ["초밥", "라멘", "스시"]
    }
    """
    result = {}
    uncached_keywords = []

    # 캐시된 것 먼저 처리
    for kw in user_keywords:
        if kw in KEYWORD_CACHE:
            result[kw] = KEYWORD_CACHE[kw]
            print(f"[NORMALIZE] Cache hit: '{kw}' → {KEYWORD_CACHE[kw]}")
        else:
            uncached_keywords.append(kw)

    # 캐시에 없는 것만 LLM 호출 (배치)
    if not uncached_keywords:
        return result

    keywords_str = ", ".join(uncached_keywords)
    prompt = f"""한국관광공사 API에서 음식점을 검색하려고 합니다.
사용자가 다음 음식점들을 찾고 있습니다: {keywords_str}

API는 음식점 이름에 포함된 키워드로 검색합니다.
각 키워드별로 음식점 이름에 자주 포함되는 검색어 3개씩 추천해주세요.

예시:
- 고깃집 → ["고기", "삼겹살", "갈비"]
- 횟집 → ["횟집", "회", "해물"]
- 일식집 → ["초밥", "라멘", "스시"]
- 중국집 → ["짬뽕", "중화", "반점"]
- 치킨집 → ["치킨", "통닭", "후라이드"]
- 카페 → ["카페", "커피", "베이커리"]

응답 형식 (JSON만, 설명 없이):
{{
  "키워드1": ["검색어1", "검색어2", "검색어3"],
  "키워드2": ["검색어1", "검색어2", "검색어3"]
}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = generate_response(messages, max_tokens=300, temperature=0.1)
        print(f"[NORMALIZE] LLM batch response: {response}")

        # JSON 파싱
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(response[json_start:json_end])
            if isinstance(parsed, dict):
                for kw in uncached_keywords:
                    if kw in parsed and isinstance(parsed[kw], list):
                        result[kw] = parsed[kw]
                        KEYWORD_CACHE[kw] = parsed[kw]  # 캐시 저장
                        print(f"[NORMALIZE] '{kw}' → {parsed[kw]}")
                    else:
                        # LLM이 해당 키워드를 반환하지 않은 경우
                        result[kw] = [kw]
                        print(f"[NORMALIZE] '{kw}' → fallback to original")
                return result
    except Exception as e:
        print(f"[NORMALIZE] Batch error: {e}")

    # 실패시 원본 키워드 반환
    for kw in uncached_keywords:
        result[kw] = [kw]
    return result


def normalize_keyword_with_llm(user_keyword: str) -> list[str]:
    """단일 키워드 정규화 (배치 함수의 wrapper)"""
    result = normalize_keywords_batch([user_keyword])
    return result.get(user_keyword, [user_keyword])


# ========== Request/Response 모델 ==========
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7

class MCPQueryRequest(BaseModel):
    query: str  # 자연어 쿼리: "강릉 바다 근처 맛집 추천해줘"
    area_code: Optional[str] = None  # 모바일에서 선택한 도 코드 (예: "32" for 강원)
    sigungu_code: Optional[str] = None  # 모바일에서 선택한 시/군/구 코드
    max_tokens: int = 1024
    temperature: float = 0.3


# ========== LLM 생성 함수 ==========
def generate_response(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


# ========== MCP 도구 호출 ==========
async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """MCP 서버의 도구 호출 (HTTP API)"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{MCP_SERVER_URL}/api/call_tool",
                json={"name": tool_name, "arguments": arguments}
            )
            result = response.json()
            if result.get("success"):
                return result.get("result", {})
            else:
                return {"error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"error": str(e)}


async def call_mcp_tool_direct(tool_name: str, arguments: dict) -> dict:
    """MCP 서버 직접 호출 (HTTP API)"""
    # tour-mcp-server의 함수를 직접 HTTP로 호출
    # FastMCP는 기본적으로 stdio지만, HTTP wrapper 추가 필요

    # 임시: 직접 Tour API 호출
    async with httpx.AsyncClient(timeout=30) as client:
        base_url = "https://apis.data.go.kr/B551011/KorService2"
        api_key = os.getenv("TOUR_API_KEY", "")

        common_params = {
            "serviceKey": api_key,
            "MobileOS": "ETC",
            "MobileApp": "TravelMCP",
            "_type": "json",
            "numOfRows": arguments.get("num_of_rows", 20),
            "arrange": arguments.get("arrange", "B")  # 기본값: 조회순(인기순)
        }

        endpoint_map = {
            "get_area_codes": "areaCode2",
            "search_by_area": "areaBasedList2",
            "search_by_keyword": "searchKeyword2",
            "search_by_location": "locationBasedList2",
            "search_festivals": "searchFestival2",
            "search_stays": "searchStay2",
            "get_detail_common": "detailCommon2",
            "get_detail_intro": "detailIntro2",
            "get_detail_images": "detailImage2",
        }

        endpoint = endpoint_map.get(tool_name)
        if not endpoint:
            return {"error": f"Unknown tool: {tool_name}"}

        params = {**common_params}

        # 파라미터 매핑
        if "area_code" in arguments:
            params["areaCode"] = arguments["area_code"]
        if "sigungu_code" in arguments:
            params["sigunguCode"] = arguments["sigungu_code"]
        if "content_type_id" in arguments:
            params["contentTypeId"] = arguments["content_type_id"]
        if "keyword" in arguments:
            params["keyword"] = arguments["keyword"]
        if "map_x" in arguments:
            params["mapX"] = arguments["map_x"]
        if "map_y" in arguments:
            params["mapY"] = arguments["map_y"]
        if "radius" in arguments:
            params["radius"] = arguments["radius"]
        if "event_start_date" in arguments:
            params["eventStartDate"] = arguments["event_start_date"]
        if "content_id" in arguments:
            params["contentId"] = arguments["content_id"]
            params["defaultYN"] = "Y"
            params["firstImageYN"] = "Y"
            params["addrinfoYN"] = "Y"
            params["overviewYN"] = "Y"

        try:
            response = await client.get(f"{base_url}/{endpoint}", params=params)
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
                "items": item_list[:20],  # 최대 20개
                "totalCount": data["response"]["body"].get("totalCount", len(item_list))
            }
        except Exception as e:
            return {"error": str(e)}


# ========== 오케스트레이션 헬퍼 함수 ==========
async def search_by_keyword_direct(keyword: str, area_code: str, sigungu_code: str, content_type_id: str = None, num_rows: int = 20) -> dict:
    """키워드 검색 직접 호출"""
    args = {
        "keyword": keyword,
        "area_code": area_code,
        "num_of_rows": num_rows
    }
    if sigungu_code:
        args["sigungu_code"] = sigungu_code
    if content_type_id:
        args["content_type_id"] = content_type_id

    return await call_mcp_tool_direct("search_by_keyword", args)


async def search_by_area_direct(area_code: str, sigungu_code: str, content_type_id: str = None, num_rows: int = 20, arrange: str = "B") -> dict:
    """지역 기반 검색 직접 호출

    arrange 옵션:
    - A: 제목순 (기본)
    - B: 조회순 (인기순) ⭐ 추천
    - C: 수정일순
    - D: 생성일순
    """
    args = {
        "area_code": area_code,
        "num_of_rows": num_rows,
        "arrange": arrange  # 인기순 정렬
    }
    if sigungu_code:
        args["sigungu_code"] = sigungu_code
    if content_type_id:
        args["content_type_id"] = content_type_id

    return await call_mcp_tool_direct("search_by_area", args)


def analyze_query_needs(query: str) -> dict:
    """쿼리에서 필요한 것들을 분석 (LLM 없이 규칙 기반)

    Returns:
        dict: {
            "food": ["맛집", "돈까스"],  # 일반 + 구체적 키워드
            "food_specific": ["돈까스"],  # 구체적인 음식만 (직접 검색용)
            "cafe": ["카페"],
            ...
        }
    """
    query_lower = query.lower()
    needs = {}

    # 구체적인 음식 키워드 (API 키워드 검색에 직접 사용)
    # 🔴 "일식", "고기" 등 카테고리 키워드도 추가 - 사용자가 원하는 정확한 검색 위해
    specific_food_keywords = [
        # 메뉴 종류
        "돈까스", "돈가스", "삼겹살", "치킨", "피자", "파스타", "스테이크",
        "초밥", "회", "라멘", "우동", "냉면", "막국수", "칼국수", "짜장면", "짬뽕",
        "떡볶이", "순대", "김밥", "비빔밥", "불고기", "갈비", "삼계탕", "설렁탕",
        "순두부", "부대찌개", "감자탕", "곱창", "족발", "보쌈", "치즈", "버거", "햄버거",
        "아이스크림", "빙수", "와플", "마카롱", "케이크",
        # 음식 카테고리 (KEYWORD_TO_API_KEYWORDS 매핑 활용)
        "일식", "일식집", "중식", "중식집", "한식", "양식",
        "고기", "고깃집", "고기집", "횟집", "해산물", "분식", "디저트"
    ]
    specific_matches = [kw for kw in specific_food_keywords if kw in query_lower]
    if specific_matches:
        needs["food_specific"] = specific_matches  # 직접 검색용

    # 음식 관련 일반 키워드 (구체적 키워드는 위에서 처리)
    food_keywords = ["맛집", "음식", "밥", "식당", "먹", "점심", "저녁", "아침"]
    food_matches = [kw for kw in food_keywords if kw in query_lower]
    if food_matches or specific_matches:
        needs["food"] = food_matches + specific_matches

    # 카페 관련 키워드
    cafe_keywords = ["카페", "커피", "디저트", "빵", "베이커리", "브런치", "차", "음료"]
    cafe_matches = [kw for kw in cafe_keywords if kw in query_lower]
    if cafe_matches:
        needs["cafe"] = cafe_matches

    # 관광지 관련 키워드 (바닷가, 해안 등 추가)
    spot_keywords = ["관광", "명소", "볼거리", "구경", "바다", "바닷가", "해안", "산", "공원", "해변",
                     "전망", "야경", "사진", "인스타", "데이트", "드라이브", "자연", "풍경", "경치", "산책"]
    spot_matches = [kw for kw in spot_keywords if kw in query_lower]
    if spot_matches:
        needs["spot"] = spot_matches

    # 숙박 관련 키워드
    stay_keywords = ["숙소", "호텔", "펜션", "모텔", "숙박", "잠", "묵", "리조트", "게스트하우스"]
    stay_matches = [kw for kw in stay_keywords if kw in query_lower]
    if stay_matches:
        needs["stay"] = stay_matches

    # 문화시설 관련 키워드
    culture_keywords = ["박물관", "미술관", "전시", "공연", "영화", "문화", "역사"]
    culture_matches = [kw for kw in culture_keywords if kw in query_lower]
    if culture_matches:
        needs["culture"] = culture_matches

    # 아무것도 매칭 안되면 기본적으로 관광지 + 음식점
    if not needs:
        needs["spot"] = ["관광"]
        needs["food"] = ["맛집"]

    # 🔴 사용자가 요청한 순서대로 카테고리 추출 (검색 및 큐레이션에 활용)
    # 예: "카페갔다가 돈까스먹고 저녁은 회" → ["카페", "돈까스", "횟집"]
    user_order = []
    order_keywords = [
        # (카테고리명, [키워드들], cat3 코드 또는 None)
        ("카페", ["카페", "커피", "디저트"], "A05020900"),
        ("관광지", ["바다", "바닷가", "해변", "관광", "구경", "산책", "공원"], None),
        ("치킨", ["치킨", "통닭", "후라이드"], "A05020700"),
        ("횟집", ["횟집", "회", "해산물", "생선", "해물"], "A05020100"),  # 한식-해물
        ("고깃집", ["고기", "고깃집", "삼겹살", "갈비", "소고기", "돼지"], "A05020100"),  # 한식
        ("돈까스", ["돈까스", "돈가스", "까스"], "A05020200"),  # 서양식
        ("일식", ["일식", "초밥", "라멘", "스시", "우동"], "A05020300"),
        ("한식", ["한식", "한정식", "백반", "비빔밥", "김치"], "A05020100"),
        ("중식", ["중식", "중국집", "짜장", "짬뽕", "탕수육"], "A05020400"),
        ("양식", ["양식", "파스타", "스테이크", "피자"], "A05020200"),
        ("분식", ["분식", "떡볶이", "순대", "김밥"], "A05020600"),
    ]

    # 쿼리에서 각 카테고리의 첫 등장 위치 찾기
    category_positions = []
    for cat_name, keywords, cat3 in order_keywords:
        for kw in keywords:
            pos = query_lower.find(kw)
            if pos >= 0:
                category_positions.append((pos, cat_name, cat3))
                break

    # 등장 순서대로 정렬
    category_positions.sort(key=lambda x: x[0])
    user_order = [(cat, cat3) for _, cat, cat3 in category_positions]

    if user_order:
        needs["user_order"] = user_order  # [(카테고리명, cat3코드), ...]
        print(f"[ORCH] User requested order: {[cat for cat, _ in user_order]}")

    print(f"[ORCH] Analyzed needs: {needs}")
    return needs


async def orchestrated_search(query: str, area_code: str, sigungu_code: str, needs: dict) -> dict:
    """
    오케스트레이션된 검색 - 폴백 전략 포함

    전략:
    0. 🔴 사용자가 요청한 카테고리 순서대로 검색 (카페 → 돈까스 → 회)
    1. 구체적인 음식 키워드가 있으면 검색 (돈까스, 피자 등)
    2. 키워드 검색 시도 (매칭된 키워드로)
    3. 결과 부족시 → 카테고리 기반 검색
    4. 여전히 부족시 → 지역 전체 검색
    """
    all_results = {}
    search_log = []

    # 🔴 Strategy 0: 사용자가 요청한 카테고리 순서대로 검색!
    # 예: "카페갔다가 돈까스먹고 저녁은 회" → 카페, 돈까스, 횟집 각각 검색
    if "user_order" in needs and needs["user_order"]:
        user_order = needs["user_order"]
        print(f"[ORCH] Strategy 0: Searching for user-requested categories: {[cat for cat, _ in user_order]}")

        for cat_name, cat3 in user_order:
            cat_results = {"items": [], "category": cat_name, "cat3": cat3}

            # 카테고리별 검색 키워드 매핑
            search_keywords = {
                "카페": ["카페", "커피", "베이커리"],
                "관광지": ["관광", "명소"],
                "치킨": ["치킨", "통닭"],
                "횟집": ["횟집", "회", "해물"],
                "고깃집": ["고기", "삼겹살", "갈비"],
                "돈까스": ["돈까스", "돈가스", "카츠"],
                "일식": ["초밥", "일식", "라멘"],
                "한식": ["한식", "한정식"],
                "중식": ["중국집", "짬뽕", "짜장"],
                "양식": ["파스타", "스테이크", "양식"],
                "분식": ["분식", "떡볶이"],
            }

            keywords = search_keywords.get(cat_name, [cat_name])
            content_type = "12" if cat_name == "관광지" else "39"  # 관광지면 12, 아니면 음식점

            for kw in keywords[:2]:
                print(f"[ORCH] Strategy 0: Searching '{kw}' for category '{cat_name}'")
                result = await search_by_keyword_direct(kw, area_code, sigungu_code, content_type)
                items = result.get("items", [])
                search_log.append(f"user_order:{cat_name}→{kw}→{len(items)}개")

                if items:
                    # cat3 필터링 (카페, 돈까스 등 세부 분류)
                    if cat3:
                        filtered_items = [i for i in items if i.get("cat3", "").startswith(cat3[:7])]  # A050209xx 식으로 prefix 매칭
                        if filtered_items:
                            items = filtered_items
                            print(f"[ORCH] Filtered by cat3 {cat3}: {len(items)} items")

                    # 중복 제거하며 추가
                    existing_ids = {i.get("contentid") for i in cat_results["items"]}
                    for item in items:
                        if item.get("contentid") not in existing_ids:
                            # 🔴 검색된 아이템에 원래 요청 카테고리 태깅
                            item["_user_category"] = cat_name
                            cat_results["items"].append(item)
                            existing_ids.add(item.get("contentid"))

                    if len(cat_results["items"]) >= 5:
                        break

            if cat_results["items"]:
                all_results[f"user_{cat_name}"] = cat_results
                print(f"[ORCH] Found {len(cat_results['items'])} items for user category '{cat_name}'")

    # Strategy 1: 구체적인 음식 키워드 최우선 검색 (돈까스, 피자 등)
    if "food_specific" in needs:
        specific_results = {"items": []}

        # 🔴 LLM 배치 정규화: 모든 키워드를 한번에 처리!
        # ["고깃집", "횟집", "일식"] → {"고깃집": ["고기",...], "횟집": ["회",...], ...}
        keyword_mapping = normalize_keywords_batch(needs["food_specific"])
        print(f"[ORCH] Strategy 0: Batch normalized {len(keyword_mapping)} keywords")

        for kw, api_keywords in keyword_mapping.items():
            print(f"[ORCH] Strategy 0: '{kw}' → API keywords: {api_keywords}")

            for api_kw in api_keywords[:3]:  # 최대 3개 API 키워드 시도
                print(f"[ORCH] Strategy 0: Searching '{api_kw}' for user keyword '{kw}'")
                result = await search_by_keyword_direct(api_kw, area_code, sigungu_code, "39")  # 음식점
                items = result.get("items", [])
                search_log.append(f"specific:{kw}→{api_kw}→{len(items)}개")

                if items:
                    # 중복 제거하며 추가
                    existing_ids = {i.get("contentid") for i in specific_results["items"]}
                    for item in items:
                        if item.get("contentid") not in existing_ids:
                            specific_results["items"].append(item)
                            existing_ids.add(item.get("contentid"))

                    # 해당 키워드에서 충분한 결과가 모이면 다음 키워드로
                    if len([i for i in specific_results["items"]]) >= 5:
                        break

        if specific_results["items"]:
            all_results["food_specific"] = specific_results
            print(f"[ORCH] Found {len(specific_results['items'])} specific food items!")

    for need_type, keywords in needs.items():
        # food_specific은 이미 처리됨
        if need_type == "food_specific":
            continue

        # 🔴 food_specific이 있으면 food need는 건너뛰기 (중복 검색 방지)
        # 치킨 검색했으면 일반 음식점 검색 안함 → 고깃집/횟집 섞임 방지
        if need_type == "food" and "food_specific" in needs:
            print(f"[ORCH] Skipping 'food' need (food_specific already processed)")
            continue

        content_type = NEED_TO_CONTENT_TYPE.get(need_type)
        results_for_need = {"items": []}

        # Strategy 1: 키워드 검색 (매칭된 키워드 중 하나로)
        if keywords and isinstance(keywords, list):
            for kw in keywords[:2]:  # 최대 2개 키워드 시도
                print(f"[ORCH] Strategy 1: keyword search '{kw}' for {need_type}")
                result = await search_by_keyword_direct(kw, area_code, sigungu_code, content_type)
                items = result.get("items", [])
                search_log.append(f"keyword:{kw}→{len(items)}개")

                if len(items) >= MIN_RESULTS_THRESHOLD:
                    results_for_need = result
                    break
                elif items:
                    # 부분 결과라도 저장
                    results_for_need["items"].extend(items)

        # Strategy 2: 카테고리 기반 검색 (결과 부족시)
        if len(results_for_need.get("items", [])) < MIN_RESULTS_THRESHOLD:
            print(f"[ORCH] Strategy 2: area search with content_type={content_type}")
            result = await search_by_area_direct(area_code, sigungu_code, content_type)
            items = result.get("items", [])
            search_log.append(f"area+type:{content_type}→{len(items)}개")

            if items:
                # 기존 결과에 추가 (중복 제거)
                existing_ids = {i.get("contentid") for i in results_for_need.get("items", [])}
                for item in items:
                    if item.get("contentid") not in existing_ids:
                        results_for_need["items"].append(item)

        # Strategy 3: 더 많은 결과 요청 (같은 카테고리 유지!)
        # 🔴 content_type 없이 검색하면 고깃집/횟집 등 관련 없는 결과가 섞임 → 제거
        if len(results_for_need.get("items", [])) < MIN_RESULTS_THRESHOLD and content_type:
            print(f"[ORCH] Strategy 3: expanded area search with content_type={content_type}")
            result = await search_by_area_direct(area_code, sigungu_code, content_type, num_rows=30)
            items = result.get("items", [])
            search_log.append(f"area_expanded:{content_type}→{len(items)}개")

            if items:
                existing_ids = {i.get("contentid") for i in results_for_need.get("items", [])}
                for item in items:
                    if item.get("contentid") not in existing_ids:
                        results_for_need["items"].append(item)

        all_results[need_type] = results_for_need
        print(f"[ORCH] {need_type}: {len(results_for_need.get('items', []))} items collected")

    # 결과 합치기 (사용자 요청 카테고리 우선!)
    combined_items = []
    seen_ids = set()

    # 🔴 1. 사용자가 요청한 카테고리 결과 먼저 추가 (순서대로!)
    user_order = needs.get("user_order", [])
    for cat_name, _ in user_order:
        key = f"user_{cat_name}"
        if key in all_results:
            for item in all_results[key].get("items", []):
                cid = item.get("contentid")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    combined_items.append(item)
            print(f"[ORCH] Added {len([i for i in combined_items if i.get('_user_category') == cat_name])} items for user category '{cat_name}'")

    # 2. 구체적인 음식 검색 결과 추가 (돈까스 검색했으면 돈까스집)
    if "food_specific" in all_results:
        for item in all_results["food_specific"].get("items", []):
            cid = item.get("contentid")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                combined_items.append(item)
        print(f"[ORCH] Added specific food items, total now: {len(combined_items)}")

    # 3. 나머지 결과 추가
    for need_type, result in all_results.items():
        if need_type == "food_specific" or need_type.startswith("user_"):
            continue  # 이미 처리됨
        for item in result.get("items", []):
            cid = item.get("contentid")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                combined_items.append(item)

    # 🔴 user_order 정보도 반환 (curate에서 활용)
    return {
        "items": combined_items,
        "totalCount": len(combined_items),
        "search_log": search_log,
        "needs_analyzed": list(needs.keys()),
        "user_order": user_order
    }


# ========== LLM 기반 도구 선택 ==========
def select_tools_with_llm(query: str, area_code: Optional[str] = None, sigungu_code: Optional[str] = None) -> list[dict]:
    """LLM을 사용해 쿼리에 맞는 도구와 파라미터 선택"""

    tools_description = "\n".join([
        f"- {t['name']}: {t['description']}\n  파라미터: {t['parameters']}" for t in MCP_TOOLS
    ])

    # area_code + sigungu_code가 제공된 경우 프롬프트에 명시적으로 주입
    area_context = ""
    if area_code and sigungu_code:
        area_context = f"""
**🔴 매우 중요: 사용자가 이미 지역을 선택했습니다 🔴**
- area_code: "{area_code}" (도/광역시 코드 - 반드시 사용)
- sigungu_code: "{sigungu_code}" (시/군/구 코드 - 반드시 사용)

**🔴 area_code + sigungu_code가 제공되면 반드시 search_by_area를 사용하세요! 🔴**
- search_by_area는 키워드 없이 지역+콘텐츠타입으로 검색합니다
- arguments에 area_code와 sigungu_code 둘 다 반드시 포함!
- 키워드 검색(search_by_keyword)은 사용하지 마세요
"""

    prompt = f"""당신은 여행 정보 검색을 위한 도구 선택 AI입니다.
사용자의 질문을 분석하고, 적절한 도구와 파라미터를 JSON 형식으로 반환하세요.

{area_context}
## 사용 가능한 도구:
{tools_description}

## 지역코드 (area_code) - 사용자가 제공하지 않은 경우에만 참고:
서울=1, 인천=2, 대전=3, 대구=4, 광주=5, 부산=6, 울산=7, 세종=8
경기=31, 강원=32, 충북=33, 충남=34, 경북=35, 경남=36, 전북=37, 전남=38, 제주=39

## 콘텐츠타입 (content_type_id):
관광지=12, 문화시설=14, 축제=15, 여행코스=25, 레포츠=28, 숙박=32, 쇼핑=38, 음식점/카페=39

## 예시 (area_code + sigungu_code 제공된 경우) - search_by_area 사용!:
질문: "맛집 추천해줘"
제공된 area_code: "6", sigungu_code: "7"
응답: {{"tools": [{{"name": "search_by_area", "arguments": {{"area_code": "6", "sigungu_code": "7", "content_type_id": "39", "num_of_rows": 20}}}}]}}

질문: "카페랑 관광지 알려줘"
제공된 area_code: "32", sigungu_code: "1"
응답: {{"tools": [{{"name": "search_by_area", "arguments": {{"area_code": "32", "sigungu_code": "1", "content_type_id": "39", "num_of_rows": 15}}}}, {{"name": "search_by_area", "arguments": {{"area_code": "32", "sigungu_code": "1", "content_type_id": "12", "num_of_rows": 15}}}}]}}

질문: "데이트하기 좋은 곳"
제공된 area_code: "1", sigungu_code: "24"
응답: {{"tools": [{{"name": "search_by_area", "arguments": {{"area_code": "1", "sigungu_code": "24", "content_type_id": "12", "num_of_rows": 15}}}}, {{"name": "search_by_area", "arguments": {{"area_code": "1", "sigungu_code": "24", "content_type_id": "39", "num_of_rows": 15}}}}]}}

## 예시 (area_code가 제공되지 않은 경우) - search_by_keyword 사용:
질문: "강릉 바다 근처 맛집 추천해줘"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "맛집", "area_code": "32", "content_type_id": "39", "num_of_rows": 20}}}}]}}

질문: "부산 해운대 숙박"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "해운대", "area_code": "6", "content_type_id": "32", "num_of_rows": 20}}}}]}}

## 핵심 규칙:
1. **area_code + sigungu_code가 제공되면 → search_by_area 사용 (키워드 검색 금지)**
2. **search_by_area 사용시 area_code와 sigungu_code 둘 다 arguments에 필수 포함!**
3. **지역코드가 없으면 → search_by_keyword 사용 (키워드는 간단한 명사 1~2개만)**
4. 음식점/맛집/카페 → content_type_id="39"
5. 숙박/호텔/펜션 → content_type_id="32"
6. 관광지/명소 → content_type_id="12"
7. 여러 종류 요청시 → 도구를 여러 개 사용
8. num_of_rows는 15~20 권장

## 사용자 질문:
{query}

## 응답 (JSON만 출력, 설명 없이):
{{"tools": [...]}}"""

    messages = [{"role": "user", "content": prompt}]

    print(f"[MCP DEBUG] Sending prompt to LLM for tool selection...")
    print(f"[MCP DEBUG] area_code={area_code}, sigungu_code={sigungu_code}")

    response = generate_response(messages, max_tokens=500, temperature=0.1)

    # 디버그: LLM의 raw response 출력
    print(f"[MCP DEBUG] LLM raw response:\n{response}")
    print(f"[MCP DEBUG] Response length: {len(response)}")

    # JSON 파싱 - 첫 번째 완전한 JSON 객체만 추출 (bracket counting)
    try:
        json_start = response.find("{")
        if json_start < 0:
            print(f"[MCP DEBUG] No JSON object found in response!")
            return []

        # Bracket counting으로 첫 번째 완전한 JSON 객체 찾기
        bracket_count = 0
        json_end = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(response[json_start:], start=json_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i + 1
                    break

        print(f"[MCP DEBUG] JSON range: {json_start} to {json_end}")

        if json_end > json_start:
            json_str = response[json_start:json_end]
            print(f"[MCP DEBUG] Extracted JSON:\n{json_str}")

            # LLM이 JSON에 주석(//)을 포함시키는 경우가 있어서 제거
            # 문자열 내부가 아닌 주석만 제거 (라인 끝까지)
            json_str_clean = re.sub(r'//[^\n]*', '', json_str)
            # 쉼표 뒤에 바로 }나 ]가 오는 경우 수정 (trailing comma)
            json_str_clean = re.sub(r',(\s*[}\]])', r'\1', json_str_clean)
            print(f"[MCP DEBUG] Cleaned JSON:\n{json_str_clean}")

            result = json.loads(json_str_clean)
            tools = result.get("tools", [])
            print(f"[MCP DEBUG] Parsed tools count: {len(tools)}")
            return tools
        else:
            print(f"[MCP DEBUG] Could not find matching closing bracket!")
    except json.JSONDecodeError as e:
        print(f"[MCP DEBUG] JSON parsing error: {e}")
        print(f"[MCP DEBUG] Failed JSON string: {json_str if 'json_str' in locals() else 'N/A'}")

    # 파싱 실패시 기본값
    print(f"[MCP DEBUG] Returning empty tools list due to parsing failure")
    return []


def curate_results_with_llm(query: str, tool_results: list[dict], user_order: list = None) -> dict:
    """LLM을 사용해 검색 결과를 큐레이션 - spots(리스트뷰) + course(코스뷰) 분리

    Args:
        user_order: 사용자가 요청한 카테고리 순서 [(카테고리명, cat3코드), ...]
    """
    user_order = user_order or []

    # 🔴 카테고리별로 분류해서 균형있게 선택
    # 1. 사용자 요청 카테고리 (_user_category 태그 활용)
    # 2. 기본 카테고리 (관광지, 음식점, 카페 등)
    items_by_category = {
        "12": [],      # 관광지
        "14": [],      # 문화시설
        "32": [],      # 숙박
        "39": [],      # 음식점 (카페 제외)
        "cafe": [],    # 카페 (별도 분리)
    }

    # 🔴 사용자 요청 카테고리별 분류 추가
    items_by_user_category = {}
    for cat_name, _ in user_order:
        items_by_user_category[cat_name] = []

    for result in tool_results:
        if "items" in result and result["items"]:
            for item in result["items"]:
                content_type = item.get("contenttypeid", "39")
                cat3 = item.get("cat3", "")

                # 🔴 사용자 요청 카테고리가 있으면 우선 분류
                user_cat = item.get("_user_category")
                if user_cat and user_cat in items_by_user_category:
                    items_by_user_category[user_cat].append(item)
                    continue  # 사용자 카테고리로 분류되면 기본 분류 스킵

                # 🔴 카페(A05020900)는 별도 카테고리로 분리
                if content_type == "39" and cat3 == "A05020900":
                    items_by_category["cafe"].append(item)
                elif content_type in items_by_category:
                    items_by_category[content_type].append(item)
                else:
                    items_by_category["39"].append(item)  # 기본값: 음식점

    # 사용자 요청 카테고리 통계 출력
    for cat_name, items in items_by_user_category.items():
        if items:
            print(f"[CURATE] User category '{cat_name}': {len(items)} items")

    # 기본 카테고리별 통계 출력
    for cat, items in items_by_category.items():
        if items:
            cat_name = {"12": "관광지", "14": "문화시설", "32": "숙박", "39": "음식점", "cafe": "카페"}.get(cat, cat)
            print(f"[CURATE] Category {cat_name}: {len(items)} items")

    # 🔴 사용자 요청 카테고리 우선 선택!
    MAX_PER_CATEGORY = 8
    results_summary = []

    # 1. 사용자가 요청한 카테고리에서 먼저 선택 (각 카테고리에서 최소 3개)
    for cat_name, _ in user_order:
        items = items_by_user_category.get(cat_name, [])
        for item in items[:max(3, MAX_PER_CATEGORY)]:  # 최소 3개, 최대 8개
            cat3 = item.get("cat3", "")
            content_type = item.get("contenttypeid", "39")
            # 한글 카테고리명으로 변환 - 사용자 요청 카테고리명 우선 사용
            category_name = cat_name  # 사용자 요청 카테고리명 그대로 사용

            results_summary.append({
                "title": item.get("title", ""),
                "addr": item.get("addr1", ""),
                "type": content_type,
                "cat3": cat3,
                "category": category_name,  # 사용자 요청 카테고리명!
                "image": item.get("firstimage", ""),
                "mapx": item.get("mapx", ""),
                "mapy": item.get("mapy", ""),
                "tel": item.get("tel", ""),
                "content_id": item.get("contentid", ""),
                "_user_category": cat_name
            })
        print(f"[CURATE] Selected {min(len(items), max(3, MAX_PER_CATEGORY))} items for user category '{cat_name}'")

    # 2. 나머지 카테고리에서 추가 선택
    for content_type, items in items_by_category.items():
        for item in items[:MAX_PER_CATEGORY]:
            cat3 = item.get("cat3", "")
            # 한글 카테고리명으로 변환 (LLM이 정확하게 이해하도록)
            category_name = _get_category_name(content_type, cat3)

            results_summary.append({
                "title": item.get("title", ""),
                "addr": item.get("addr1", ""),
                "type": content_type,
                "cat3": cat3,
                "category": category_name,  # 한글 카테고리명 추가!
                "image": item.get("firstimage", ""),
                "mapx": item.get("mapx", ""),  # 경도
                "mapy": item.get("mapy", ""),  # 위도
                "tel": item.get("tel", ""),
                "content_id": item.get("contentid", "")
            })

    print(f"[CURATE] Total balanced results: {len(results_summary)} places")

    if not results_summary:
        return {
            "spots": [],
            "course": None,
            "message": "요청하신 조건에 맞는 장소를 찾지 못했습니다."
        }

    # 🔴 Step 1: 지리적 클러스터링 - 너무 멀리 떨어진 장소 필터링 (반경 10km)
    results_summary = filter_by_geographic_cluster(results_summary, max_radius_km=10.0)
    print(f"[CURATE] After geographic clustering: {len(results_summary)} places")

    # 🔴 Step 2: 동선 최적화 - Greedy TSP로 가까운 순서 정렬
    results_summary = optimize_route_order(results_summary)
    print(f"[CURATE] Route optimized")

    # 🔴 Step 3: 거리 계산하여 nearby 정보 추가 (LLM 참고용)
    results_summary = calculate_nearby_places(results_summary)
    print(f"[CURATE] Added nearby info to {len(results_summary)} places")

    # 🔴 사용자 요청에서 카테고리 순서 추출
    user_categories = []
    query_lower = query.lower()

    # 순서대로 매칭 (쿼리에서 등장하는 순서대로)
    category_keywords = [
        ("카페", ["카페", "커피", "디저트"]),
        ("관광지", ["바다", "바닷가", "해변", "관광", "구경", "산책", "공원"]),
        ("치킨", ["치킨", "통닭"]),
        ("횟집", ["횟집", "회", "해산물", "생선"]),
        ("고깃집", ["고기", "고깃집", "삼겹살", "갈비", "소고기", "돼지"]),
        ("돈까스", ["돈까스", "돈가스", "까스"]),
        ("일식", ["일식", "초밥", "라멘", "스시"]),
        ("한식", ["한식", "한정식", "백반"]),
        ("중식", ["중식", "중국", "짜장", "짬뽕"]),
        ("양식", ["양식", "파스타", "스테이크", "피자"]),
    ]

    # 쿼리에서 각 카테고리의 첫 등장 위치 찾기
    category_positions = []
    for cat_name, keywords in category_keywords:
        for kw in keywords:
            pos = query_lower.find(kw)
            if pos >= 0:
                category_positions.append((pos, cat_name))
                break

    # 등장 순서대로 정렬
    category_positions.sort(key=lambda x: x[0])
    user_categories = [cat for _, cat in category_positions]

    user_order_text = ""
    if user_categories:
        user_order_text = f"""
## 🔴🔴🔴 사용자가 요청한 순서 (이 순서대로 코스 구성 필수!):
{' → '.join(user_categories)}

**위 순서를 반드시 지켜서 코스를 구성하세요!**
- 각 카테고리에서 최소 1개 이상 선택
- category 필드를 확인해서 올바른 장소 선택
"""
        print(f"[CURATE] User requested order: {' → '.join(user_categories)}")

    prompt = f"""당신은 코레일 동행열차 여행 큐레이터입니다.

## 서비스 컨텍스트:
- 대상: **커플 여행객** (코레일 동행열차 서비스)
- 목적: 관광/데이트
- 분위기: 로맨틱하고 특별한 추억 만들기
{user_order_text}
## 사용자 요청:
{query}

## 검색된 장소들 (총 {len(results_summary)}개):
**nearby 필드는 해당 장소에서 5km 이내 가까운 장소들입니다. 동선 구성에 활용하세요!**
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

## 응답 형식 (JSON만 출력, 설명 없이):
{{
  "course": {{
    "title": "코스 제목 (예: 강릉 바다향 데이트 코스)",
    "stops": [
      {{
        "order": 1,
        "name": "장소명",
        "address": "주소",
        "mapx": "경도값",
        "mapy": "위도값",
        "content_id": "콘텐츠ID",
        "category": "카페/음식점/관광지/숙박",
        "time": "오전 10시",
        "duration": "1시간",
        "travel_time_to_next": "다음 장소까지 약 10분",
        "reason": "커플에게 추천하는 이유",
        "tip": "방문 팁"
      }}
    ],
    "total_duration": "약 6시간",
    "summary": "코스 요약 (2-3문장, 커플 여행 관점)"
  }}
}}

## 이동시간 계산 규칙 (travel_time_to_next):
- nearby 필드의 거리 정보를 활용하세요
- 거리 기준 예상 이동시간: **1km당 약 3분** (차량 기준)
  - 1km → 약 3분
  - 2km → 약 6분
  - 5km → 약 15분
  - 10km → 약 30분
- 마지막 정차지는 travel_time_to_next 생략 (null)

## 규칙:
- **사용자가 요청한 순서대로** 코스를 구성하세요!
  - 예: "카페 → 점심 일식 → 저녁 고기" 요청 시 → 카페 먼저, 일식집, 고기집 순서로!
- 3~6개 장소를 **사용자 요청 순서 + 동선 고려**하여 선정
- **nearby 필드를 활용**해서 가까운 장소끼리 묶어서 동선 최적화!
  - 예: A장소의 nearby에 B장소가 있으면 A→B 순서가 이동 효율적
- **커플 데이트 관점**에서 추천 이유 작성
- mapx, mapy 값이 있는 장소 우선 선택 (지도 연동용)
- content_id 반드시 포함 (상세정보 조회용)
- 중복/비슷한 장소 제외
- 반드시 유효한 JSON만 출력

## 🔴🔴🔴 절대 규칙 - 반드시 준수:
1. **사용자가 원하는 순서대로 코스 구성!** (카페→일식→고기 요청시 순서 지키기)
2. **검색된 장소 목록에 있는 장소만 선택하세요!**
3. **새로운 장소를 임의로 만들지 마세요!** (예: "해변 산책", "카페 방문" 등 임의 추가 금지)
4. **content_id가 없으면 그 장소는 사용할 수 없습니다**

## 🔴 매우 중요 - 정확한 정보 사용:
- **category 필드를 그대로 사용하세요** (추측하지 마세요!)
  - "한식" → 한식 음식점
  - "일식" → 일식 음식점 (초밥, 라멘 등)
  - "서양식" → 서양 음식점 (돈까스, 파스타, 스테이크 등)
  - "카페" → 카페/디저트
- 장소 이름만 보고 음식 종류를 **추측하지 마세요**
- 예: "황태전파는집"은 category가 "한식"이면 황태 전문 한식당입니다 (고깃집 아님!)"""

    messages = [{"role": "user", "content": prompt}]
    response = generate_response(messages, max_tokens=1500, temperature=0.5)

    print(f"[CURATE DEBUG] LLM response length: {len(response)}")
    print(f"[CURATE DEBUG] LLM response preview: {response[:500]}...")

    # JSON 파싱 - bracket counting 방식 (select_tools_with_llm과 동일)
    curated_course = None
    try:
        json_start = response.find("{")
        if json_start < 0:
            print("[CURATE DEBUG] No JSON object found in response!")
        else:
            # Bracket counting으로 첫 번째 완전한 JSON 객체 찾기
            bracket_count = 0
            json_end = -1
            in_string = False
            escape_next = False

            for i, char in enumerate(response[json_start:], start=json_start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i + 1
                        break

            print(f"[CURATE DEBUG] JSON range: {json_start} to {json_end}")

            if json_end > json_start:
                json_str = response[json_start:json_end]
                # 주석 제거 및 trailing comma 수정
                json_str_clean = re.sub(r'//[^\n]*', '', json_str)
                json_str_clean = re.sub(r',(\s*[}\]])', r'\1', json_str_clean)

                parsed = json.loads(json_str_clean)
                curated_course = parsed.get("course")
                print(f"[CURATE DEBUG] Successfully parsed course: {curated_course is not None}")
            else:
                print("[CURATE DEBUG] Could not find matching closing bracket!")
    except json.JSONDecodeError as e:
        print(f"[CURATE DEBUG] JSON parsing error: {e}")
        print(f"[CURATE DEBUG] Failed JSON string: {json_str[:500] if 'json_str' in locals() else 'N/A'}...")

    # spots 리스트 생성 (전체 검색 결과, 좌표 포함)
    spots = []
    for r in results_summary:
        spots.append({
            "name": r["title"],
            "address": r["addr"],
            "category": _get_category_name(r["type"], r.get("cat3")),  # cat3로 카페/음식점 구분
            "image_url": r["image"],
            "mapx": r["mapx"],
            "mapy": r["mapy"],
            "tel": r["tel"],
            "content_id": r["content_id"]
        })

    # 🔴 코스 거리 검증 및 실제 거리 추가
    if curated_course and "stops" in curated_course:
        curated_course = _add_actual_distances_to_course(curated_course)

    return {
        "spots": spots,  # 리스트 뷰용 (전체)
        "course": curated_course,  # 코스 뷰용 (LLM 큐레이션)
        "message": f"{len(spots)}개의 장소를 찾았습니다."
    }


def _add_actual_distances_to_course(course: dict) -> dict:
    """
    코스의 각 정차지 간 실제 거리를 계산하여 추가
    LLM이 추정한 travel_time_to_next와 별개로 실제 거리 제공
    """
    stops = course.get("stops", [])
    if len(stops) < 2:
        return course

    total_distance = 0.0

    for i in range(len(stops)):
        stop = stops[i]

        if i < len(stops) - 1:
            next_stop = stops[i + 1]
            try:
                lat1 = float(stop.get("mapy", 0))
                lon1 = float(stop.get("mapx", 0))
                lat2 = float(next_stop.get("mapy", 0))
                lon2 = float(next_stop.get("mapx", 0))

                if lat1 != 0 and lon1 != 0 and lat2 != 0 and lon2 != 0:
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    stop["distance_to_next_km"] = round(dist, 1)
                    total_distance += dist
                    print(f"[DISTANCE] {stop.get('name', '')} → {next_stop.get('name', '')}: {dist:.1f}km")
                else:
                    stop["distance_to_next_km"] = None
            except:
                stop["distance_to_next_km"] = None
        else:
            # 마지막 정차지
            stop["distance_to_next_km"] = None

    course["total_distance_km"] = round(total_distance, 1)
    print(f"[DISTANCE] Total course distance: {total_distance:.1f}km")

    return course


def _get_category_name(content_type_id: str, cat3: str = None) -> str:
    """content_type_id + cat3를 카테고리명으로 변환

    cat3 코드 (음식점 세부 분류):
    - A05020100: 한식
    - A05020200: 서양식 (돈까스, 파스타, 스테이크 등)
    - A05020300: 일식 (초밥, 라멘 등)
    - A05020400: 중식
    - A05020500: 아시아음식
    - A05020600: 패밀리레스토랑
    - A05020700: 이색음식점
    - A05020800: 패스트푸드
    - A05020900: 카페/전통찻집
    """
    # 음식점(39)인 경우 cat3로 세부 분류
    if content_type_id == "39" and cat3:
        cat3_map = {
            "A05020100": "한식",
            "A05020200": "서양식",
            "A05020300": "일식",
            "A05020400": "중식",
            "A05020500": "아시아음식",
            "A05020600": "패밀리레스토랑",
            "A05020700": "이색음식점",
            "A05020800": "패스트푸드",
            "A05020900": "카페",
        }
        if cat3 in cat3_map:
            return cat3_map[cat3]

    type_map = {
        "12": "관광지",
        "14": "문화시설",
        "15": "축제/행사",
        "25": "여행코스",
        "28": "레포츠",
        "32": "숙박",
        "38": "쇼핑",
        "39": "음식점"
    }
    return type_map.get(content_type_id, "기타")


# ========== API 엔드포인트 ==========
@app.get("/health")
async def health():
    return {"status": "ok", "mcp_enabled": True}

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "object": "model"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """기존 OpenAI 호환 엔드포인트"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    response_text = generate_response(messages, request.max_tokens, request.temperature)

    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }


@app.post("/v1/mcp/query")
async def mcp_query(request: MCPQueryRequest):
    """
    MCP Host 엔드포인트 (오케스트레이션 버전)
    자연어 쿼리 → 쿼리 분석 → 폴백 전략 검색 → 결과 큐레이션

    응답 구조:
    - spots: 리스트 뷰용 (전체 검색 결과, 좌표 포함)
    - course: 코스 뷰용 (LLM이 큐레이션한 동선)
    """
    request_id = f"mcp_{int(time.time() * 1000)}"
    start_time = time.time()

    query = request.query
    area_code = request.area_code
    sigungu_code = request.sigungu_code

    logger.info("=" * 70)
    logger.info(f"[{request_id}] /v1/mcp/query 요청 시작")
    logger.info(f"[{request_id}] 시간: {datetime.now().isoformat()}")
    logger.info(f"[{request_id}] 쿼리: {query}")
    logger.info(f"[{request_id}] area_code: {area_code}, sigungu_code: {sigungu_code}")
    logger.info("=" * 70)

    # area_code가 없으면 기존 LLM 기반 방식 사용
    if not area_code:
        logger.info(f"[{request_id}] area_code 없음 → LLM 기반 도구 선택 모드")
        selected_tools = select_tools_with_llm(query, area_code, sigungu_code)

        if not selected_tools:
            elapsed = time.time() - start_time
            logger.warning(f"[{request_id}] 도구 선택 실패 (소요시간: {elapsed:.2f}초)")
            return {
                "success": False,
                "error": "적절한 도구를 찾지 못했습니다.",
                "query": query,
                "spots": [],
                "course": None
            }

        tool_results = []
        for tool in selected_tools:
            logger.debug(f"[{request_id}] MCP 도구 호출: {tool.get('name')}")
            result = await call_mcp_tool(tool.get("name"), tool.get("arguments", {}))
            tool_results.append({"result": result})

        curated = curate_results_with_llm(query, [r["result"] for r in tool_results])
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] LLM 기반 모드 완료 (소요시간: {elapsed:.2f}초)")
        return {
            "success": True,
            "query": query,
            "spots": curated.get("spots", []),
            "course": curated.get("course"),
            "message": curated.get("message", ""),
            "search_mode": "llm_based"
        }

    # ========== 오케스트레이션 모드 ==========
    logger.info(f"[{request_id}] 오케스트레이션 모드 시작")

    # Phase 1: 쿼리 분석 (규칙 기반 - 빠름)
    phase1_start = time.time()
    needs = analyze_query_needs(query)
    phase1_elapsed = time.time() - phase1_start
    logger.info(f"[{request_id}] [Phase 1] 쿼리 분석 완료 (소요시간: {phase1_elapsed:.3f}초)")
    logger.info(f"[{request_id}]   - needs: {list(needs.keys())}")
    if "user_order" in needs:
        logger.info(f"[{request_id}]   - user_order: {[cat for cat, _ in needs['user_order']]}")

    # Phase 2: 오케스트레이션 검색 (폴백 전략 포함)
    phase2_start = time.time()
    search_result = await orchestrated_search(query, area_code, sigungu_code, needs)
    phase2_elapsed = time.time() - phase2_start
    logger.info(f"[{request_id}] [Phase 2] 검색 완료 (소요시간: {phase2_elapsed:.2f}초)")
    logger.info(f"[{request_id}]   - 검색 결과: {search_result.get('totalCount')} items")
    logger.info(f"[{request_id}]   - 검색 로그: {search_result.get('search_log')}")

    # Phase 3: 결과 검증 - 최소 기준 미달시 추가 검색
    if search_result.get("totalCount", 0) < MIN_RESULTS_THRESHOLD:
        logger.warning(f"[{request_id}] [Phase 3] 결과 부족 ({search_result.get('totalCount')} < {MIN_RESULTS_THRESHOLD}), 광역 검색 수행")
        phase3_start = time.time()
        broad_result = await search_by_area_direct(area_code, sigungu_code, None, num_rows=50)
        if broad_result.get("items"):
            existing_ids = {i.get("contentid") for i in search_result.get("items", [])}
            added_count = 0
            for item in broad_result["items"]:
                if item.get("contentid") not in existing_ids:
                    search_result["items"].append(item)
                    added_count += 1
            search_result["totalCount"] = len(search_result["items"])
            search_result["search_log"].append(f"broad_fallback→{len(broad_result['items'])}개")
            logger.info(f"[{request_id}]   - 광역 검색으로 {added_count}개 추가")
        phase3_elapsed = time.time() - phase3_start
        logger.info(f"[{request_id}] [Phase 3] 광역 검색 완료 (소요시간: {phase3_elapsed:.2f}초)")

    # Phase 4: LLM 큐레이션 (코스 생성) - user_order 전달
    phase4_start = time.time()
    user_order = search_result.get("user_order", [])
    logger.info(f"[{request_id}] [Phase 4] LLM 큐레이션 시작 (입력 items: {len(search_result.get('items', []))}개)")
    curated = curate_results_with_llm(query, [search_result], user_order=user_order)
    phase4_elapsed = time.time() - phase4_start
    logger.info(f"[{request_id}] [Phase 4] LLM 큐레이션 완료 (소요시간: {phase4_elapsed:.2f}초)")
    logger.info(f"[{request_id}]   - spots: {len(curated.get('spots', []))}개")
    logger.info(f"[{request_id}]   - course: {'있음' if curated.get('course') else '없음'}")
    if curated.get("course"):
        course = curated["course"]
        logger.info(f"[{request_id}]   - course.title: {course.get('title')}")
        logger.info(f"[{request_id}]   - course.stops: {len(course.get('stops', []))}개")
        logger.info(f"[{request_id}]   - course.total_distance_km: {course.get('total_distance_km')}")

    # 최종 응답
    total_elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"[{request_id}] /v1/mcp/query 요청 완료")
    logger.info(f"[{request_id}] 총 소요시간: {total_elapsed:.2f}초")
    logger.info(f"[{request_id}]   - Phase 1 (분석): {phase1_elapsed:.3f}초")
    logger.info(f"[{request_id}]   - Phase 2 (검색): {phase2_elapsed:.2f}초")
    logger.info(f"[{request_id}]   - Phase 4 (큐레이션): {phase4_elapsed:.2f}초")
    logger.info("=" * 70)

    return {
        "success": True,
        "query": query,
        "area_code": area_code,
        "sigungu_code": sigungu_code,
        "spots": curated.get("spots", []),
        "course": curated.get("course"),
        "message": curated.get("message", ""),
        "search_mode": "orchestrated",
        "search_log": search_result.get("search_log", []),
        "needs_analyzed": search_result.get("needs_analyzed", [])
    }


@app.get("/v1/mcp/tools")
async def list_mcp_tools():
    """사용 가능한 MCP 도구 목록"""
    return {"tools": MCP_TOOLS}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
