"""
EXAONE-3.5-32B Server with MCP Host functionality
LLM이 MCP 도구를 선택하고 호출하는 기능 포함
"""
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
from typing import Optional

app = FastAPI(title="EXAONE-3.5-32B Server + MCP Host")

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
    specific_food_keywords = [
        "돈까스", "돈가스", "삼겹살", "치킨", "피자", "파스타", "스테이크",
        "초밥", "회", "라멘", "우동", "냉면", "막국수", "칼국수", "짜장면", "짬뽕",
        "떡볶이", "순대", "김밥", "비빔밥", "불고기", "갈비", "삼계탕", "설렁탕",
        "순두부", "부대찌개", "감자탕", "곱창", "족발", "보쌈", "치즈", "버거", "햄버거",
        "아이스크림", "빙수", "와플", "마카롱", "케이크"
    ]
    specific_matches = [kw for kw in specific_food_keywords if kw in query_lower]
    if specific_matches:
        needs["food_specific"] = specific_matches  # 직접 검색용

    # 음식 관련 일반 키워드
    food_keywords = ["맛집", "음식", "밥", "식당", "먹", "점심", "저녁", "아침",
                     "한식", "중식", "일식", "양식", "분식", "고기", "해산물"]
    food_matches = [kw for kw in food_keywords if kw in query_lower]
    if food_matches or specific_matches:
        needs["food"] = food_matches + specific_matches

    # 카페 관련 키워드
    cafe_keywords = ["카페", "커피", "디저트", "빵", "베이커리", "브런치", "차", "음료"]
    cafe_matches = [kw for kw in cafe_keywords if kw in query_lower]
    if cafe_matches:
        needs["cafe"] = cafe_matches

    # 관광지 관련 키워드
    spot_keywords = ["관광", "명소", "볼거리", "구경", "바다", "산", "공원", "해변", "전망", "야경",
                     "사진", "인스타", "데이트", "드라이브", "자연", "풍경", "경치"]
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

    print(f"[ORCH] Analyzed needs: {needs}")
    return needs


async def orchestrated_search(query: str, area_code: str, sigungu_code: str, needs: dict) -> dict:
    """
    오케스트레이션된 검색 - 폴백 전략 포함

    전략:
    0. 구체적인 음식 키워드가 있으면 최우선 검색 (돈까스, 피자 등)
    1. 키워드 검색 시도 (매칭된 키워드로)
    2. 결과 부족시 → 카테고리 기반 검색
    3. 여전히 부족시 → 지역 전체 검색
    """
    all_results = {}
    search_log = []

    # Strategy 0: 구체적인 음식 키워드 최우선 검색 (돈까스, 피자 등)
    if "food_specific" in needs:
        specific_results = {"items": []}
        for kw in needs["food_specific"]:
            print(f"[ORCH] Strategy 0: SPECIFIC food keyword search '{kw}'")
            result = await search_by_keyword_direct(kw, area_code, sigungu_code, "39")  # 음식점
            items = result.get("items", [])
            search_log.append(f"specific:{kw}→{len(items)}개")
            if items:
                specific_results["items"].extend(items)

        if specific_results["items"]:
            all_results["food_specific"] = specific_results
            print(f"[ORCH] Found {len(specific_results['items'])} specific food items!")

    for need_type, keywords in needs.items():
        # food_specific은 이미 처리됨
        if need_type == "food_specific":
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

        # Strategy 3: 지역 전체 검색 (여전히 부족시)
        if len(results_for_need.get("items", [])) < MIN_RESULTS_THRESHOLD:
            print(f"[ORCH] Strategy 3: area search without content_type")
            result = await search_by_area_direct(area_code, sigungu_code, None, num_rows=30)
            items = result.get("items", [])
            search_log.append(f"area_only→{len(items)}개")

            if items:
                existing_ids = {i.get("contentid") for i in results_for_need.get("items", [])}
                for item in items:
                    if item.get("contentid") not in existing_ids:
                        results_for_need["items"].append(item)

        all_results[need_type] = results_for_need
        print(f"[ORCH] {need_type}: {len(results_for_need.get('items', []))} items collected")

    # 결과 합치기 (food_specific 우선)
    combined_items = []
    seen_ids = set()

    # 1. 구체적인 음식 검색 결과 먼저 추가 (돈까스 검색했으면 돈까스집 먼저)
    if "food_specific" in all_results:
        for item in all_results["food_specific"].get("items", []):
            cid = item.get("contentid")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                combined_items.append(item)
        print(f"[ORCH] Added {len(combined_items)} specific food items first")

    # 2. 나머지 결과 추가
    for need_type, result in all_results.items():
        if need_type == "food_specific":
            continue  # 이미 처리됨
        for item in result.get("items", []):
            cid = item.get("contentid")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                combined_items.append(item)

    return {
        "items": combined_items,
        "totalCount": len(combined_items),
        "search_log": search_log,
        "needs_analyzed": list(needs.keys())
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


def curate_results_with_llm(query: str, tool_results: list[dict]) -> dict:
    """LLM을 사용해 검색 결과를 큐레이션 - spots(리스트뷰) + course(코스뷰) 분리"""

    # 결과 요약 (좌표 정보 + cat3 포함)
    results_summary = []
    for result in tool_results:
        if "items" in result and result["items"]:
            for item in result["items"][:15]:
                results_summary.append({
                    "title": item.get("title", ""),
                    "addr": item.get("addr1", ""),
                    "type": item.get("contenttypeid", ""),
                    "cat3": item.get("cat3", ""),  # 세부 카테고리 (카페 구분용)
                    "image": item.get("firstimage", ""),
                    "mapx": item.get("mapx", ""),  # 경도
                    "mapy": item.get("mapy", ""),  # 위도
                    "tel": item.get("tel", ""),
                    "content_id": item.get("contentid", "")
                })

    if not results_summary:
        return {
            "spots": [],
            "course": None,
            "message": "요청하신 조건에 맞는 장소를 찾지 못했습니다."
        }

    prompt = f"""당신은 코레일 동행열차 여행 큐레이터입니다.

## 서비스 컨텍스트:
- 대상: **커플 여행객** (코레일 동행열차 서비스)
- 목적: 관광/데이트
- 분위기: 로맨틱하고 특별한 추억 만들기

## 사용자 요청:
{query}

## 검색된 장소들 (총 {len(results_summary)}개):
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
        "reason": "커플에게 추천하는 이유",
        "tip": "방문 팁"
      }}
    ],
    "total_duration": "약 6시간",
    "summary": "코스 요약 (2-3문장, 커플 여행 관점)"
  }}
}}

## 규칙:
- 사용자 요청에 맞게 3~6개 장소를 **동선 순서대로** 선정
- **커플 데이트 관점**에서 추천 이유 작성
- mapx, mapy 값이 있는 장소 우선 선택 (지도 연동용)
- content_id 반드시 포함 (상세정보 조회용)
- 중복/비슷한 장소 제외
- 반드시 유효한 JSON만 출력"""

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

    return {
        "spots": spots,  # 리스트 뷰용 (전체)
        "course": curated_course,  # 코스 뷰용 (LLM 큐레이션)
        "message": f"{len(spots)}개의 장소를 찾았습니다."
    }


def _get_category_name(content_type_id: str, cat3: str = None) -> str:
    """content_type_id + cat3를 카테고리명으로 변환

    cat3 코드 (음식점 세부 분류):
    - A05020900: 카페/전통찻집
    - A05020100: 한식
    - A05020200: 서양식 (돈까스, 파스타 등)
    - A05020300: 일식
    - A05020400: 중식
    - A05020700: 이색음식점
    """
    # 음식점(39)인 경우 cat3로 카페 구분
    if content_type_id == "39" and cat3:
        if cat3 == "A05020900":
            return "카페"

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
    query = request.query
    area_code = request.area_code
    sigungu_code = request.sigungu_code

    print(f"[MCP-ORCH] Query: {query}, area_code: {area_code}, sigungu_code: {sigungu_code}")

    # area_code가 없으면 기존 LLM 기반 방식 사용
    if not area_code:
        print("[MCP-ORCH] No area_code, falling back to LLM-based tool selection")
        selected_tools = select_tools_with_llm(query, area_code, sigungu_code)

        if not selected_tools:
            return {
                "success": False,
                "error": "적절한 도구를 찾지 못했습니다.",
                "query": query,
                "spots": [],
                "course": None
            }

        tool_results = []
        for tool in selected_tools:
            result = await call_mcp_tool(tool.get("name"), tool.get("arguments", {}))
            tool_results.append({"result": result})

        curated = curate_results_with_llm(query, [r["result"] for r in tool_results])
        return {
            "success": True,
            "query": query,
            "spots": curated.get("spots", []),
            "course": curated.get("course"),
            "message": curated.get("message", ""),
            "search_mode": "llm_based"
        }

    # ========== 오케스트레이션 모드 ==========
    print("[MCP-ORCH] Using orchestrated search with fallback strategies")

    # Phase 1: 쿼리 분석 (규칙 기반 - 빠름)
    needs = analyze_query_needs(query)
    print(f"[MCP-ORCH] Needs: {needs}")

    # Phase 2: 오케스트레이션 검색 (폴백 전략 포함)
    search_result = await orchestrated_search(query, area_code, sigungu_code, needs)
    print(f"[MCP-ORCH] Search result: {search_result.get('totalCount')} items")
    print(f"[MCP-ORCH] Search log: {search_result.get('search_log')}")

    # Phase 3: 결과 검증 - 최소 기준 미달시 추가 검색
    if search_result.get("totalCount", 0) < MIN_RESULTS_THRESHOLD:
        print("[MCP-ORCH] Results below threshold, doing broad search")
        # 최후의 수단: 지역 전체 검색
        broad_result = await search_by_area_direct(area_code, sigungu_code, None, num_rows=50)
        if broad_result.get("items"):
            existing_ids = {i.get("contentid") for i in search_result.get("items", [])}
            for item in broad_result["items"]:
                if item.get("contentid") not in existing_ids:
                    search_result["items"].append(item)
            search_result["totalCount"] = len(search_result["items"])
            search_result["search_log"].append(f"broad_fallback→{len(broad_result['items'])}개")

    # Phase 4: LLM 큐레이션 (코스 생성)
    curated = curate_results_with_llm(query, [search_result])

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
