"""高德地图工具 - 基于 LangChain 实现"""

import os
import requests
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool


def get_amap_api_key() -> str:
    """获取高德地图 API 密钥"""
    from ..config import get_settings
    settings = get_settings()
    return settings.amap_api_key


@tool
def maps_text_search(keywords: str, city: str, types: str = "") -> str:
    """
    使用高德地图搜索地点信息（景点、酒店、餐厅等）

    Args:
        keywords: 搜索关键词，如"景点"、"酒店"、"餐厅"
        city: 城市名称，如"北京"、"上海"
        types: 地点类型（可选），如"旅游景点"、"宾馆"

    Returns:
        JSON 格式的搜索结果
    """
    api_key = get_amap_api_key()
    
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": api_key,
        "keywords": keywords,
        "city": city,
        "types": types,
        "offset": 10,
        "page": 1,
        "extensions": "all"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1":
            pois = data.get("pois", [])
            results = []
            for poi in pois[:10]:
                result = {
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "type": poi.get("type", ""),
                    "tel": poi.get("tel", ""),
                    "location": poi.get("location", ""),
                    "rating": poi.get("biz_ext", {}).get("rating", ""),
                    "cost": poi.get("biz_ext", {}).get("cost", "")
                }
                
                # 解析经纬度
                location_str = result["location"]
                if location_str and "," in location_str:
                    lng, lat = location_str.split(",")
                    result["longitude"] = float(lng)
                    result["latitude"] = float(lat)
                else:
                    result["longitude"] = 0.0
                    result["latitude"] = 0.0
                
                results.append(result)
            
            return f"找到 {len(results)} 个相关地点:\n" + "\n".join(
                f"{i+1}. {r['name']} - {r['address']} (类型：{r['type']}, 评分：{r['rating']})"
                for i, r in enumerate(results)
            )
        else:
            return f"搜索失败：{data.get('info', '未知错误')}"
    
    except Exception as e:
        return f"搜索出错：{str(e)}"


@tool
def maps_weather(city: str) -> str:
    """
    查询城市天气信息

    Args:
        city: 城市名称，如"北京"、"上海"

    Returns:
        天气信息文本
    """
    api_key = get_amap_api_key()
    
    # 首先获取城市的 adcode
    geo_url = "https://restapi.amap.com/v3/config/district"
    geo_params = {
        "key": api_key,
        "keywords": city,
        "subdistrict": 0
    }
    
    try:
        geo_response = requests.get(geo_url, params=geo_params, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if geo_data.get("status") != "1" or not geo_data.get("districts"):
            return f"未找到城市：{city}"
        
        adcode = geo_data["districts"][0].get("adcode", "")
        
        # 查询天气
        weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        weather_params = {
            "key": api_key,
            "city": adcode,
            "extensions": "all"
        }
        
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        if weather_data.get("status") == "1":
            forecasts = weather_data.get("forecasts", [])
            if forecasts:
                forecast = forecasts[0]
                casts = forecast.get("casts", [])
                
                weather_info = f"{city}天气预报:\n"
                for i, cast in enumerate(casts[:3]):
                    weather_info += (
                        f"{cast.get('date', '')}: "
                        f"{cast.get('dayweather', '')}转{cast.get('nightweather', '')}, "
                        f"温度{cast.get('nighttemp', '')}-{cast.get('daytemp', '')}℃, "
                        f"风向：{cast.get('daywind', '')}, 风力：{cast.get('daypower', '')}级\n"
                    )
                
                # 添加实时天气
                lives = weather_data.get("lives", [])
                if lives:
                    live = lives[0]
                    weather_info += (
                        f"\n实时天气：{live.get('weather', '')}, "
                        f"温度：{live.get('temperature', '')}℃, "
                        f"湿度：{live.get('humidity', '')}%, "
                        f"风向：{live.get('winddirection', '')}, 风力：{live.get('windpower', '')}级"
                    )
                
                return weather_info
            else:
                return "未找到天气信息"
        else:
            return f"天气查询失败：{weather_data.get('info', '未知错误')}"
    
    except Exception as e:
        return f"天气查询出错：{str(e)}"


@tool
def maps_direction_transit(origin: str, destination: str, city: str = "") -> str:
    """
    查询公交/地铁路线规划（支持跨城）

    Args:
        origin: 起点坐标，格式：经度，纬度 (如：116.397128,39.916527)
        destination: 终点坐标，格式：经度，纬度
        city: 城市名称或 adcode（同城查询时使用，如"北京"，跨城可不填）

    Returns:
        公交路线规划信息
    """
    api_key = get_amap_api_key()

    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    params = {
        "key": api_key,
        "origin": origin,
        "destination": destination,
        "city": city,
        "strategy": 0,
        "extensions": "all",
        "output": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "1":
            route = data.get("route", {})
            transits = route.get("transits", [])

            if not transits:
                return "未找到公交路线"

            result_parts = []
            for i, transit in enumerate(transits[:3]):
                total_duration = int(transit.get("duration", 0) or 0)
                total_price = float(transit.get("cost", 0) or 0)
                walking_distance = int(transit.get("walking_distance", 0) or 0)
                segments = transit.get("segments", [])

                result_parts.append(f"\n方案{i+1}（预计{total_duration//60}分钟，费用{total_price:.0f}元，步行{walking_distance}米）:")

                for seg in segments:
                    if "bus" in seg and seg["bus"].get("buslines"):
                        bus = seg["bus"]
                        busline = bus["buslines"][0]
                        line_name = busline.get("name", "未知线路")
                        dep_stop = busline.get("departure_stop", "")
                        arr_stop = busline.get("arrival_stop", "")
                        if isinstance(dep_stop, dict):
                            dep_stop = dep_stop.get("name", "")
                        if isinstance(arr_stop, dict):
                            arr_stop = arr_stop.get("name", "")
                        via_num = busline.get("via_num", 0)
                        if isinstance(via_num, str):
                            via_num = int(via_num) if via_num.isdigit() else 0
                        line_type = busline.get("bus_type", "") or busline.get("type", "")
                        result_parts.append(f"   🚌 {line_name}")
                        if line_type:
                            result_parts.append(f"      {line_type}")
                        result_parts.append(f"      上车: {dep_stop} → 下车: {arr_stop}（{via_num}站）")
                    elif "walking" in seg:
                        walk = seg["walking"]
                        if isinstance(walk, dict):
                            walk_dist = int(walk.get("distance", 0) or 0)
                            walk_dur = int(walk.get("duration", 0) or 0)
                            if walk_dist > 0:
                                result_parts.append(f"   🚶 步行{walk_dist}米（约{walk_dur//60}分钟）")
                    elif "railway" in seg:
                        railway = seg["railway"]
                        if isinstance(railway, dict):
                            result_parts.append(f"   🚄 {railway.get('name', '铁路')}")

            if len(transits) > 3:
                result_parts.append(f"\n... 还有 {len(transits) - 3} 个方案未显示")

            return "\n".join(result_parts)
        else:
            return f"公交路线查询失败：{data.get('info', '未知错误')}"

    except Exception as e:
        return f"公交路线查询出错：{str(e)}"


@tool
def maps_direction_driving(origin: str, destination: str) -> str:
    """
    查询驾车路线规划

    Args:
        origin: 起点坐标，格式：经度，纬度 (如：116.397128,39.916527)
        destination: 终点坐标，格式：经度，纬度

    Returns:
        路线规划信息
    """
    api_key = get_amap_api_key()
    
    url = "https://restapi.amap.com/v3/direction/driving"
    params = {
        "key": api_key,
        "origin": origin,
        "destination": destination,
        "extensions": "all"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1":
            route = data.get("route", {})
            paths = route.get("paths", [])
            
            if paths:
                path = paths[0]
                distance = int(path.get("distance", 0))
                duration = int(path.get("duration", 0))
                
                return f"距离：{distance/1000:.1f}公里，预计时间：{duration/60:.0f}分钟"
            else:
                return "未找到路线"
        else:
            return f"路线查询失败：{data.get('info', '未知错误')}"
    
    except Exception as e:
        return f"路线查询出错：{str(e)}"


def get_amap_tools() -> List:
    """获取所有高德地图工具"""
    return [
        maps_text_search,
        maps_weather,
        maps_direction_driving,
        maps_direction_transit,
    ]
