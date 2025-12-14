# This pipeline is slow as heck even on a 4070 use the Gemini Api version instead
import pandas as pd
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "ministral-3:3b"  # Using 3B for 8GB VRAM (laptop GPU)

#Change this to your path
INPUT_FILE = r"C:\Users\paeki\OneDrive\Desktop\pun\2110403-DSDE-M150-Lover\Scraper_new\Python Pipeline\MEA scraped\mea_power_outages_page_001.csv"
OUTPUT_FILE = "mea_power_outages_page_001.csv"

# ==========================================
# 2. DEFINE DATA STRUCTURE
# ==========================================
class OutageEvent(BaseModel):
    start_time_24h: str = Field(..., description="Start time in HH:MM (24h) format.")
    end_time_24h: str = Field(..., description="End time in HH:MM (24h) format.")
    location_detail: str = Field(..., description="Specific street, soi, or village.")
    district: str = Field(
        default="Unknown",
        description="IMPORTANT: Extract the District name if explicitly mentioned. "
                    "Examples: 'Pathumwan', 'Bang Kapi', 'Din Daeng', 'Vadhana'. "
                    "If the text only mentions a road name (like 'Lat Phrao Road') without a district, return 'Unknown'. "
                    "Only return a district if it's clearly stated in the text."
    )
    province: str = Field(
        default="Bangkok",
        description="Province name. Default to 'Bangkok' for Bangkok addresses."
    )

class DailyOutageSchedule(BaseModel):
    events: List[OutageEvent]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_events_with_local_llm(row_text, date_context):
    """Sends text to local LLM (via Ollama) to extract structured JSON."""
    if not isinstance(row_text, str) or not row_text.strip():
        return []

    # Configure local OpenAI-compatible client
    local_client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"  # Ollama doesn't need a real API key
    )

    print(f"  [DEBUG] Using local model: {LOCAL_MODEL} for date: {date_context}")

    # Create instructor client with OpenAI mode
    client = instructor.from_openai(
        local_client,
        mode=instructor.Mode.JSON
    )

    try:
        print(f"  [DEBUG] Sending request to local LLM...")
        resp = client.chat.completions.create(
            model=LOCAL_MODEL,
            response_model=DailyOutageSchedule,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured data from power outage announcements. "
                               "The text is in English. Extract time periods, specific street/soi names, and district if mentioned. "
                               "Common Bangkok districts: Pathumwan, Bang Kapi, Din Daeng, Huai Khwang, Vadhana, Khlong Toei, Don Mueang, etc."
                },
                {
                    "role": "user",
                    "content": f"Date: {date_context}\n\nExtract all power outage events from this English text. "
                               f"For each event, extract the time range and location details:\n\n{row_text}"
                }
            ],
        )
        print(f"  [DEBUG] Received {len(resp.events)} events from local LLM")
        return resp.events
    except Exception as e:
        print(f"  [!] AI Error on {date_context}: {e}")
        import traceback
        print(f"  [DEBUG] Full traceback:\n{traceback.format_exc()}")
        return []

# Landmark/Street/Area to District mapping (English & Thai)
# This helps when the MEA message only mentions streets or areas, not districts
AREA_TO_DISTRICT = {
    # Major Roads (English)
    "Sukhumvit": "วัฒนา",          # Sukhumvit -> Vadhana
    "Rama 9": "ห้วยขวาง",          # Rama 9 -> Huai Khwang
    "Rama IX": "ห้วยขวาง",         # Rama IX -> Huai Khwang
    "Phetchaburi": "ราชเทวี",      # Phetchaburi -> Ratchathewi
    "Petchaburi": "ราชเทวี",       # Petchaburi -> Ratchathewi
    "Lat Phrao": "ลาดพร้าว",       # Lat Phrao -> Lat Phrao
    "Latphrao": "ลาดพร้าว",        # Latphrao -> Lat Phrao
    "Ramkhamhaeng": "บางกะปิ",     # Ramkhamhaeng -> Bang Kapi
    "Ram Khamhaeng": "บางกะปิ",    # Ram Khamhaeng -> Bang Kapi
    "Ratchada": "ดินแดง",          # Ratchada -> Din Daeng
    "Ratchadaphisek": "ดินแดง",    # Ratchadaphisek -> Din Daeng
    "Vibhavadi": "จตุจักร",        # Vibhavadi -> Chatuchak
    "Viphavadi": "จตุจักร",        # Viphavadi -> Chatuchak
    "Sathorn": "สาทร",             # Sathorn -> Sathon
    "Sathon": "สาทร",              # Sathon -> Sathon
    "Silom": "บางรัก",             # Silom -> Bang Rak
    "Nawamin": "บึงกุ่ม",          # Nawamin -> Bueng Kum

    # Major Roads (Thai)
    "สุขุมวิท": "วัฒนา",
    "พระราม 9": "ห้วยขวาง",
    "เพชรบุรี": "ราชเทวี",
    "ลาดพร้าว": "ลาดพร้าว",
    "รามคำแหง": "บางกะปิ",
    "รัชดา": "ดินแดง",
    "วิภาวดี": "จตุจักร",
    "สาทร": "สาทร",
    "สีลม": "บางรัก",

    # Famous Areas/Landmarks (English)
    "Siam": "ปทุมวัน",             # Siam -> Pathumwan
    "Chidlom": "ปทุมวัน",          # Chidlom -> Pathumwan
    "Asoke": "วัฒนา",              # Asoke -> Vadhana
    "Ekkamai": "วัฒนา",            # Ekkamai -> Vadhana
    "Thong Lo": "วัฒนา",           # Thong Lo -> Vadhana
    "Thonglor": "วัฒนา",           # Thonglor -> Vadhana
    "Ari": "พญาไท",                # Ari -> Phaya Thai
    "Huai Khwang": "ห้วยขวาง",     # Huai Khwang -> Huai Khwang
    "On Nut": "ประเวศ",            # On Nut -> Prawet
    "Onnut": "ประเวศ",             # Onnut -> Prawet
    "Bang Na": "ประเวศ",           # Bang Na -> Prawet
    "Bangna": "ประเวศ",            # Bangna -> Prawet
    "Lat Krabang": "ลาดกระบัง",    # Lat Krabang -> Lat Krabang

    # Famous Areas/Landmarks (Thai)
    "สยาม": "ปทุมวัน",
    "ชิดลม": "ปทุมวัน",
    "อโศก": "วัฒนา",
    "เอกมัย": "วัฒนา",
    "ทองหล่อ": "วัฒนา",
    "อารีย์": "พญาไท",
    "ห้วยขวาง": "ห้วยขวาง",
    "รัชดาภิเษก": "ดินแดง",
    "อ่อนนุช": "ประเวศ",
    "บางนา": "ประเวศ",
    "ลาดกระบัง": "ลาดกระบัง",

    # District names (English -> Thai)
    "Pathumwan": "ปทุมวัน",
    "Bang Kapi": "บางกะปิ",
    "Din Daeng": "ดินแดง",
    "Huai Khwang": "ห้วยขวาง",
    "Vadhana": "วัฒนา",
    "Watthana": "วัฒนา",
    "Khlong Toei": "คลองเตย",
    "Klong Toei": "คลองเตย",
    "Don Mueang": "ดอนเมือง",
    "Bang Khen": "บางเขน",
    "Prawet": "ประเวศ",
    "Saphan Sung": "สะพานสูง",

    # Subdistricts (Thai)
    "คลองเตย": "คลองเตย",
    "ปทุมวัน": "ปทุมวัน",
    "บางกะปิ": "บางกะปิ",
}

# Bangkok & Metro district coordinates (approximate centers)
DISTRICT_COORDS = {
    # Bangkok Districts (เขต)
    "ปทุมวัน": (13.75, 100.53),      # Pathumwan
    "บางกะปิ": (13.76, 100.64),      # Bang Kapi
    "ดินแดง": (13.76, 100.54),        # Din Daeng
    "ห้วยขวาง": (13.78, 100.57),     # Huai Khwang
    "วัฒนา": (13.73, 100.56),         # Vadhana (Watthana)
    "คลองเตย": (13.72, 100.58),      # Khlong Toei
    "ดอนเมือง": (13.91, 100.61),     # Don Mueang
    "บางเขน": (13.85, 100.60),       # Bang Khen
    "ประเวศ": (13.70, 100.66),       # Prawet
    "สะพานสูง": (13.80, 100.65),     # Saphan Sung
    "บางพลัด": (13.79, 100.49),      # Bang Phlat
    "บางกอกน้อย": (13.76, 100.48),   # Bangkok Noi
    "บางกอกใหญ่": (13.73, 100.50),   # Bangkok Yai
    "ธนบุรี": (13.73, 100.49),        # Thonburi
    "ราชเทวี": (13.75, 100.54),      # Ratchathewi
    "พญาไท": (13.78, 100.54),        # Phaya Thai
    "จตุจักร": (13.81, 100.56),      # Chatuchak
    "ลาดพร้าว": (13.82, 100.60),     # Lat Phrao
    "วังทองหลาง": (13.78, 100.59),   # Wang Thonglang
    "บึงกุ่ม": (13.81, 100.64),       # Bueng Kum
    "สาทร": (13.72, 100.53),          # Sathon
    "บางรัก": (13.73, 100.52),        # Bang Rak
    "ยานนาวา": (13.69, 100.53),      # Yan Nawa
    "ราษฎร์บูรณะ": (13.66, 100.51),  # Rat Burana
    "ทุ่งครุ": (13.62, 100.50),       # Thung Khru
    "ลาดกระบัง": (13.73, 100.78),    # Lat Krabang

    # Surrounding Provinces
    "เมืองนนทบุรี": (13.86, 100.52), # Mueang Nonthaburi
    "เมืองปทุมธานี": (14.02, 100.53), # Mueang Pathum Thani
    "เมืองสมุทรปราการ": (13.60, 100.60), # Mueang Samut Prakan

    "default": (13.75, 100.50)  # Central Bangkok fallback
}

def infer_district_from_location(location_text, extracted_district):
    """
    Infer district from location details like street names or areas.
    Falls back to LLM-extracted district, then to default.
    """
    if not location_text or pd.isna(location_text):
        location_text = ""

    # Try to find area/street keywords in the location text (case-insensitive)
    location_lower = location_text.lower()
    for area_keyword, district in AREA_TO_DISTRICT.items():
        if area_keyword.lower() in location_lower:
            print(f"    🔍 Inferred '{district}' from area keyword '{area_keyword}' in location")
            return district

    # Use the LLM-extracted district if available and map to Thai if English
    if extracted_district and not pd.isna(extracted_district) and extracted_district != "Unknown":
        # Try to map English district name to Thai
        if extracted_district in AREA_TO_DISTRICT:
            mapped_district = AREA_TO_DISTRICT[extracted_district]
            print(f"    🗺️  Mapped '{extracted_district}' to '{mapped_district}'")
            return mapped_district
        # Check if it's already a Thai district name
        elif extracted_district in DISTRICT_COORDS:
            return extracted_district
        else:
            print(f"    ⚠️  Unknown district name '{extracted_district}', using default")
            return "default"

    # Default fallback
    print(f"    ⚠️  Could not infer district from '{location_text[:50] if len(location_text) > 50 else location_text}', using default")
    return "default"

def get_district_coords(district):
    """Get coordinates for a district, with fallback to central Bangkok."""
    if not district or pd.isna(district):
        return DISTRICT_COORDS["default"]

    # Try exact match first
    if district in DISTRICT_COORDS:
        return DISTRICT_COORDS[district]

    # Try partial match (in case of formatting differences)
    for known_district, coords in DISTRICT_COORDS.items():
        if known_district in district or district in known_district:
            print(f"    ℹ️  Matched '{district}' to '{known_district}'")
            return coords

    # No match found
    return DISTRICT_COORDS["default"]

def fetch_weather_history(dates, lat=13.75, long=100.50):
    """Fetches valid historical weather for the given dates."""
    print(f"☁️ Fetching weather for {len(dates)} days at ({lat:.2f}, {long:.2f})...")
    
    # Open-Meteo Archive requires a start/end range
    # It has a ~5 day lag. We filter later, but the request asks for the range.
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')

    # Setup Cache & Retry to be polite to the API
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "rain", "wind_gusts_10m"],
        "timezone": "Asia/Bangkok"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process Hourly Data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )
        }
        hourly_data["temp"] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data["rain"] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data["wind_gust"] = hourly.Variables(2).ValuesAsNumpy()

        weather_df = pd.DataFrame(data = hourly_data)
        
        # Aggregate Hourly -> Daily (Max Temp, Total Rain, Max Gust)
        # Note: We align by DATE only
        weather_df['date_key'] = weather_df['date'].dt.strftime('%Y-%m-%d')
        
        daily_weather = weather_df.groupby('date_key').agg({
            'temp': 'max',
            'rain': 'sum',
            'wind_gust': 'max'
        }).reset_index()
        
        return daily_weather

    except Exception as e:
        print(f"  [!] Weather API Error: {e}")
        return pd.DataFrame()

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print("🚀 Starting Pipeline...")
    
    # --- STEP 1: LOAD & FILTER DATES ---
    df = pd.read_csv(INPUT_FILE)
    df['outage_date'] = pd.to_datetime(df['outage_date'])
    
    # The Filter: Keep only dates strictly BEFORE "Tomorrow"
    # We also buffer 5 days for the Weather Archive (it lags slightly)
    today = pd.Timestamp.now().normalize()
    archive_cutoff = today - pd.Timedelta(days=5)
    
    # Set 1: Valid History (We can get weather for these)
    valid_history_df = df[df['outage_date'] < archive_cutoff].copy()
    
    # Set 2: Recent/Future (We keep them, but skip weather fetch to avoid errors)
    recent_df = df[df['outage_date'] >= archive_cutoff].copy()
    # If you strictly want to DELETE future data (Dec 5th):
    recent_df = recent_df[recent_df['outage_date'] <= today] 

    print(f"   - Total Rows: {len(df)}")
    print(f"   - Valid History (Weather Available): {len(valid_history_df)}")
    print(f"   - Recent/Today (Weather Skipped): {len(recent_df)}")
    print(f"   - Future Dropped: {len(df) - len(valid_history_df) - len(recent_df)}")

    # --- STEP 2: EXTRACT STRUCTURE (Gemini) ---
    all_extracted_events = []
    
    # Process both sets (we still want the location data for recent days)
    processing_queue = pd.concat([valid_history_df, recent_df])
    
    print(f"\n🧠 Extracting data from {len(processing_queue)} rows with local LLM...")
    for index, row in processing_queue.iterrows():
        date_str = row['outage_date'].strftime('%Y-%m-%d')
        events = extract_events_with_local_llm(row['outage_data'], date_str)
        
        for event in events:
            all_extracted_events.append({
                "date": date_str,
                "day_of_week": row['day_of_week'],
                "start": event.start_time_24h,
                "end": event.end_time_24h,
                "location": event.location_detail,
                "district": event.district,
                "province": event.province
            })
            
    structured_df = pd.DataFrame(all_extracted_events)
    
    # --- STEP 3: MERGE WEATHER ---
    if not structured_df.empty:
        print("\n☁️  Fetching Weather Data per District...")

        # Add weather columns
        structured_df['temp'] = None
        structured_df['rain'] = None
        structured_df['wind_gust'] = None

        # Group by district and date to minimize API calls
        weather_cache = {}

        for idx, row in structured_df.iterrows():
            # Only fetch weather for historical dates (not recent/future)
            date_obj = pd.to_datetime(row['date'])
            archive_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=5)

            if date_obj >= archive_cutoff:
                continue  # Skip recent dates (no weather available)

            # Infer district from location + LLM extraction
            district = infer_district_from_location(row['location'], row['district'])
            date_str = row['date']
            cache_key = f"{district}_{date_str}"

            # Check cache first
            if cache_key in weather_cache:
                weather = weather_cache[cache_key]
            else:
                # Get coordinates for this district
                lat, lon = get_district_coords(district)

                # Fetch weather for this specific date and location
                date_series = pd.Series([pd.to_datetime(date_str)])
                weather_df = fetch_weather_history(date_series, lat=lat, long=lon)

                if not weather_df.empty:
                    weather = {
                        'temp': weather_df.iloc[0]['temp'],
                        'rain': weather_df.iloc[0]['rain'],
                        'wind_gust': weather_df.iloc[0]['wind_gust']
                    }
                    weather_cache[cache_key] = weather
                else:
                    weather = None

            # Apply weather to this row
            if weather:
                structured_df.at[idx, 'temp'] = weather['temp']
                structured_df.at[idx, 'rain'] = weather['rain']
                structured_df.at[idx, 'wind_gust'] = weather['wind_gust']

        final_df = structured_df
            
        # --- STEP 4: SAVE ---
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Done! Saved {len(final_df)} events to {OUTPUT_FILE}")
        print(final_df.head())
        print("\nNote: Rows with empty weather columns are from the last 5 days (Archive lag).")

if __name__ == "__main__":
    main()