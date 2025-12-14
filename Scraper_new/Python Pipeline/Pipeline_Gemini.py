import pandas as pd
import instructor
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import datetime
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Gemini API Configuration
GEMINI_API_KEY = "GEMINI_ENVIRONMENT_KEY_HERE"
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable. Get your key from: https://aistudio.google.com/apikey")

GEMINI_MODEL = "gemini-2.5-flash-lite"  # balanced option

INPUT_FILE = r"D:\Chula Year2\DSDE Proj2\2110403-DSDE-M150-Lover\Scraper_new\MEA Scraper\data\external\scraped\mea_power_outages_page_013.csv"
OUTPUT_FILE = "mea_power_outages_page_013_gemini.csv"
BATCH_SIZE = 20

# ==========================================
# 2. DEFINE DATA STRUCTURE
# ==========================================
class OutageEvent(BaseModel):
    start_time_24h: str = Field(..., description="Start time in HH:MM (24h) format.")
    end_time_24h: str = Field(..., description="End time in HH:MM (24h) format.")
    location_detail: str = Field(..., description="Specific street, soi, or village.")
    district: str = Field(
        default="Unknown",
        description="Infer the Bangkok district from the location detail. "
                    "Use your knowledge of Bangkok geography. For example, if the location is 'Siam Paragon', the district should be 'Pathumwan'. "
                    "If you cannot infer a district, return 'Unknown'."
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
def extract_events_with_gemini(row_text, date_context):
    """Sends text to Gemini API to extract structured JSON."""
    if not isinstance(row_text, str) or not row_text.strip():
        return []

    # Configure Gemini client
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"  [DEBUG] Using Gemini model: {GEMINI_MODEL} for date: {date_context}")

    # Create instructor client with Gemini mode
    client = instructor.from_genai(
        client=gemini_client,
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
    )

    try:
        print(f"  [DEBUG] Sending request to Gemini API...")
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            response_model=DailyOutageSchedule,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert at extracting structured data from power outage announcements and you have deep knowledge of Bangkok's geography.
The text is in English. For each power outage event, extract the time range and the specific location.
Based on the location, infer the corresponding district in Bangkok.

For example, if the location mentioned is 'Siam Paragon', you should infer the district as 'Pathumwan'.
If a street name can be in multiple districts, use other information in the text to infer the most likely one.
If no specific district can be inferred, you can use 'Unknown'.

Date: {date_context}

Extract all power outage events from this English text.

{row_text}"""
                }
            ],
        )
        print(f"  [DEBUG] Received {len(resp.events)} events from Gemini")
        return resp.events
    except Exception as e:
        print(f"  [!] Gemini API Error on {date_context}: {e}")
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


def infer_district_from_location(extracted_district):
    """
    Tries to map the LLM-extracted district to a Thai name for coordinate lookup.
    """
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
            print(f"    ⚠️  Unknown district name from LLM '{extracted_district}', using default")
            return "default"

    # Default fallback
    print(f"    ⚠️  LLM did not provide a district, using default")
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

def add_weather_data(structured_df, weather_cache):
    """Fetches and merges weather data for the events in the DataFrame."""
    if structured_df.empty:
        return structured_df

    print("☁️  Fetching and merging weather data...")

    structured_df['temp'] = None
    structured_df['rain'] = None
    structured_df['wind_gust'] = None

    for idx, row in structured_df.iterrows():
        date_obj = pd.to_datetime(row['date'])
        archive_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=5)

        if date_obj >= archive_cutoff:
            continue

        district = infer_district_from_location(row['district'])
        date_str = row['date']
        cache_key = f"{district}_{date_str}"

        if cache_key in weather_cache:
            weather = weather_cache[cache_key]
        else:
            lat, lon = get_district_coords(district)
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

        if weather:
            structured_df.at[idx, 'temp'] = weather['temp']
            structured_df.at[idx, 'rain'] = weather['rain']
            structured_df.at[idx, 'wind_gust'] = weather['wind_gust']

    return structured_df


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print("🚀 Starting Pipeline with Gemini API...")

    # --- STEP 1: LOAD & FILTER DATES ---
    df = pd.read_csv(INPUT_FILE)
    df['outage_date'] = pd.to_datetime(df['outage_date'])

    today = pd.Timestamp.now().normalize()
    archive_cutoff = today - pd.Timedelta(days=5)
    
    valid_history_df = df[df['outage_date'] < archive_cutoff].copy()
    recent_df = df[df['outage_date'] >= archive_cutoff].copy()
    recent_df = recent_df[recent_df['outage_date'] <= today]

    print(f"   - Total Rows: {len(df)}")
    print(f"   - Valid History (Weather Available): {len(valid_history_df)}")
    print(f"   - Recent/Today (Weather Skipped): {len(recent_df)}")

    # --- STEP 2: EXTRACT AND SAVE IN BATCHES ---
    all_extracted_events = []
    weather_cache = {}
    output_file_exists = os.path.exists(OUTPUT_FILE)

    if output_file_exists:
        print(f"⚠️ Output file '{OUTPUT_FILE}' exists and will be overwritten.")
        os.remove(OUTPUT_FILE)
        output_file_exists = False

    processing_queue = pd.concat([valid_history_df, recent_df])
    # Sort by date descending (newest first) to match input CSV order
    processing_queue = processing_queue.sort_values('outage_date', ascending=False).reset_index(drop=True)

    print(f"\n🧠 Extracting data from {len(processing_queue)} rows with Gemini API...")

    for index, row in processing_queue.iterrows():
        print(f"  - Processing row {index + 1}/{len(processing_queue)}")
        date_str = row['outage_date'].strftime('%Y-%m-%d')
        events = extract_events_with_gemini(row['outage_data'], date_str)
        time.sleep(1)

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

        if (index + 1) % BATCH_SIZE == 0 or (index + 1) == len(processing_queue):
            if not all_extracted_events:
                continue

            print(f"\n--- Processing batch of {len(all_extracted_events)} extracted events ---")
            batch_df = pd.DataFrame(all_extracted_events)
            
            # Add weather data
            batch_df = add_weather_data(batch_df, weather_cache)

            # Save batch
            print(f"💾 Saving batch to '{OUTPUT_FILE}'...")
            if not output_file_exists:
                batch_df.to_csv(OUTPUT_FILE, index=False, mode='w')
                output_file_exists = True
            else:
                batch_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
            
            all_extracted_events = [] # Reset for next batch

    print(f"\n✅ Done! All batches saved to {OUTPUT_FILE}")
    # Displaying the head of the final file for verification
    if os.path.exists(OUTPUT_FILE):
        final_df = pd.read_csv(OUTPUT_FILE)
        print("\nFinal Data Preview:")
        print(final_df.head())

if __name__ == "__main__":
    main()
