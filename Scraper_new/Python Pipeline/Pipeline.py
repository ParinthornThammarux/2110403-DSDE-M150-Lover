import pandas as pd
import instructor
import google.generativeai as genai
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
os.environ["GEMINI_API_KEY"] = "AIzaSyCuuv-YBlW1whCyOh4pFUBxs34WezYgYpY" # <--- Paste your Gemini Key
INPUT_FILE = "mea_power_outages_page_001.csv"
OUTPUT_FILE = "training_data_final.csv"

# ==========================================
# 2. DEFINE DATA STRUCTURE
# ==========================================
class OutageEvent(BaseModel):
    start_time_24h: str = Field(..., description="Start time in HH:MM (24h) format.")
    end_time_24h: str = Field(..., description="End time in HH:MM (24h) format.")
    location_detail: str = Field(..., description="Specific street, soi, or village.")
    # We add lat/long placeholders. The AI will leave them null, 
    # but we want the columns in our final CSV for future use.
    district: str = Field(..., description="The District (Khet) if mentioned or inferable.")

class DailyOutageSchedule(BaseModel):
    events: List[OutageEvent]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_events_with_gemini(row_text, date_context):
    """Sends text to Gemini Flash to extract structured JSON."""
    if not isinstance(row_text, str) or not row_text.strip():
        return []

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name="gemini-1.5-flash"),
        mode=instructor.Mode.GEMINI_JSON
    )

    try:
        resp = client.chat.completions.create(
            response_model=DailyOutageSchedule,
            messages=[{
                "role": "user", 
                "content": f"Date: {date_context}. Extract outage events: {row_text}"
            }],
        )
        return resp.events
    except Exception as e:
        print(f"  [!] AI Error on {date_context}: {e}")
        return []

def fetch_weather_history(dates, lat=13.75, long=100.50):
    """Fetches valid historical weather for the given dates."""
    print(f"☁️ Fetching weather for {len(dates)} days...")
    
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
    
    print(f"\n🧠 Extracting data from {len(processing_queue)} rows with Gemini...")
    for index, row in processing_queue.iterrows():
        date_str = row['outage_date'].strftime('%Y-%m-%d')
        events = extract_events_with_gemini(row['outage_data'], date_str)
        
        for event in events:
            all_extracted_events.append({
                "date": date_str,
                "day_of_week": row['day_of_week'],
                "start": event.start_time_24h,
                "end": event.end_time_24h,
                "location": event.location_detail,
                "district": event.district
            })
            
    structured_df = pd.DataFrame(all_extracted_events)
    
    # --- STEP 3: MERGE WEATHER ---
    if not structured_df.empty:
        print("\n☁️  Merging Weather Data...")
        
        # Get unique dates from the Valid History portion only
        unique_dates = valid_history_df['outage_date'].unique()
        
        if len(unique_dates) > 0:
            weather_data = fetch_weather_history(pd.to_datetime(unique_dates))
            
            if not weather_data.empty:
                # Merge: We use 'left' join so we don't lose the recent data 
                # (recent data will just have NaN for weather, which is expected)
                final_df = pd.merge(structured_df, weather_data, 
                                    left_on='date', right_on='date_key', 
                                    how='left')
                final_df.drop(columns=['date_key'], inplace=True)
            else:
                final_df = structured_df
        else:
            final_df = structured_df
            
        # --- STEP 4: SAVE ---
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Done! Saved {len(final_df)} events to {OUTPUT_FILE}")
        print(final_df.head())
        print("\nNote: Rows with empty weather columns are from the last 5 days (Archive lag).")

if __name__ == "__main__":
    main()