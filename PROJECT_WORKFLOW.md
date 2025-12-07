# Urban Issue Forecasting System - Complete Project Workflow

## Project Overview
**Team:** DSDE M150-Lover (Chulalongkorn University CEDT)
**Project:** Urban Issue Forecasting System with Multi-Source Integration
**Data Sources:** Bangkok Traffy Fondue (Citizen Complaints) + MEA Power Outage Data
**Scale:** 780,000+ processed records

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Data Collection (Scraping)](#phase-1-data-collection-scraping)
3. [Phase 2: Data Extraction & Enrichment](#phase-2-data-extraction--enrichment)
4. [Phase 3: Data Cleaning & Integration](#phase-3-data-cleaning--integration)
5. [Phase 4: Machine Learning Models](#phase-4-machine-learning-models)
6. [Phase 5: Visualization & Dashboard](#phase-5-visualization--dashboard)
7. [Complete Workflow Diagram](#complete-workflow-diagram)
8. [Technology Stack](#technology-stack)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                              │
│  ┌─────────────────────┐              ┌─────────────────────┐            │
│  │  Bangkok Traffy     │              │  MEA Website        │            │
│  │  API/Database       │              │  (Web Scraping)     │            │
│  └──────────┬──────────┘              └──────────┬──────────┘            │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                                  │
│  ┌─────────────────────┐              ┌─────────────────────┐            │
│  │  Direct Load        │              │  Selenium Scraper   │            │
│  │  (CSV Import)       │              │  + BeautifulSoup    │            │
│  │                     │              └──────────┬──────────┘            │
│  │                     │                         │                        │
│  │                     │              ┌──────────▼──────────┐            │
│  │                     │              │  Gemini LLM         │            │
│  │                     │              │  Extraction         │            │
│  │                     │              │  (Pydantic Models)  │            │
│  │                     │              └──────────┬──────────┘            │
│  │                     │                         │                        │
│  │                     │              ┌──────────▼──────────┐            │
│  │                     │              │  OpenMeteo API      │            │
│  │                     │              │  Weather Enrichment │            │
│  └──────────┬──────────┘              └──────────┬──────────┘            │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATA INTEGRATION LAYER                                 │
│  ┌───────────────────────────────────────────────────────────┐           │
│  │  Jupyter Notebooks (ETL)                                  │           │
│  │  - clean_data_traffy.ipynb                                │           │
│  │  - clean_data_scarping.ipynb                              │           │
│  │  - merge_file.ipynb                                       │           │
│  │                                                            │           │
│  │  Processing:                                               │           │
│  │  • Missing value imputation                                │           │
│  │  • Timestamp parsing & validation                          │           │
│  │  • Geographic coordinate validation                        │           │
│  │  • One-hot encoding (state, star_rating)                  │           │
│  │  • Record deduplication                                    │           │
│  └───────────────────────────┬────────────────────────────────┘           │
└────────────────────────────┼─────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        CLEAN DATA STORAGE                                 │
│  ┌─────────────────────┐              ┌─────────────────────┐            │
│  │  clean_data.csv     │              │  clean_scraping     │            │
│  │  (Traffy Data)      │              │  _data.csv          │            │
│  │  780,000+ records   │              │  (MEA Data)         │            │
│  └──────────┬──────────┘              └──────────┬──────────┘            │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                                 │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐│
│  │ RandomForest         │  │ Isolation Forest │  │ K-Means             ││
│  │ Time-Series          │  │ Anomaly          │  │ Outage              ││
│  │ Forecasting          │  │ Detection        │  │ Clustering          ││
│  │                      │  │                  │  │                     ││
│  │ Input:               │  │ Input:           │  │ Input:              ││
│  │ • Historical counts  │  │ • Multi-feature  │  │ • Outage patterns   ││
│  │ • Time features      │  │   vectors        │  │ • Weather data      ││
│  │ • Rolling stats      │  │ • Geo coordinates│  │ • Location data     ││
│  │                      │  │ • Temporal data  │  │                     ││
│  │ Output:              │  │                  │  │ Output:             ││
│  │ • 7-day forecast     │  │ Output:          │  │ • Cluster labels    ││
│  │ • Confidence metrics │  │ • Anomaly flags  │  │ • Cluster centers   ││
│  │ • Model: .pkl        │  │ • Model: .pkl    │  │ • Model: .pkl       ││
│  └──────────────────────┘  └──────────────────┘  └─────────────────────┘│
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION LAYER                                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Streamlit Dashboard                             │  │
│  │                    (main_dashboard.py)                             │  │
│  │                                                                    │  │
│  │  Tab 1: Geospatial Visualization                                  │  │
│  │    • Folium HeatMaps (complaint density)                          │  │
│  │    • Interactive Marker Clusters                                  │  │
│  │    • PyDeck 3D Grid Layers                                        │  │
│  │                                                                    │  │
│  │  Tab 2: District & Type Analysis                                  │  │
│  │    • Bar charts (top districts)                                   │  │
│  │    • Stacked bar charts (complaint types)                         │  │
│  │    • Time-series line charts                                      │  │
│  │                                                                    │  │
│  │  Tab 3: MEA Power Outage Data                                     │  │
│  │    • Outage distribution charts                                   │  │
│  │    • Weather correlation analysis                                 │  │
│  │                                                                    │  │
│  │  Tab 4: ML Predictive Forecasting                                 │  │
│  │    • Forecast line charts                                         │  │
│  │    • Actual vs Predicted overlays                                 │  │
│  │    • Per-horizon MAE metrics                                      │  │
│  │                                                                    │  │
│  │  Tab 5: ML Anomaly Detection                                      │  │
│  │    • Anomaly scatter plots                                        │  │
│  │    • Distribution by type/district                                │  │
│  │                                                                    │  │
│  │  Tab 6: ML Power Outage Clustering                                │  │
│  │    • Cluster distribution charts                                  │  │
│  │    • Cluster characteristics (6-subplot layout)                   │  │
│  │                                                                    │  │
│  │  Tab 7: Additional Analysis                                       │  │
│  │    • Custom analytics                                             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Collection (Scraping)

### 1.1 MEA Power Outage Web Scraper

**File:** [Scraper_new/MEA Scraper/mea_outage_scraper_v2.py](Scraper_new/MEA Scraper/mea_outage_scraper_v2.py)

#### Overview
Automated web scraper that collects power outage announcements from the Metropolitan Electricity Authority (MEA) website.

#### Technology Stack
- **Selenium WebDriver**: Dynamic content rendering
- **BeautifulSoup**: HTML parsing
- **ThreadPoolExecutor**: Parallel processing (12 workers)
- **Pandas**: Data structuring
- **Requests**: HTTP client with session management

#### Workflow Steps

1. **Initialization**
   ```
   Input: MEA website URL
   Configuration:
   - Headless Chrome browser
   - User-Agent spoofing
   - 12 parallel workers (configurable)
   ```

2. **Page Navigation**
   ```
   For each page in pagination:
     - Load page with Selenium
     - Wait for dynamic content (3s delay)
     - Parse HTML with BeautifulSoup
     - Extract announcement links
   ```

3. **Parallel Announcement Processing**
   ```
   For each announcement link (parallel):
     - Create dedicated Chrome driver
     - Navigate to announcement detail page
     - Extract content from div.content-description
     - Parse day-based sections using regex
     - Create one record per outage day
   ```

4. **Data Extraction Pattern**
   ```
   Regex Pattern: (Monday|Tuesday|...|Sunday), (Month) (Day), (Year)

   For each day section:
     - Extract day_of_week
     - Parse outage_date
     - Clean and format outage_data text
     - Store as structured record
   ```

5. **Text Cleaning Pipeline**
   ```
   - Fix scattered time formats ("08. 3 0" → "08:30")
   - Normalize time separators ("." → ":")
   - Remove footer text (apologies, contact info)
   - Fix CamelCase words spacing
   - Remove zero-width spaces
   - Clean excessive newlines
   - Filter out single-digit/punctuation-only lines
   ```

6. **Batch Export**
   ```
   For each scraped page:
     - Create DataFrame with columns:
       * source (always "MEA")
       * announcement_url
       * day_of_week
       * outage_date
       * outage_data (raw text)

     - Remove duplicates (by date + URL)
     - Sort by outage_date (descending)
     - Save as: mea_power_outages_page_{page_num:03d}.csv
   ```

#### Output Format
```csv
source,announcement_url,day_of_week,outage_date,outage_data
MEA,https://www.mea.or.th/en/...,Saturday,2024-09-06,"Time: 08:30 AM - 04:30 PM..."
```

#### Command-Line Usage
```bash
python mea_outage_scraper_v2.py \
  --start-page 1 \
  --max-pages 50 \
  --workers 12 \
  --output combined_outages.csv
```

#### Performance Metrics
- **Parallel Workers**: 12 concurrent Selenium instances
- **Rate**: ~50-100 announcements per minute
- **Error Handling**: Automatic retry with 3-failure limit
- **Logging**: UTF-8 encoded logs with timestamps

---

### 1.2 Bangkok Traffy Data Collection

**Source:** Direct API/Database access to Traffy Fondue platform

#### Data Fields
```
- ticket_id: Unique complaint identifier
- timestamp: Report datetime
- type: Complaint category (flood, traffic, waste, etc.)
- district: Bangkok district
- latitude, longitude: Geographic coordinates
- state: Processing status
- star_rating: User satisfaction rating
- solve_days: Resolution time in days
```

#### Collection Method
```
Direct CSV import from Traffy Fondue database export
No scraping required - official data source
```

---

## Phase 2: Data Extraction & Enrichment

### 2.1 LLM-Based Structured Extraction

**File:** [Scraper_new/Python Pipeline/Pipeline_Gemini.py](Scraper_new/Python Pipeline/Pipeline_Gemini.py)

#### Overview
Transforms unstructured MEA outage text into structured data using Google Gemini LLM with geographic and weather enrichment.

#### Technology Stack
- **Google Gemini API**: LLM for structured extraction (gemini-2.5-flash-lite)
- **Instructor Framework**: Type-safe LLM outputs with Pydantic
- **OpenMeteo API**: Historical weather data
- **Pydantic**: Data validation and serialization

#### Workflow Steps

1. **Data Model Definition**
   ```python
   class OutageEvent(BaseModel):
       start_time_24h: str          # HH:MM format
       end_time_24h: str            # HH:MM format
       location_detail: str         # Street/Soi/Village
       district: str                # Inferred Bangkok district
       province: str                # Default: "Bangkok"

   class DailyOutageSchedule(BaseModel):
       events: List[OutageEvent]
   ```

2. **LLM Extraction Process**
   ```
   For each row in scraped CSV:
     Input:
       - outage_data (raw text)
       - outage_date (context for LLM)

     LLM Prompt:
       "Extract power outage events from this English text.
        For each event, extract time range and location.
        Infer Bangkok district from location using geographic knowledge.
        Example: 'Siam Paragon' → district: 'Pathumwan'"

     Output:
       - Structured JSON matching OutageEvent schema
       - Multiple events per day if applicable
   ```

3. **Geographic Mapping**
   ```
   District Inference Chain:

   Step 1: LLM extracts English location/district
   Step 2: Map to Thai district name using AREA_TO_DISTRICT dict
           Examples:
           - "Sukhumvit" → "วัฒนา" (Vadhana)
           - "Siam" → "ปทุมวัน" (Pathumwan)
           - "Asoke" → "วัฒนา" (Vadhana)

   Step 3: Get coordinates from DISTRICT_COORDS dict
           Example: "ปทุมวัน" → (13.75, 100.53)

   Fallback: Unknown → Central Bangkok (13.75, 100.50)
   ```

4. **Weather Data Enrichment**
   ```
   For each extracted event:

     Filter:
       - Only fetch for dates > 5 days old
         (OpenMeteo Archive has 5-day lag)

     API Call:
       URL: https://archive-api.open-meteo.com/v1/archive
       Parameters:
         - latitude, longitude (from district mapping)
         - date range (start_date to end_date)
         - hourly metrics:
           * temperature_2m
           * rain
           * wind_gusts_10m
         - timezone: Asia/Bangkok

     Aggregation:
       Hourly → Daily:
         - temperature: max
         - rain: sum (total rainfall)
         - wind_gust: max

     Cache:
       Key: "{district}_{date}"
       Purpose: Avoid duplicate API calls
   ```

5. **Batch Processing**
   ```
   Batch Size: 20 rows

   For each batch:
     1. Extract events with Gemini (1s delay between calls)
     2. Flatten events to DataFrame rows
     3. Add weather data (with caching)
     4. Append to CSV output

   Output Mode:
     - First batch: Write with headers
     - Subsequent batches: Append mode (no headers)
   ```

#### Output Format
```csv
date,day_of_week,start,end,location,district,province,temp,rain,wind_gust
2024-09-06,Saturday,08:30,16:30,Soi Sukhumvit 23,Pathumwan,Bangkok,32.5,5.2,25.3
```

#### API Rate Limiting
```
- Gemini API: 1 second delay between requests
- OpenMeteo: Cached session with 5-retry backoff
- Weather cache: In-memory dict (district_date key)
```

---

## Phase 3: Data Cleaning & Integration

### 3.1 Traffy Data Cleaning

**File:** [clean_data/clean_data_traffy.ipynb](clean_data/clean_data_traffy.ipynb)

#### Processing Steps

1. **Missing Value Handling**
   ```python
   # Identify missing values
   missing_summary = df.isnull().sum()

   # Imputation strategies:
   - Numeric: median imputation (solve_days)
   - Categorical: mode imputation (district, type)
   - Geographic: drop records with missing lat/lon (< 0.1%)
   ```

2. **Timestamp Parsing**
   ```python
   # Convert to datetime
   df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

   # Extract features
   df['hour'] = df['timestamp'].dt.hour
   df['day_of_week'] = df['timestamp'].dt.dayofweek
   df['month'] = df['timestamp'].dt.month
   df['is_weekend'] = df['day_of_week'].isin([5, 6])
   ```

3. **Categorical Encoding**
   ```python
   # One-hot encoding for state
   state_dummies = pd.get_dummies(df['state'], prefix='state')

   # One-hot encoding for star_rating
   rating_dummies = pd.get_dummies(df['star_rating'], prefix='rating')

   df = pd.concat([df, state_dummies, rating_dummies], axis=1)
   ```

4. **Geographic Validation**
   ```python
   # Bangkok coordinate bounds
   lat_min, lat_max = 13.5, 14.0
   lon_min, lon_max = 100.3, 100.9

   # Filter invalid coordinates
   df = df[
       (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
       (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
   ]
   ```

5. **Deduplication**
   ```python
   # Remove exact duplicates
   df = df.drop_duplicates(subset=['ticket_id'], keep='first')

   # Remove near-duplicates (same location + time within 1 hour)
   df = df.drop_duplicates(
       subset=['latitude', 'longitude', 'timestamp_hour'],
       keep='first'
   )
   ```

---

### 3.2 MEA Scraping Data Cleaning

**File:** [clean_data/clean_data_scarping.ipynb](clean_data/clean_data_scarping.ipynb)

#### Processing Steps

1. **Date Validation**
   ```python
   # Parse dates
   df['date'] = pd.to_datetime(df['date'], errors='coerce')

   # Remove future dates
   df = df[df['date'] <= pd.Timestamp.now()]

   # Remove invalid dates (NaT)
   df = df.dropna(subset=['date'])
   ```

2. **Time Format Standardization**
   ```python
   # Convert to 24-hour format
   def parse_time(time_str):
       # Handle "08:30 AM" → "08:30"
       # Handle "02:30 PM" → "14:30"
       return pd.to_datetime(time_str, format='%I:%M %p').strftime('%H:%M')

   df['start'] = df['start'].apply(parse_time)
   df['end'] = df['end'].apply(parse_time)
   ```

3. **Duration Calculation**
   ```python
   # Calculate outage duration in minutes
   df['duration'] = (
       pd.to_datetime(df['end'], format='%H:%M') -
       pd.to_datetime(df['start'], format='%H:%M')
   ).dt.total_seconds() / 60

   # Handle overnight outages
   df.loc[df['duration'] < 0, 'duration'] += 24 * 60
   ```

4. **District Normalization**
   ```python
   # Standardize Thai district names
   district_mapping = {
       'ปทุมวัน': 'Pathumwan',
       'วัฒนา': 'Vadhana',
       # ... (25 districts)
   }

   df['district_eng'] = df['district'].map(district_mapping)
   ```

---

### 3.3 Data Merge & Integration

**File:** [clean_data/merge_file.ipynb](clean_data/merge_file.ipynb)

#### Merging Strategy

1. **Schema Alignment**
   ```python
   # Standardize column names
   traffy_cols = {
       'ticket_id': 'id',
       'timestamp': 'datetime',
       'type': 'category'
   }

   mea_cols = {
       'date': 'datetime',
       'location': 'address',
       'district': 'district'
   }
   ```

2. **Temporal Alignment**
   ```python
   # Aggregate both datasets by date + district
   traffy_daily = traffy_df.groupby(['date', 'district']).agg({
       'ticket_id': 'count',
       'solve_days': 'mean'
   }).reset_index()

   mea_daily = mea_df.groupby(['date', 'district']).agg({
       'duration': 'sum'
   }).reset_index()

   # Left join (keep all Traffy records)
   merged_df = pd.merge(
       traffy_daily,
       mea_daily,
       on=['date', 'district'],
       how='left'
   )
   ```

3. **Feature Engineering**
   ```python
   # Create combined features
   merged_df['has_outage'] = merged_df['duration'].notna()
   merged_df['outage_duration_hours'] = merged_df['duration'] / 60
   merged_df['complaints_per_outage_hour'] = (
       merged_df['ticket_id'] / merged_df['outage_duration_hours']
   )
   ```

4. **Export**
   ```python
   # Save cleaned datasets
   traffy_df.to_csv('clean_data.csv', index=False)
   mea_df.to_csv('clean_scraping_data.csv', index=False)
   merged_df.to_csv('merged_data.csv', index=False)
   ```

---

## Phase 4: Machine Learning Models

### 4.1 RandomForest Time-Series Forecasting

**File:** [ml_models/forecasting/train_lstm_model.py](ml_models/forecasting/train_lstm_model.py)

#### Model Architecture

```
Input Features:
  - Historical complaint counts (30-day lookback)
  - Temporal features: dayofweek, month, is_weekend
  - Rolling statistics: 7-day MA, 30-day MA

Model: RandomForestRegressor (Multi-output)
  - n_estimators: Tuned via GridSearchCV
  - max_depth: Tuned via GridSearchCV
  - min_samples_split: Tuned via GridSearchCV

Output:
  - 7-day ahead forecast (7 values simultaneously)
```

#### Training Pipeline

1. **Data Preparation**
   ```python
   # Load daily complaint counts
   df = pd.read_csv('clean_data.csv')
   daily_counts = df.groupby('date').size().reset_index(name='count')

   # Create time series
   ts = daily_counts.set_index('date')['count']
   ts = ts.asfreq('D', fill_value=0)  # Ensure daily frequency
   ```

2. **Feature Engineering**
   ```python
   # Temporal features
   features['dayofweek'] = ts.index.dayofweek
   features['month'] = ts.index.month
   features['is_weekend'] = ts.index.dayofweek.isin([5, 6]).astype(int)

   # Rolling statistics
   features['ma_7'] = ts.rolling(window=7, min_periods=1).mean()
   features['ma_30'] = ts.rolling(window=30, min_periods=1).mean()

   # Lag features (30-day lookback)
   for i in range(1, 31):
       features[f'lag_{i}'] = ts.shift(i)
   ```

3. **Time-Series Split**
   ```python
   # TimeSeriesSplit for cross-validation
   tscv = TimeSeriesSplit(n_splits=5)

   # Prevents data leakage
   for train_idx, test_idx in tscv.split(X):
       X_train, X_test = X[train_idx], X[test_idx]
       y_train, y_test = y[train_idx], y[test_idx]
   ```

4. **Hyperparameter Tuning**
   ```python
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30, None],
       'min_samples_split': [2, 5, 10]
   }

   grid_search = GridSearchCV(
       RandomForestRegressor(),
       param_grid,
       cv=tscv,
       scoring='neg_mean_absolute_error'
   )

   best_model = grid_search.fit(X_train, y_train)
   ```

5. **Multi-Horizon Forecasting**
   ```python
   # Train separate model for each horizon
   models = {}
   for horizon in range(1, 8):  # 7-day forecast
       y_horizon = create_target(ts, horizon)
       models[horizon] = RandomForestRegressor(**best_params)
       models[horizon].fit(X_train, y_horizon)

   # Or: Single multi-output model
   y_multi = create_multi_target(ts, horizons=[1,2,3,4,5,6,7])
   model = RandomForestRegressor(**best_params)
   model.fit(X_train, y_multi)  # y_multi shape: (n_samples, 7)
   ```

6. **Baseline Comparison**
   ```python
   # Naive forecast (last value)
   naive_forecast = ts.iloc[-1]

   # 7-day moving average
   ma_forecast = ts.rolling(7).mean().iloc[-1]

   # Compare MAE
   model_mae = mean_absolute_error(y_test, y_pred)
   naive_mae = mean_absolute_error(y_test, naive_forecast)
   ma_mae = mean_absolute_error(y_test, ma_forecast)
   ```

7. **Evaluation Metrics**
   ```python
   # Per-horizon metrics
   for horizon in range(1, 8):
       mae = mean_absolute_error(y_test[:, horizon-1], y_pred[:, horizon-1])
       rmse = np.sqrt(mean_squared_error(y_test[:, horizon-1], y_pred[:, horizon-1]))
       mape = np.mean(np.abs((y_test[:, horizon-1] - y_pred[:, horizon-1]) / y_test[:, horizon-1])) * 100

       print(f"Horizon {horizon}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
   ```

8. **Model Persistence**
   ```python
   import joblib

   # Save model
   joblib.dump(model, 'forecasting_model.pkl')

   # Save metadata
   metadata = {
       'features': feature_names,
       'target_horizons': 7,
       'best_params': best_params,
       'metrics': metrics_dict
   }
   joblib.dump(metadata, 'forecasting_metadata.pkl')
   ```

#### Output Format
```python
# Forecast dictionary
{
    'forecast_dates': ['2024-09-08', '2024-09-09', ..., '2024-09-14'],
    'predicted_counts': [150, 145, 160, 140, 155, 165, 150],
    'lower_bound': [140, 135, 150, 130, 145, 155, 140],
    'upper_bound': [160, 155, 170, 150, 165, 175, 160]
}
```

---

### 4.2 Isolation Forest Anomaly Detection

**File:** [ml_models/anomaly_detection/detect_anomalies.py](ml_models/anomaly_detection/detect_anomalies.py)

#### Model Architecture

```
Input Features:
  - Temporal: hour, day_of_week, month
  - Geospatial: latitude, longitude
  - Complaint: solve_days, category (one-hot)
  - Aggregated: district_daily_count

Preprocessing:
  - StandardScaler (mean=0, std=1)

Model: Isolation Forest
  - contamination: 0.05 (5% expected anomalies)
  - n_estimators: 100 (default)
  - max_samples: 'auto'

Output:
  - Anomaly score: [-1, 1] (lower = more anomalous)
  - Binary label: 1 (normal), -1 (anomaly)
```

#### Training Pipeline

1. **Feature Engineering**
   ```python
   # Temporal features
   df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
   df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
   df['month'] = pd.to_datetime(df['timestamp']).dt.month

   # Geospatial features
   features['latitude'] = df['latitude']
   features['longitude'] = df['longitude']

   # Complaint characteristics
   features['solve_days'] = df['solve_days'].fillna(df['solve_days'].median())

   # Category one-hot encoding
   category_cols = ['flood', 'traffic', 'waste', 'roads', 'other']
   for cat in category_cols:
       features[f'type_{cat}'] = (df['type'] == cat).astype(int)

   # District-level aggregation
   district_counts = df.groupby(['date', 'district']).size().reset_index(name='count')
   df = df.merge(district_counts, on=['date', 'district'], how='left')
   features['district_daily_count'] = df['count']
   ```

2. **Data Splitting**
   ```python
   # Time-based split (80/20)
   split_date = df['date'].quantile(0.8)

   train_df = df[df['date'] <= split_date]
   val_df = df[df['date'] > split_date]

   X_train = features.loc[train_df.index]
   X_val = features.loc[val_df.index]
   ```

3. **Preprocessing**
   ```python
   from sklearn.preprocessing import StandardScaler

   # Fit scaler on training data only
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_val_scaled = scaler.transform(X_val)

   # Save scaler for inference
   joblib.dump(scaler, 'anomaly_scaler.pkl')
   ```

4. **Model Training**
   ```python
   from sklearn.ensemble import IsolationForest

   model = IsolationForest(
       contamination=0.05,        # 5% anomaly rate
       n_estimators=100,
       max_samples='auto',
       random_state=42,
       n_jobs=-1                  # Parallel processing
   )

   model.fit(X_train_scaled)
   ```

5. **Prediction & Scoring**
   ```python
   # Predict on validation set
   val_predictions = model.predict(X_val_scaled)
   val_scores = model.decision_function(X_val_scaled)

   # Add to DataFrame
   val_df['anomaly_label'] = val_predictions  # 1 or -1
   val_df['anomaly_score'] = val_scores       # Continuous score

   # Extract anomalies
   anomalies = val_df[val_df['anomaly_label'] == -1]
   print(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(val_df)*100:.2f}%)")
   ```

6. **Unsupervised Evaluation**
   ```python
   from sklearn.metrics import silhouette_score

   # Silhouette score (higher = better separation)
   sil_score = silhouette_score(X_val_scaled, val_predictions)
   print(f"Silhouette Score: {sil_score:.3f}")

   # Stability test with noise injection
   noise = np.random.normal(0, 0.1, X_val_scaled.shape)
   X_noisy = X_val_scaled + noise
   noisy_predictions = model.predict(X_noisy)

   stability = np.mean(val_predictions == noisy_predictions)
   print(f"Stability Score: {stability:.3f}")
   ```

7. **Feature Alignment for New Data**
   ```python
   # Save feature names
   feature_names = X_train.columns.tolist()
   joblib.dump(feature_names, 'anomaly_features.pkl')

   # Inference function
   def detect_anomalies_new(new_df):
       # Load artifacts
       model = joblib.load('anomaly_model.pkl')
       scaler = joblib.load('anomaly_scaler.pkl')
       features = joblib.load('anomaly_features.pkl')

       # Engineer features (same as training)
       X_new = engineer_features(new_df)

       # Ensure same feature order
       X_new = X_new[features]

       # Scale
       X_scaled = scaler.transform(X_new)

       # Predict
       predictions = model.predict(X_scaled)
       scores = model.decision_function(X_scaled)

       return predictions, scores
   ```

#### Output Format
```csv
ticket_id,timestamp,district,type,anomaly_label,anomaly_score,reason
12345,2024-09-06 03:00,Pathumwan,flood,-1,-0.25,Unusual hour for flood complaints
12346,2024-09-06 14:00,Vadhana,traffic,1,0.15,Normal traffic complaint
```

---

### 4.3 K-Means Outage Clustering

**File:** [ml_models/outage_model/train_outage_model.py](ml_models/outage_model/train_outage_model.py)

#### Model Architecture

```
Input Features:
  - Time: day_of_week (0-6), start_min (minutes since midnight)
  - Duration: outage duration in minutes
  - Weather: temperature, rainfall, wind_gust
  - Location: district (one-hot encoded)

Preprocessing:
  - OneHotEncoder for categorical (district)
  - StandardScaler for numerical

Model: K-Means Clustering
  - n_clusters: Optimized via Silhouette score (range: 2-8)
  - init: 'k-means++'
  - n_init: 10

Output:
  - Cluster labels (0, 1, 2, ...)
  - Cluster centers
  - Cluster characteristics
```

#### Training Pipeline

1. **Feature Engineering**
   ```python
   # Load MEA cleaned data
   df = pd.read_csv('clean_scraping_data.csv')

   # Time-based features
   df['date'] = pd.to_datetime(df['date'])
   df['day_of_week'] = df['date'].dt.dayofweek

   # Convert start time to minutes
   df['start_hour'] = pd.to_datetime(df['start'], format='%H:%M').dt.hour
   df['start_minute'] = pd.to_datetime(df['start'], format='%H:%M').dt.minute
   df['start_min'] = df['start_hour'] * 60 + df['start_minute']

   # Duration already calculated in cleaning phase
   # Weather data already merged from enrichment phase

   # Select features
   feature_cols = [
       'day_of_week',
       'start_min',
       'duration',
       'temp',
       'rain',
       'wind_gust',
       'district'  # Categorical
   ]
   ```

2. **Preprocessing Pipeline**
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler

   # Define transformers
   categorical_features = ['district']
   numerical_features = ['day_of_week', 'start_min', 'duration', 'temp', 'rain', 'wind_gust']

   preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), numerical_features),
           ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
       ]
   )

   X = df[feature_cols]
   X_transformed = preprocessor.fit_transform(X)
   ```

3. **Optimal K Selection**
   ```python
   from sklearn.metrics import silhouette_score

   # Test range of k values
   silhouette_scores = []
   K_range = range(2, 9)

   for k in K_range:
       kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
       cluster_labels = kmeans.fit_predict(X_transformed)

       score = silhouette_score(X_transformed, cluster_labels)
       silhouette_scores.append(score)

       print(f"k={k}: Silhouette Score = {score:.3f}")

   # Select best k
   best_k = K_range[np.argmax(silhouette_scores)]
   print(f"Optimal k: {best_k}")
   ```

4. **Model Training**
   ```python
   from sklearn.cluster import KMeans

   # Train final model with best k
   best_kmeans = KMeans(
       n_clusters=best_k,
       init='k-means++',
       n_init=10,
       max_iter=300,
       random_state=42
   )

   cluster_labels = best_kmeans.fit_predict(X_transformed)
   df['cluster'] = cluster_labels
   ```

5. **Train/Test Split Evaluation**
   ```python
   from sklearn.model_selection import train_test_split

   # Split data
   X_train, X_test = train_test_split(X_transformed, test_size=0.2, random_state=42)

   # Train on training set
   kmeans_train = KMeans(n_clusters=best_k, random_state=42)
   kmeans_train.fit(X_train)

   # Predict on test set
   test_clusters = kmeans_train.predict(X_test)

   # Evaluate consistency
   test_silhouette = silhouette_score(X_test, test_clusters)
   print(f"Test Silhouette Score: {test_silhouette:.3f}")
   ```

6. **Cluster Characterization**
   ```python
   # Aggregate cluster characteristics
   cluster_summary = df.groupby('cluster').agg({
       'day_of_week': lambda x: x.mode()[0],      # Most common day
       'start_min': 'mean',                        # Average start time
       'duration': 'mean',                         # Average duration
       'temp': 'mean',                             # Average temperature
       'rain': 'mean',                             # Average rainfall
       'wind_gust': 'mean',                        # Average wind gust
       'district': lambda x: x.mode()[0]          # Most common district
   }).reset_index()

   # Cluster sizes
   cluster_counts = df['cluster'].value_counts().sort_index()
   cluster_summary['count'] = cluster_counts.values

   print(cluster_summary)
   ```

7. **Model & Artifacts Export**
   ```python
   import joblib

   # Save model
   joblib.dump(best_kmeans, 'outage_clustering_model.pkl')

   # Save preprocessor
   joblib.dump(preprocessor, 'outage_preprocessor.pkl')

   # Save cluster summary
   cluster_summary.to_csv('cluster_summary.csv', index=False)

   # Save clustered data for visualization
   df.to_csv('clustered_outages.csv', index=False)
   ```

#### Cluster Interpretation Example
```
Cluster 0: "Weekday Morning Outages"
  - Day: Monday-Friday (mode: 1)
  - Start: ~480 min (08:00 AM)
  - Duration: ~300 min (5 hours)
  - District: Central Bangkok
  - Weather: Normal (temp ~30°C, low rain)

Cluster 1: "Storm-Related Outages"
  - Day: All days
  - Start: Variable
  - Duration: ~120 min (2 hours)
  - District: Various
  - Weather: High rainfall (>20mm), high wind gusts

Cluster 2: "Weekend Maintenance"
  - Day: Saturday-Sunday (mode: 6)
  - Start: ~540 min (09:00 AM)
  - Duration: ~240 min (4 hours)
  - District: Outer districts
  - Weather: Normal
```

---

## Phase 5: Visualization & Dashboard

### 5.1 Main Dashboard Architecture

**File:** [visualization/main_dashboard.py](visualization/main_dashboard.py)

#### Technology Stack
- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **Folium**: Map rendering with Leaflet.js
- **PyDeck**: 3D visualizations
- **Pandas**: Data manipulation

#### Dashboard Structure

```python
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
import pydeck as pdk

# Page configuration
st.set_page_config(
    page_title="Urban Issue Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Date range
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date)
    )

    # District selection
    districts = st.multiselect(
        "Districts",
        options=sorted(df['district'].unique()),
        default=[]
    )

    # Complaint type
    types = st.multiselect(
        "Complaint Types",
        options=sorted(df['type'].unique()),
        default=[]
    )

# Apply filters
filtered_df = df[
    (df['date'] >= date_range[0]) &
    (df['date'] <= date_range[1])
]

if districts:
    filtered_df = filtered_df[filtered_df['district'].isin(districts)]

if types:
    filtered_df = filtered_df[filtered_df['type'].isin(types)]
```

---

### 5.2 Tab 1: Geospatial Visualization

#### 5.2.1 HeatMap (Complaint Density)

**Lines 230-260 in main_dashboard.py**

```python
def create_heatmap(df, max_points=10000):
    """Create Folium HeatMap for complaint density"""
    import folium
    from folium.plugins import HeatMap

    # Sample data if too large
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df

    # Create base map (centered on Bangkok)
    m = folium.Map(
        location=[13.75, 100.50],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Prepare heat data
    heat_data = df_sample[['latitude', 'longitude']].values.tolist()

    # Add HeatMap layer
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        max_zoom=15,
        gradient={
            0.0: 'blue',
            0.25: 'green',
            0.5: 'yellow',
            0.75: 'orange',
            1.0: 'red'
        }
    ).add_to(m)

    return m

# Display in Streamlit
st.subheader("Complaint Density HeatMap")
heatmap = create_heatmap(filtered_df)
st_folium(heatmap, width=1200, height=600)
```

#### 5.2.2 Marker Clusters (Interactive Points)

**Lines 261-285 in main_dashboard.py**

```python
def create_marker_cluster(df, max_markers=1000):
    """Create Folium MarkerCluster for individual complaints"""
    from folium.plugins import MarkerCluster

    # Sample data
    df_sample = df.sample(n=min(len(df), max_markers), random_state=42)

    # Create base map
    m = folium.Map(
        location=[13.75, 100.50],
        zoom_start=11
    )

    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers
    for idx, row in df_sample.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"""
                <b>District:</b> {row['district']}<br>
                <b>Type:</b> {row['type']}<br>
                <b>Date:</b> {row['date']}<br>
                <b>Status:</b> {row['state']}<br>
                <b>Solve Days:</b> {row['solve_days']:.1f}
            """,
            tooltip=f"{row['type']} in {row['district']}"
        ).add_to(marker_cluster)

    return m

# Display
st.subheader("Interactive Marker Clusters")
cluster_map = create_marker_cluster(filtered_df)
st_folium(cluster_map, width=1200, height=600)
```

#### 5.2.3 3D Grid Layer (PyDeck)

**Lines 286-302 in main_dashboard.py**

```python
def create_3d_grid(df):
    """Create PyDeck 3D GridLayer for complaint concentration"""

    # Prepare data
    data = df[['latitude', 'longitude']].rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Create PyDeck layer
    layer = pdk.Layer(
        'GridLayer',
        data=data,
        get_position=['lon', 'lat'],
        cell_size=200,              # Cell size in meters
        elevation_scale=50,         # Height multiplier
        elevation_range=[0, 1000],
        extruded=True,              # 3D extrusion
        pickable=True,
        auto_highlight=True,
        coverage=1
    )

    # Set view state
    view_state = pdk.ViewState(
        latitude=13.75,
        longitude=100.50,
        zoom=11,
        pitch=45,                   # Tilt angle (0-60)
        bearing=0
    )

    # Create deck
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            'html': '<b>Complaint Count:</b> {count}',
            'style': {'color': 'white'}
        }
    )

    return deck

# Display
st.subheader("3D Complaint Density Grid")
grid_3d = create_3d_grid(filtered_df)
st.pydeck_chart(grid_3d)
```

---

### 5.3 Tab 2: District & Type Analysis

**File:** [visualization/viz_modules.py](visualization/viz_modules.py)

#### 5.3.1 Top Districts (Bar Chart)

```python
def plot_top_districts(df, top_n=10):
    """Bar chart of top N districts by complaint count"""

    # Aggregate by district
    district_counts = df['district'].value_counts().head(top_n).reset_index()
    district_counts.columns = ['district', 'count']

    # Create bar chart
    fig = px.bar(
        district_counts,
        x='district',
        y='count',
        title=f'Top {top_n} Districts by Complaint Count',
        labels={'district': 'District', 'count': 'Number of Complaints'},
        color='count',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    return fig

# Display
st.plotly_chart(plot_top_districts(filtered_df), use_container_width=True)
```

#### 5.3.2 Stacked Bar (Types within Districts)

```python
def plot_district_type_breakdown(df, top_districts=10):
    """Stacked bar chart of complaint types within top districts"""

    # Get top districts
    top_dist = df['district'].value_counts().head(top_districts).index

    # Filter and aggregate
    df_top = df[df['district'].isin(top_dist)]
    grouped = df_top.groupby(['district', 'type']).size().reset_index(name='count')

    # Create stacked bar
    fig = px.bar(
        grouped,
        x='district',
        y='count',
        color='type',
        title='Complaint Type Breakdown by District',
        labels={'count': 'Number of Complaints', 'type': 'Complaint Type'},
        barmode='stack'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        legend_title_text='Complaint Type'
    )

    return fig

# Display
st.plotly_chart(plot_district_type_breakdown(filtered_df), use_container_width=True)
```

#### 5.3.3 Time-Series (Multi-District Comparison)

```python
def plot_time_series_by_district(df, districts):
    """Line chart comparing complaint trends across districts"""

    # Filter districts
    df_filtered = df[df['district'].isin(districts)]

    # Aggregate by date and district
    daily = df_filtered.groupby(['date', 'district']).size().reset_index(name='count')

    # Create line chart
    fig = px.line(
        daily,
        x='date',
        y='count',
        color='district',
        title='Complaint Trends by District',
        labels={'date': 'Date', 'count': 'Daily Complaints', 'district': 'District'}
    )

    fig.update_layout(
        hovermode='x unified',
        height=500
    )

    return fig

# Display with district selector
selected_districts = st.multiselect(
    "Select Districts to Compare",
    options=sorted(filtered_df['district'].unique()),
    default=filtered_df['district'].value_counts().head(5).index.tolist()
)

st.plotly_chart(
    plot_time_series_by_district(filtered_df, selected_districts),
    use_container_width=True
)
```

#### 5.3.4 Hourly Patterns (Heatmap-style)

```python
def plot_hourly_patterns(df):
    """Heatmap showing complaint patterns by hour and day of week"""

    # Extract hour and day
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_name'] = pd.to_datetime(df['timestamp']).dt.day_name()

    # Aggregate
    hourly = df.groupby(['day_name', 'hour']).size().reset_index(name='count')

    # Pivot for heatmap
    heatmap_data = hourly.pivot(index='day_name', columns='hour', values='count')

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x='Hour of Day', y='Day of Week', color='Complaints'),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title='Complaint Patterns by Hour and Day',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )

    fig.update_layout(height=500)

    return fig

# Display
st.plotly_chart(plot_hourly_patterns(filtered_df), use_container_width=True)
```

---

### 5.4 Tab 3: MEA Power Outage Data

**File:** [visualization/outage_viz.py](visualization/outage_viz.py)

#### 5.4.1 Outage Distribution by District

```python
def plot_outage_distribution(df):
    """Bar chart of outages by district"""

    district_outages = df['district'].value_counts().reset_index()
    district_outages.columns = ['district', 'outage_count']

    fig = px.bar(
        district_outages,
        x='district',
        y='outage_count',
        title='Power Outage Distribution by District',
        labels={'district': 'District', 'outage_count': 'Number of Outages'},
        color='outage_count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(xaxis_tickangle=-45, height=500)

    return fig

st.plotly_chart(plot_outage_distribution(mea_df), use_container_width=True)
```

#### 5.4.2 Weather Correlation (Scatter)

```python
def plot_weather_correlation(df):
    """Scatter plot showing outage duration vs weather factors"""

    fig = px.scatter(
        df,
        x='rain',
        y='duration',
        color='wind_gust',
        size='temp',
        hover_data=['district', 'date'],
        title='Outage Duration vs Weather Conditions',
        labels={
            'rain': 'Rainfall (mm)',
            'duration': 'Outage Duration (minutes)',
            'wind_gust': 'Wind Gust (km/h)',
            'temp': 'Temperature (°C)'
        },
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=600)

    return fig

st.plotly_chart(plot_weather_correlation(mea_df), use_container_width=True)
```

---

### 5.5 Tab 4: ML Predictive Forecasting

#### 5.5.1 Forecast Visualization

```python
def plot_forecast(historical_df, forecast_dict):
    """Line chart showing historical data + forecast"""

    # Prepare historical data
    historical = historical_df.groupby('date').size().reset_index(name='count')
    historical['type'] = 'Historical'

    # Prepare forecast data
    forecast_df = pd.DataFrame({
        'date': forecast_dict['forecast_dates'],
        'count': forecast_dict['predicted_counts'],
        'type': 'Forecast'
    })

    # Combine
    combined = pd.concat([historical, forecast_df], ignore_index=True)

    # Create line chart
    fig = px.line(
        combined,
        x='date',
        y='count',
        color='type',
        title='Complaint Forecast (7-Day Ahead)',
        labels={'date': 'Date', 'count': 'Number of Complaints', 'type': ''},
        line_dash='type',
        line_dash_map={'Historical': 'solid', 'Forecast': 'dash'}
    )

    # Add confidence intervals
    if 'lower_bound' in forecast_dict:
        fig.add_scatter(
            x=forecast_dict['forecast_dates'],
            y=forecast_dict['lower_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )

        fig.add_scatter(
            x=forecast_dict['forecast_dates'],
            y=forecast_dict['upper_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 100, 250, 0.2)',
            name='Confidence Interval'
        )

    fig.update_layout(height=500)

    return fig

# Load model and forecast
if st.button("Generate Forecast"):
    with st.spinner("Running forecasting model..."):
        forecast_result = run_forecast_model(filtered_df)
        st.session_state['forecast'] = forecast_result

    st.success("Forecast complete!")

if 'forecast' in st.session_state:
    st.plotly_chart(
        plot_forecast(filtered_df, st.session_state['forecast']),
        use_container_width=True
    )
```

#### 5.5.2 Model Performance Metrics

```python
def display_forecast_metrics(metrics_dict):
    """Display model evaluation metrics"""

    st.subheader("Model Performance")

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall MAE", f"{metrics_dict['mae']:.2f}")

    with col2:
        st.metric("Overall RMSE", f"{metrics_dict['rmse']:.2f}")

    with col3:
        st.metric("Overall MAPE", f"{metrics_dict['mape']:.2f}%")

    # Per-horizon metrics
    st.subheader("Per-Horizon Performance")

    horizon_df = pd.DataFrame(metrics_dict['per_horizon'])

    fig = px.bar(
        horizon_df,
        x='horizon',
        y='mae',
        title='MAE by Forecast Horizon',
        labels={'horizon': 'Days Ahead', 'mae': 'Mean Absolute Error'},
        color='mae',
        color_continuous_scale='Oranges'
    )

    st.plotly_chart(fig, use_container_width=True)

# Display
if 'forecast' in st.session_state:
    display_forecast_metrics(st.session_state['forecast']['metrics'])
```

---

### 5.6 Tab 5: ML Anomaly Detection

#### 5.6.1 Anomaly Scatter Plot

```python
def plot_anomalies(df):
    """Scatter plot highlighting detected anomalies"""

    # Separate normal and anomalies
    normal = df[df['anomaly_label'] == 1]
    anomalies = df[df['anomaly_label'] == -1]

    # Create figure
    fig = go.Figure()

    # Add normal points
    fig.add_trace(go.Scatter(
        x=normal['longitude'],
        y=normal['latitude'],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.3),
        name='Normal',
        text=normal['type'],
        hovertemplate='<b>%{text}</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}'
    ))

    # Add anomaly points
    fig.add_trace(go.Scatter(
        x=anomalies['longitude'],
        y=anomalies['latitude'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Anomaly',
        text=anomalies['type'],
        hovertemplate='<b>ANOMALY: %{text}</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}'
    ))

    fig.update_layout(
        title='Detected Anomalies (Geographic Distribution)',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=600
    )

    return fig

# Run anomaly detection
if st.button("Detect Anomalies"):
    with st.spinner("Running anomaly detection..."):
        anomaly_df = run_anomaly_detection(filtered_df)
        st.session_state['anomalies'] = anomaly_df

    st.success(f"Detected {len(anomaly_df[anomaly_df['anomaly_label'] == -1])} anomalies!")

if 'anomalies' in st.session_state:
    st.plotly_chart(
        plot_anomalies(st.session_state['anomalies']),
        use_container_width=True
    )
```

#### 5.6.2 Anomaly Distribution

```python
def plot_anomaly_distribution(df):
    """Bar charts showing anomaly distribution"""

    anomalies = df[df['anomaly_label'] == -1]

    # By type
    type_dist = anomalies['type'].value_counts().reset_index()
    type_dist.columns = ['type', 'count']

    fig1 = px.bar(
        type_dist,
        x='type',
        y='count',
        title='Anomalies by Complaint Type',
        labels={'type': 'Complaint Type', 'count': 'Anomaly Count'},
        color='count',
        color_continuous_scale='Reds'
    )

    # By district
    district_dist = anomalies['district'].value_counts().head(10).reset_index()
    district_dist.columns = ['district', 'count']

    fig2 = px.bar(
        district_dist,
        x='district',
        y='count',
        title='Top 10 Districts with Anomalies',
        labels={'district': 'District', 'count': 'Anomaly Count'},
        color='count',
        color_continuous_scale='Oranges'
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

# Display
if 'anomalies' in st.session_state:
    plot_anomaly_distribution(st.session_state['anomalies'])
```

---

### 5.7 Tab 6: ML Power Outage Clustering

**File:** [visualization/outage_viz.py](visualization/outage_viz.py)

#### 5.7.1 Cluster Distribution

```python
def plot_cluster_distribution(df):
    """Bar chart showing cluster sizes"""

    cluster_counts = df['cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['cluster', 'count']

    fig = px.bar(
        cluster_counts,
        x='cluster',
        y='count',
        title='Power Outage Cluster Distribution',
        labels={'cluster': 'Cluster ID', 'count': 'Number of Outages'},
        color='count',
        color_continuous_scale='Viridis',
        text='count'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)

    return fig

# Run clustering
if st.button("Run Clustering"):
    with st.spinner("Clustering outage patterns..."):
        clustered_df = run_outage_clustering(mea_df)
        st.session_state['clustered'] = clustered_df

    st.success("Clustering complete!")

if 'clustered' in st.session_state:
    st.plotly_chart(
        plot_cluster_distribution(st.session_state['clustered']),
        use_container_width=True
    )
```

#### 5.7.2 Cluster Characteristics (6-Subplot Layout)

```python
def plot_cluster_characteristics(df):
    """6-subplot figure showing cluster characteristics"""

    from plotly.subplots import make_subplots

    # Create 2x3 subplot grid
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            'Start Time Distribution',
            'Duration Distribution',
            'Day of Week',
            'Temperature',
            'Rainfall',
            'Wind Gust'
        )
    )

    clusters = sorted(df['cluster'].unique())

    # 1. Start time (minutes since midnight)
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(
            go.Histogram(
                x=cluster_data['start_min'],
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=True
            ),
            row=1, col=1
        )

    # 2. Duration
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(
            go.Histogram(
                x=cluster_data['duration'],
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Day of week
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        day_counts = cluster_data['day_of_week'].value_counts().sort_index()

        fig.add_trace(
            go.Bar(
                x=[day_names[i] for i in day_counts.index],
                y=day_counts.values,
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=False
            ),
            row=1, col=3
        )

    # 4. Temperature
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(
            go.Box(
                y=cluster_data['temp'],
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=False
            ),
            row=2, col=1
        )

    # 5. Rainfall
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(
            go.Box(
                y=cluster_data['rain'],
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=False
            ),
            row=2, col=2
        )

    # 6. Wind gust
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]

        fig.add_trace(
            go.Box(
                y=cluster_data['wind_gust'],
                name=f'Cluster {cluster}',
                legendgroup=f'cluster_{cluster}',
                showlegend=False
            ),
            row=2, col=3
        )

    fig.update_layout(
        height=800,
        title_text="Cluster Characteristics Overview",
        showlegend=True
    )

    # Update axes labels
    fig.update_xaxes(title_text="Minutes since midnight", row=1, col=1)
    fig.update_xaxes(title_text="Duration (minutes)", row=1, col=2)
    fig.update_xaxes(title_text="Day of Week", row=1, col=3)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Rainfall (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Wind Gust (km/h)", row=2, col=3)

    return fig

# Display
if 'clustered' in st.session_state:
    st.plotly_chart(
        plot_cluster_characteristics(st.session_state['clustered']),
        use_container_width=True
    )
```

#### 5.7.3 Cluster Summary Table

```python
def display_cluster_summary(df):
    """Display cluster summary table"""

    st.subheader("Cluster Summary")

    summary = df.groupby('cluster').agg({
        'day_of_week': lambda x: x.mode()[0],
        'start_min': 'mean',
        'duration': 'mean',
        'temp': 'mean',
        'rain': 'mean',
        'wind_gust': 'mean',
        'district': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
    }).reset_index()

    # Add count
    summary['count'] = df['cluster'].value_counts().sort_index().values

    # Format columns
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    summary['day_of_week'] = summary['day_of_week'].apply(lambda x: day_names[int(x)])
    summary['start_min'] = summary['start_min'].apply(lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}")
    summary['duration'] = summary['duration'].apply(lambda x: f"{x:.1f} min")
    summary['temp'] = summary['temp'].apply(lambda x: f"{x:.1f}°C")
    summary['rain'] = summary['rain'].apply(lambda x: f"{x:.2f} mm")
    summary['wind_gust'] = summary['wind_gust'].apply(lambda x: f"{x:.1f} km/h")

    # Rename columns
    summary.columns = [
        'Cluster',
        'Most Common Day',
        'Avg Start Time',
        'Avg Duration',
        'Avg Temperature',
        'Avg Rainfall',
        'Avg Wind Gust',
        'Most Common District',
        'Count'
    ]

    st.dataframe(summary, use_container_width=True)

# Display
if 'clustered' in st.session_state:
    display_cluster_summary(st.session_state['clustered'])
```

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         START: DATA SOURCES                              │
│  ┌──────────────────────┐              ┌─────────────────────┐          │
│  │ Bangkok Traffy       │              │ MEA Website         │          │
│  │ Database             │              │ Power Outage Alerts │          │
│  └──────────┬───────────┘              └──────────┬──────────┘          │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              │ Direct                            │ Web
              │ Export                            │ Scraping
              │                                   │
              ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PHASE 1: DATA COLLECTION                           │
│  ┌──────────────────────┐              ┌─────────────────────┐          │
│  │ traffy_raw.csv       │              │ Selenium + BS4      │          │
│  │ • 780K+ records      │              │ • 12 parallel       │          │
│  │ • Direct import      │              │   workers           │          │
│  │ • Structured         │              │ • Pagination        │          │
│  └──────────┬───────────┘              │ • Day-based         │          │
│             │                          │   records           │          │
│             │                          └──────────┬──────────┘          │
│             │                                     │                      │
│             │                                     ▼                      │
│             │                          ┌─────────────────────┐          │
│             │                          │ mea_page_*.csv      │          │
│             │                          │ • Unstructured text │          │
│             │                          │ • Batch exports     │          │
│             │                          └──────────┬──────────┘          │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              │                                   ▼
              │                        ┌─────────────────────────┐
              │                        │ PHASE 2: LLM EXTRACTION │
              │                        │ • Gemini API            │
              │                        │ • Pydantic validation   │
              │                        │ • District inference    │
              │                        │ • Weather enrichment    │
              │                        └──────────┬──────────────┘
              │                                   │
              │                                   ▼
              │                        ┌─────────────────────────┐
              │                        │ mea_structured.csv      │
              │                        │ • Structured events     │
              │                        │ • Geographic coords     │
              │                        │ • Weather data          │
              │                        └──────────┬──────────────┘
              │                                   │
              ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: DATA CLEANING & ETL                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Jupyter Notebooks                                                   ││
│  │                                                                     ││
│  │ clean_data_traffy.ipynb              clean_data_scarping.ipynb     ││
│  │ • Missing value imputation           • Date validation             ││
│  │ • Timestamp parsing                  • Time standardization        ││
│  │ • One-hot encoding                   • Duration calculation        ││
│  │ • Geo validation                     • District normalization      ││
│  │ • Deduplication                      • Deduplication               ││
│  │                                                                     ││
│  │                   merge_file.ipynb                                 ││
│  │                   • Schema alignment                               ││
│  │                   • Temporal join                                  ││
│  │                   • Feature engineering                            ││
│  └─────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CLEAN DATA OUTPUTS                                 │
│  ┌──────────────────────┐              ┌─────────────────────┐          │
│  │ clean_data.csv       │              │ clean_scraping      │          │
│  │ (Traffy)             │              │ _data.csv (MEA)     │          │
│  └──────────┬───────────┘              └──────────┬──────────┘          │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: MACHINE LEARNING                             │
│                                                                          │
��  ┌────────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ ML 1: Forecasting  │  │ ML 2: Anomaly  │  │ ML 3: Clustering     │  │
│  │                    │  │  Detection     │  │                      │  │
│  │ RandomForest       │  │ Isolation      │  │ K-Means              │  │
│  │ • TimeSeriesSplit  │  │  Forest        │  │ • Silhouette opt     │  │
│  │ • GridSearchCV     │  │ • StandardScal │  │ • Train/test split   │  │
│  │ • 30-day lookback  │  │ • 5% contamin  │  │ • Weather features   │  │
│  │ • 7-day forecast   │  │ • Multi-featur │  │ • Time patterns      │  │
│  │                    │  │                │  │                      │  │
│  │ Output:            │  │ Output:        │  │ Output:              │  │
│  │ • .pkl model       │  │ • .pkl model   │  │ • .pkl model         │  │
│  │ • Forecast CSV     │  │ • Anomaly CSV  │  │ • Clustered CSV      │  │
│  │ • Metrics JSON     │  │ • Metrics JSON │  │ • Summary CSV        │  │
│  └────────────────────┘  └────────────────┘  └──────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE 5: VISUALIZATION & DASHBOARD                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Streamlit Dashboard (main_dashboard.py)         │ │
│  │                                                                    │ │
│  │  [Sidebar: Filters]                                               │ │
│  │    - Date range picker                                            │ │
│  │    - District multiselect                                         │ │
│  │    - Type multiselect                                             │ │
│  │                                                                    │ │
│  │  [Tab 1: Geospatial]          [Tab 2: District Analysis]          │ │
│  │    • Folium HeatMap              • Bar charts (top districts)     │ │
│  │    • Marker Clusters             • Stacked bars (type breakdown)  │ │
│  │    • PyDeck 3D Grid              • Time-series (multi-district)   │ │
│  │                                  • Hourly patterns heatmap        │ │
│  │                                                                    │ │
│  │  [Tab 3: MEA Outages]          [Tab 4: ML Forecasting]            │ │
│  │    • Outage distribution         • Forecast line chart            │ │
│  │    • Weather correlation         • Confidence intervals           │ │
│  │                                  • Performance metrics            │ │
│  │                                  • Per-horizon MAE                │ │
│  │                                                                    │ │
│  │  [Tab 5: Anomaly Detection]    [Tab 6: Clustering]                │ │
│  │    • Anomaly scatter plot        • Cluster distribution           │ │
│  │    • Distribution by type        • 6-subplot characteristics      │ │
│  │    • Distribution by district    • Cluster summary table          │ │
│  │                                                                    │ │
│  │  [Tab 7: Additional Analysis]                                     │ │
│  │    • Custom analytics                                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   END: USER ACCESS  │
                         │   Web Dashboard     │
                         │   (localhost:8501)  │
                         └─────────────────────┘
```

---

## Technology Stack

### Data Collection Layer
- **Selenium WebDriver**: Browser automation for dynamic content
- **BeautifulSoup4**: HTML parsing
- **Requests**: HTTP client
- **concurrent.futures**: Parallel processing
- **Pandas**: Data structuring

### Data Processing Layer
- **Google Gemini API**: LLM-based extraction (gemini-2.5-flash-lite)
- **Instructor**: Type-safe LLM outputs
- **Pydantic**: Data validation
- **OpenMeteo API**: Weather data
- **Jupyter Notebooks**: ETL workflows

### Machine Learning Layer
- **scikit-learn**: ML algorithms
  - RandomForestRegressor
  - IsolationForest
  - KMeans
  - TimeSeriesSplit
  - GridSearchCV
- **NumPy**: Numerical operations
- **joblib**: Model persistence

### Visualization Layer
- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **Folium**: Map rendering (Leaflet.js backend)
- **PyDeck**: 3D visualizations (deck.gl backend)
- **streamlit-folium**: Streamlit-Folium integration

### Data Storage
- **CSV files**: Primary data format
- **Pickle (.pkl)**: Model serialization
- **JSON**: Configuration & metadata

---

## Running the Complete Pipeline

### Step 1: Data Collection
```bash
# Scrape MEA data
cd "Scraper_new/MEA Scraper"
python mea_outage_scraper_v2.py --max-pages 50 --workers 12
```

### Step 2: Data Extraction
```bash
# Extract structured data with Gemini
cd "../Python Pipeline"
python Pipeline_Gemini.py
```

### Step 3: Data Cleaning
```bash
# Run Jupyter notebooks
cd "../../clean_data"
jupyter notebook clean_data_traffy.ipynb
jupyter notebook clean_data_scarping.ipynb
jupyter notebook merge_file.ipynb
```

### Step 4: Train ML Models
```bash
# Train all models
cd ../ml_models

# Forecasting
cd forecasting
python train_lstm_model.py

# Anomaly detection
cd ../anomaly_detection
python detect_anomalies.py

# Clustering
cd ../outage_model
python train_outage_model.py
```

### Step 5: Launch Dashboard
```bash
# Start Streamlit app
cd ../../visualization
streamlit run main_dashboard.py
```

Access dashboard at: http://localhost:8501

---

## Summary

This project implements a complete data science pipeline:

1. **Data Collection**: Automated web scraping + API access
2. **Data Processing**: LLM-based extraction + weather enrichment
3. **Data Integration**: Multi-source ETL with quality checks
4. **Machine Learning**: 3 production models (forecasting, anomaly detection, clustering)
5. **Visualization**: Interactive dashboard with 7 analysis tabs

**Total Records**: 780,000+
**Data Sources**: 2 (Bangkok Traffy + MEA)
**ML Models**: 3 (RandomForest, Isolation Forest, K-Means)
**Visualizations**: 15+ interactive charts and maps
**Technologies**: 20+ libraries and frameworks

The system is production-ready with proper error handling, logging, model persistence, and comprehensive documentation.
