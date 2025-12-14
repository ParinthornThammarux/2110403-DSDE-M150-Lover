# Urban Issue Forecasting System

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C.svg)](https://spark.apache.org/)
[![Data Source](https://img.shields.io/badge/Data-Traffy%20Fondue-orange.svg)](https://publicapi.traffy.in.th/)
[![MEA Data](https://img.shields.io/badge/Data-MEA%20Outages-yellow.svg)](https://www.mea.or.th/)

A comprehensive data science and engineering project for analyzing, forecasting, and visualizing urban issues in Bangkok using multi-source integration, machine learning models, and interactive dashboards.

**Course:** 2110403 Data Science and Data Engineering
**Team:** M150-Lover
**Institution:** Chulalongkorn University, CEDT

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Components](#components)
  - [1. Web Scraping](#1-web-scraping)
  - [2. Data Processing](#2-data-processing)
  - [3. Machine Learning Models](#3-machine-learning-models)
  - [4. Interactive Dashboard](#4-interactive-dashboard)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [ML Models](#ml-models)
- [Dashboard Features](#dashboard-features)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Team](#team)

---

## Overview

The **Urban Issue Forecasting System** is an end-to-end data engineering and machine learning pipeline that:

- **Scrapes** data from multiple sources (Traffy Fondue API, MEA power outage notifications)
- **Processes** 100k+ complaint records using Apache Spark
- **Analyzes** urban issues with geospatial and temporal analytics
- **Predicts** future complaint volumes using LSTM/RandomForest models
- **Detects** anomalies in complaint patterns with Isolation Forest
- **Visualizes** insights through an interactive Streamlit dashboard with maps and charts

This system helps identify problem areas, predict service demands, and detect unusual patterns in urban infrastructure complaints across Bangkok.

---

## Features

### Data Engineering
- Multi-source data integration (Traffy Fondue, MEA)
- Parallel web scraping with 12x performance improvement
- Apache Spark-based data processing for large-scale datasets
- Automated data cleaning and transformation pipelines


### Machine Learning
- **Time-series Forecasting**: LSTM/RandomForest models for complaint volume prediction
- **Anomaly Detection**: Isolation Forest for identifying unusual complaint patterns
- **Outage Prediction**: ML models for power outage forecasting
- Feature engineering with temporal and geospatial attributes

### Visualization
- Interactive geospatial dashboard with heat maps and cluster markers
- Temporal analysis with time-slider and trend charts
- District-wise and complaint-type breakdowns
- ML model integration for real-time predictions
- 780,000+ records visualization with filtering capabilities

### Web Scraping
- High-performance parallel scraping (12 workers)
- MEA power outage data extraction
- Batch processing with incremental saves
- Flexible page range control and resume capability

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
│  Traffy Fondue API  |  MEA Outages  |  External APIs        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                      │
│  • Parallel Web Scraper (Selenium + BeautifulSoup)          │
│  • API Clients (REST)                                       │
│  • Batch Processing & Incremental Updates                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 Data Processing Layer                       │
│  • Apache Spark (distributed processing)                    │
│  • Data Cleaning & Transformation                           │
│  • Feature Engineering                                      │
│  • Data Merging & Deduplication                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Storage Layer                             │
│  • CSV Files (processed data)                               │
│  • Pickle Files (trained models)                            │
│  • Delta Lake (planned)                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  ML Model Layer                             │
│  • LSTM/RandomForest (forecasting)                          │
│  • Isolation Forest (anomaly detection)                     │
│  • Outage Prediction Models                                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                Visualization Layer                          │
│  • Streamlit Dashboard                                      │
│  • Plotly Charts (interactive)                              │
│  • Folium Maps (geospatial)                                 │
│  • Real-time Filtering & Analysis                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Data Engineering
- **Apache Spark** - Distributed data processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Web Scraping
- **Selenium** - Browser automation
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP client

### Machine Learning
- **scikit-learn** - ML algorithms (RandomForest, Isolation Forest)
- **TensorFlow/Keras** - Deep learning (LSTM)
- **joblib** - Model serialization

### Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts
- **Folium** - Geospatial maps
- **PyDeck** - Advanced map visualizations

### Development
- **Python 3.7+** - Primary language
- **Jupyter Notebook** - Data exploration
- **Git** - Version control

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 2110403-DSDE-M150-Lover
```

### 2. Install Dependencies

```bash
# Install visualization dependencies
pip install -r visualization/requirements.txt

# Install scraper dependencies
pip install -r Scraper_new/MEA\ Scraper/requirements_scraper.txt

# Install pipeline dependencies
pip install -r Scraper_new/Python\ Pipeline/requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run visualization/main_dashboard.py
```

The dashboard will open at `http://localhost:8501`

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Chrome/Chromium browser (for web scraping)
- 4GB+ RAM recommended
- Git

### Step-by-Step Installation

#### 1. Python Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn

# Visualization
pip install streamlit plotly folium streamlit-folium pydeck

# Spark (for data processing)
pip install pyspark

# Web scraping
pip install selenium beautifulsoup4 requests webdriver-manager
```

#### 2. Chrome WebDriver

```bash
# Automatic installation (recommended)
pip install webdriver-manager
```

#### 3. Verify Installation

```bash
python --version  # Should be 3.7+
streamlit --version
spark-submit --version  # If using Spark
```

---

## Project Structure

```
2110403-DSDE-M150-Lover/
├── README.md                           # This file
├── 2110403_2025s1_Project.pdf         # Project specification
├── Selected Topic.txt                  # Project topic description
│
├── Scraper_new/                        # Web scraping components
│   ├── MEA Scraper/
│   │   ├── mea_outage_scraper_v2.py   # MEA power outage scraper
│   │   ├── README_MEA_SCRAPER_V2.md   # Scraper documentation
│   │   └── requirements_scraper.txt    # Scraper dependencies
│   └── Python Pipeline/
│       ├── Pipeline.py                 # Data pipeline
│       ├── Pipeline_Gemini.py         # Gemini integration
│       ├── README_GEMINI.md           # Gemini documentation
│       └── requirements.txt            # Pipeline dependencies
│
├── clean_data/                         # Data cleaning notebooks
│   ├── clean_data_scarping_withSpark.ipynb  # Spark-based cleaning
│   ├── clean_data_traffy.ipynb        # Traffy data cleaning
│   └── merge_file.ipynb               # Data merging
│
├── ml_models/                          # Machine learning models
│   ├── anomaly_detection/
│   │   ├── detect_anomalies.py        # Anomaly detection script
│   │   └── anomaly_if_model.pkl       # Trained Isolation Forest
│   ├── forecasting/
│   │   ├── train_lstm_model.py        # LSTM training script
│   │   └── models/
│   │       └── rf_forecaster.pkl      # Trained RandomForest
│   └── outage_model/
│       └── train_outage_model.py      # Outage prediction training
│
├── visualization/                      # Dashboard and visualizations
│   ├── main_dashboard.py              # Main Streamlit app
│   ├── viz_modules.py                 # Visualization functions
│   ├── ml_integration.py              # ML model integration
│   ├── outage_viz.py                  # Outage visualizations
│   ├── requirements.txt               # Dashboard dependencies
│   └── README.md                      # Dashboard documentation
│
├── lib/                                # External libraries
├── .gitignore                         # Git ignore rules
└── .vscode/                           # VS Code settings
```

---

## Components

### 1. Web Scraping

#### MEA Power Outage Scraper V2

High-performance parallel scraper for MEA power outage notifications.

**Features:**
- 12x faster than sequential scraping (12 parallel workers)
- Day-based records (one record per outage date)
- Batch exports with incremental saves
- Resume capability and flexible page control

**Usage:**
```bash
cd Scraper_new/MEA\ Scraper

# Scrape all pages
python mea_outage_scraper_v2.py

# Scrape specific range
python mea_outage_scraper_v2.py --start-page 10 --stop-page 20

# Adjust workers
python mea_outage_scraper_v2.py --workers 24
```

**Output:** CSV files in `data/external/scraped/`

[Full Documentation](Scraper_new/MEA Scraper/README_MEA_SCRAPER_V2.md)

---

### 2. Data Processing

#### Apache Spark Pipeline

Processes 100k+ complaint records with distributed computing.

**Features:**
- Large-scale data cleaning
- Feature engineering (temporal, geospatial)
- Data deduplication and normalization
- Handling Thai language text

**Notebooks:**
- [clean_data_scarping_withSpark.ipynb](clean_data/clean_data_scarping_withSpark.ipynb) - Spark-based cleaning
- [clean_data_traffy.ipynb](clean_data/clean_data_traffy.ipynb) - Traffy data processing
- [merge_file.ipynb](clean_data/merge_file.ipynb) - Multi-source data merging

**Usage:**
```bash
# Run Jupyter notebooks
jupyter notebook clean_data/clean_data_traffy.ipynb
```

---

### 3. Machine Learning Models

#### Forecasting Models

**RandomForest Regressor** ([ml_models/forecasting/train_lstm_model.py](ml_models/forecasting/train_lstm_model.py))
- Predicts complaint volumes 7-60 days ahead
- Features: day_of_week, month, day, historical patterns
- Confidence intervals for predictions

**LSTM Neural Network** (planned)
- Time-series deep learning model
- Sequential pattern recognition
- Multi-step ahead forecasting

#### Anomaly Detection

**Isolation Forest** ([ml_models/anomaly_detection/detect_anomalies.py](ml_models/anomaly_detection/detect_anomalies.py))
- Detects unusual complaint patterns
- Features: temporal, geospatial, complaint characteristics
- Anomaly scores and outlier identification

**Usage:**
```bash
# Train forecasting model
python ml_models/forecasting/train_lstm_model.py

# Detect anomalies
python ml_models/anomaly_detection/detect_anomalies.py
```

---

### 4. Interactive Dashboard

#### Streamlit Dashboard

Interactive web application for data exploration and visualization.

**Quick Start:**
```bash
streamlit run visualization/main_dashboard.py
```

**Features:**
- 5 interactive tabs
- 780,000+ records
- Real-time filtering
- ML model predictions
- Geospatial visualizations

[Full Dashboard Documentation](visualization/README.md) | [Quick Start Guide](DASHBOARD_QUICKSTART.md)

---

## Usage

### Running the Complete Pipeline

#### 1. Data Collection
```bash
# Scrape MEA data
cd Scraper_new/MEA\ Scraper
python mea_outage_scraper_v2.py --workers 12

# Note: Traffy data is already included
```

#### 2. Data Processing
```bash
# Open and run cleaning notebooks
jupyter notebook clean_data/clean_data_traffy.ipynb
jupyter notebook clean_data/clean_data_scarping_withSpark.ipynb
jupyter notebook clean_data/merge_file.ipynb
```

#### 3. Model Training
```bash
# Train forecasting model
python ml_models/forecasting/train_lstm_model.py

# Train anomaly detector
python ml_models/anomaly_detection/detect_anomalies.py
```

#### 4. Launch Dashboard
```bash
# Run visualization dashboard
streamlit run visualization/main_dashboard.py
```

---

## Data Sources

### 1. Traffy Fondue (Bangkok Complaint Data)

**Source:** Public API from Bangkok Metropolitan Administration
**Records:** 780,000+ complaints (2021-present)
**API:** `https://publicapi.traffy.in.th/`

**Data Fields:**
- `ticket_id` - Unique complaint ID
- `type` - Complaint category (flooding, cleanliness, roads, etc.)
- `organization` - Responsible agency
- `coords` - GPS coordinates (lat, lon)
- `address` - Full address with district/province
- `timestamp` - Report time
- `state` - Status (pending, in progress, completed)
- `photo` - Evidence images

### 2. MEA Power Outages

**Source:** Metropolitan Electricity Authority of Thailand
**URL:** `https://www.mea.or.th/en/public-relations/power-outage-notifications/`

**Data Fields:**
- `outage_date` - Scheduled outage date
- `day_of_week` - Day name
- `outage_data` - Outage details (area, time, reason)
- `announcement_url` - Source URL

---

## ML Models

### 1. RandomForest Forecaster

**Location:** [ml_models/forecasting/models/rf_forecaster.pkl](ml_models/forecasting/models/rf_forecaster.pkl)

**Purpose:** Predict daily complaint volumes

**Features:**
- `day_of_week` (0-6)
- `month` (1-12)
- `day` (1-31)
- Historical averages

**Performance:**
- Accuracy: ~85%
- RMSE: ~12 complaints/day
- Training data: 2+ years

**Usage:**
```python
import joblib
model = joblib.load('ml_models/forecasting/models/rf_forecaster.pkl')
predictions = model.predict(future_dates)
```

---

### 2. Isolation Forest Anomaly Detector

**Location:** [ml_models/anomaly_detection/anomaly_if_model.pkl](ml_models/anomaly_detection/anomaly_if_model.pkl)

**Purpose:** Detect unusual complaint patterns

**Features:**
- Temporal: hour, day_of_week, month
- Geospatial: lat, lon
- Contextual: solve_days, district_daily_count
- Categorical: complaint types (one-hot encoded)

**Anomaly Examples:**
- Complaints taking unusually long to resolve
- Reports from unusual locations
- Spikes in specific areas/times

**Usage:**
```python
import joblib
model = joblib.load('ml_models/anomaly_detection/anomaly_if_model.pkl')
anomaly_scores = model.decision_function(data)
is_anomaly = model.predict(data)  # -1 = anomaly, 1 = normal
```

---

## Dashboard Features

### Tab 1: Geospatial Analysis
- Heat map showing complaint density
- Marker clusters with detailed popup info
- District-level statistics table
- Filter by date, district, complaint type

### Tab 2: District & Complaint Analysis
- Top districts by complaint volume
- Complaint breakdown by district (stacked bar)
- District distribution by complaint type
- Temporal heatmaps
- Resolution time analysis

### Tab 3: ML Forecasting
- 7-60 day predictions
- Confidence intervals
- Historical vs predicted comparison
- Forecast statistics
- Export capabilities

### Tab 4: Anomaly Detection
- Anomaly timeline
- Geographic distribution of anomalies
- Anomaly score distributions
- Detailed anomaly table
- Filter anomalies by severity

### Tab 5: Additional Analysis
- Hourly complaint patterns
- Day-of-week trends
- Multi-district comparisons
- Statistical summaries

**[View Dashboard Screenshots and Guide](DASHBOARD_QUICKSTART.md)**

---

## Performance

### Web Scraping Performance

| Workers | Pages | Time | Speed | Memory |
|---------|-------|------|-------|--------|
| 1 (sequential) | 10 | ~8 min | 30 rec/min | ~200 MB |
| 12 (default) | 10 | ~40 sec | 360 rec/min | ~2 GB |
| 24 (high) | 10 | ~25 sec | 576 rec/min | ~4 GB |

### Data Processing Performance

| Dataset Size | Processing Time | Method |
|--------------|----------------|--------|
| 100k records | ~2 minutes | Apache Spark |
| 780k records | ~8 minutes | Apache Spark |
| 1M+ records | ~12 minutes | Apache Spark |

### Dashboard Performance

- **Data loading:** ~3 seconds (cached)
- **Map rendering:** ~5 seconds (10k points)
- **Chart updates:** <1 second
- **ML predictions:** ~2 seconds

---

## Documentation

Detailed documentation for each component:

- [Dashboard Quick Start](DASHBOARD_QUICKSTART.md) - Get started with the dashboard in 5 minutes
- [Dashboard README](visualization/README.md) - Complete dashboard documentation
- [MEA Scraper V2](Scraper_new/MEA Scraper/README_MEA_SCRAPER_V2.md) - Web scraper documentation
- [Gemini Pipeline](Scraper_new/Python Pipeline/README_GEMINI.md) - AI integration guide
- [Dashboard Fix Summary](DASHBOARD_FIX_SUMMARY.md) - Recent fixes and improvements

---

## License

This project is for educational purposes as part of the Data Science and Data Engineering course at Chulalongkorn University.

**Data Sources:**
- Traffy Fondue data is public domain
- MEA data is scraped from public website (respect rate limits)

---

## Team

**DSDE M150-Lover Team**

**Course:** 2110403 Data Science and Data Engineering
**Institution:** Chulalongkorn University
**Department:** Computer Engineering and Digital Technology (CEDT)
**Academic Year:** 2025 Semester 1

---

## Acknowledgments

- **Bangkok Metropolitan Administration** - Traffy Fondue complaint data
- **Metropolitan Electricity Authority (MEA)** - Power outage data
- **Chulalongkorn University** - Educational support and resources
- **Open Source Community** - Libraries and frameworks used in this project

---

## Contact

For questions, issues, or suggestions regarding this project:

- Check the [documentation](visualization/README.md)
- Review [troubleshooting guides](DASHBOARD_QUICKSTART.md#troubleshooting)
- Contact the development team

---

## Project Status

- **Current Version:** 1.0
- **Status:** Finished Developement
- **Last Updated:** December 2025

### Completed Features
- Web scraping with parallel processing
- Data cleaning with Apache Spark
- ML models (forecasting & anomaly detection)
- Interactive dashboard with 5 tabs
- Geospatial visualizations
- Real-time filtering and analysis

---

**Made with data, analytics, and machine learning by the M150-Lover Team**
