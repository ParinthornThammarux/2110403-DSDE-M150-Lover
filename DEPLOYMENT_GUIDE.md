# Deployment Guide - Urban Issue Forecasting Dashboard

This guide explains how to deploy your Streamlit dashboard to a public webpage for your CV/portfolio.

## Quick Start - Streamlit Community Cloud (RECOMMENDED)

**Best Option:** Free, easy, and perfect for portfolios

### Prerequisites
- GitHub account
- Your code pushed to GitHub repository

### Deployment Steps

#### 1. Prepare Your Repository

Ensure these files are in your repo:
- ✅ `visualization/main_dashboard.py` (main app)
- ✅ `visualization/requirements.txt` (dependencies)
- ✅ `clean_data.csv` (your data file)
- ✅ ML model files in `ml_models/` (optional)

#### 2. Optimize for Cloud (Important!)

Since you have 780k+ records, you need to optimize for Streamlit Cloud's 1GB RAM limit.

**Option A: Sample the data (Recommended)**

Create `visualization/config.py`:
```python
# Configuration for deployment
SAMPLE_SIZE = 100000  # Use 100k records instead of 780k
ENABLE_SAMPLING = True  # Set to False for local development
```

**Option B: Use data compression**

Compress your CSV:
```bash
# In your project directory
gzip -k clean_data.csv
# This creates clean_data.csv.gz
```

Then update your data loading code to handle compressed files.

#### 3. Create .streamlit/config.toml

Create directory `.streamlit/` with file `config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 500
enableXsrfProtection = true
enableCORS = false
```

#### 4. Create .gitignore (if not exists)

```
__pycache__/
*.pyc
.DS_Store
.vscode/
*.pkl.bak
.env
*.log
```

#### 5. Deploy to Streamlit Cloud

1. **Go to:** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click** "New app"

4. **Fill in the details:**
   - Repository: `your-username/2110403-DSDE-M150-Lover`
   - Branch: `main`
   - Main file path: `visualization/main_dashboard.py`

5. **Advanced settings** (click "Advanced"):
   - Python version: 3.10
   - Secrets: Add any API keys if needed

6. **Click "Deploy"**

7. **Wait** 5-10 minutes for deployment

8. **You'll get a URL** like:
   ```
   https://your-username-urban-forecasting.streamlit.app
   ```

#### 6. Custom Domain (Optional)

You can use a custom domain:
- Go to Settings → General
- Add your custom domain
- Update DNS records as instructed

---

## Alternative Deployment Options

### Option 2: Render (Free Tier)

**URL:** [render.com](https://render.com)

#### Setup Files

Create `render.yaml`:
```yaml
services:
  - type: web
    name: urban-forecasting-dashboard
    runtime: python
    buildCommand: pip install -r visualization/requirements.txt
    startCommand: streamlit run visualization/main_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

#### Deploy
1. Connect GitHub repo to Render
2. Create new Web Service
3. Select your repo
4. Render will auto-detect and deploy

**Free Tier Limits:**
- 750 hours/month
- Sleeps after 15 min of inactivity
- 512 MB RAM

---

### Option 3: Hugging Face Spaces (FREE)

**URL:** [huggingface.co/spaces](https://huggingface.co/spaces)

Great for ML/data science projects!

#### Setup

1. **Create Space**
   - Go to Hugging Face Spaces
   - Click "Create new Space"
   - Select "Streamlit" as SDK
   - Name: `urban-issue-forecasting`

2. **Upload Files**
   - Upload `visualization/` folder
   - Upload `clean_data.csv`
   - Upload `requirements.txt`

3. **Create app.py** in root:
   ```python
   import subprocess
   subprocess.run(["streamlit", "run", "visualization/main_dashboard.py"])
   ```

4. **Deploy automatically**

**URL will be:**
```
https://huggingface.co/spaces/your-username/urban-issue-forecasting
```

---

### Option 4: Railway (Paid but Affordable)

**URL:** [railway.app](https://railway.app)

**Cost:** ~$5/month for small apps

#### Setup

1. Connect GitHub repo
2. Railway auto-detects Streamlit
3. Set start command:
   ```
   streamlit run visualization/main_dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```
4. Deploy

**Pros:**
- More resources than free tiers
- No sleep/auto-shutdown
- Better for production

---

## Optimizing Your Dashboard for Deployment

### 1. Reduce Data Size

**Method 1: Sample the data**

Update your data loading in `main_dashboard.py`:

```python
@st.cache_data
def load_data():
    """Load and cache the complaint data"""
    try:
        df = pd.read_csv('clean_data.csv')

        # For deployment: sample data to reduce memory
        if len(df) > 100000:
            df = df.sample(n=100000, random_state=42)

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
```

**Method 2: Pre-aggregate data**

Create aggregated datasets for charts:
```python
# Instead of loading full dataset for every chart
# Pre-aggregate before deployment
district_counts = df.groupby('district').size().to_csv('data/district_aggregates.csv')
```

**Method 3: Use Parquet instead of CSV**

```python
# Convert CSV to Parquet (much smaller)
df = pd.read_csv('clean_data.csv')
df.to_parquet('clean_data.parquet', compression='gzip')

# Then load Parquet in your app
df = pd.read_parquet('clean_data.parquet')
```

### 2. Optimize ML Models

If your ML models are too large:

```python
# Don't load models until needed
@st.cache_resource
def load_ml_model():
    if st.session_state.get('use_ml', False):
        return joblib.load('ml_models/forecasting/models/rf_forecaster.pkl')
    return None
```

### 3. Add Loading States

Improve UX with loading indicators:

```python
with st.spinner('Loading data...'):
    df = load_data()

with st.spinner('Generating visualization...'):
    fig = plot_complaints_by_district(df)
    st.plotly_chart(fig)
```

### 4. Reduce Map Complexity

For the Folium maps:

```python
# Instead of showing all 780k points
# Sample for map display only
map_sample = df.sample(n=min(10000, len(df)))

# Create map with sampled data
m = folium.Map(...)
for idx, row in map_sample.iterrows():
    # Add markers
```

---

## Environment Variables & Secrets

If you have API keys or sensitive data:

### For Streamlit Cloud

1. Go to your app settings
2. Click "Secrets"
3. Add in TOML format:
   ```toml
   [api]
   gemini_key = "your-api-key"

   [database]
   connection_string = "your-db-url"
   ```

4. Access in code:
   ```python
   import streamlit as st
   api_key = st.secrets["api"]["gemini_key"]
   ```

---

## Deployment Checklist

Before deploying:

- [ ] Code is pushed to GitHub
- [ ] `requirements.txt` is complete and up-to-date
- [ ] Data files are included (or accessible via URL)
- [ ] Remove any hardcoded paths (use relative paths)
- [ ] Test locally: `streamlit run visualization/main_dashboard.py`
- [ ] Optimize data size (< 100 MB recommended)
- [ ] Add loading states for better UX
- [ ] Configure `.streamlit/config.toml`
- [ ] Add proper error handling
- [ ] Test with sample data first

---

## Post-Deployment

### 1. Test Your Deployment

- [ ] All visualizations load correctly
- [ ] Filters work properly
- [ ] ML predictions function (if enabled)
- [ ] Maps render correctly
- [ ] No errors in console
- [ ] Performance is acceptable

### 2. Monitor Usage

Streamlit Cloud provides analytics:
- View count
- Active users
- Error logs
- Resource usage

### 3. Update Your CV/Portfolio

Add the link to your CV with a good description:

**Example:**
```
Urban Issue Forecasting Dashboard
Interactive data visualization and ML prediction system for analyzing
780,000+ Bangkok urban complaints using Streamlit, Apache Spark, and
scikit-learn.
🔗 https://your-app.streamlit.app
📁 GitHub: github.com/your-username/2110403-DSDE-M150-Lover
```

---

## Troubleshooting

### Issue: "Out of Memory" Error

**Solution:**
- Reduce data sample size
- Remove heavy ML models
- Use data aggregation
- Consider paid tier

### Issue: "Module not found"

**Solution:**
- Check `requirements.txt` is complete
- Verify all imports are correct
- Check Python version compatibility

### Issue: App is Slow

**Solution:**
- Use `@st.cache_data` for data loading
- Use `@st.cache_resource` for ML models
- Reduce map markers
- Optimize queries

### Issue: "File not found"

**Solution:**
- Use relative paths, not absolute
- Ensure files are in repo
- Check file paths in code

---

## Example Requirements.txt for Deployment

Make sure your `visualization/requirements.txt` includes:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
folium>=0.14.0
streamlit-folium>=0.15.0
scikit-learn>=1.3.0
joblib>=1.3.0
pydeck>=0.8.0
```

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Streamlit Cloud** | 1 GB RAM, Unlimited apps | N/A | Portfolios, demos |
| **Render** | 512 MB RAM, 750 hrs/mo | $7/mo | Small apps |
| **Heroku** | 512 MB RAM (limited) | $7/mo | Production apps |
| **Railway** | $5 free credit | $5-20/mo | Growing apps |
| **Hugging Face** | Free | N/A | ML/AI projects |

---

## Recommended Setup for Your Project

**For CV/Portfolio:**

1. **Use Streamlit Community Cloud** (FREE)
2. **Sample data to 100k records** for performance
3. **Enable caching** for all data loads
4. **Simplify maps** (show 10k points max)
5. **Add nice README** with screenshots
6. **Include GitHub link** in dashboard

This gives you a professional, fast, free deployment perfect for showing to recruiters!

---

## Example GitHub README for Portfolio

Add this to your main README to showcase deployment:

```markdown
## 🌐 Live Demo

**View the live dashboard:** [https://your-app.streamlit.app](https://your-app.streamlit.app)

![Dashboard Screenshot](screenshots/dashboard.png)

### Features
- 📊 Interactive visualizations with 100k+ records
- 🤖 ML-powered forecasting and anomaly detection
- 🗺️ Geospatial heat maps and cluster analysis
- ⚡ Real-time filtering and analysis
```

---

## Next Steps

1. **Choose a platform** (Streamlit Cloud recommended)
2. **Optimize your code** for deployment
3. **Test locally** with sampled data
4. **Deploy** following the guide above
5. **Share** the link on your CV and LinkedIn!

---

**Need Help?**
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Check deployment logs

---

**Last Updated:** December 2025
**Project:** Urban Issue Forecasting System
**Team:** M150-Lover, Chulalongkorn University
