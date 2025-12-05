# Pipeline with Gemini API

This is an alternative version of the pipeline that uses Google's Gemini API instead of local Ollama.

## Files

- **Pipeline_Gemini.py** - Main pipeline using Gemini API
- **Pipeline.py** - Original pipeline using local Ollama
- **requirements_gemini.txt** - Python dependencies for Gemini version
- **requirements.txt** - Python dependencies for Ollama version

## Comparison: Gemini vs Ollama

| Feature | Gemini API | Local Ollama |
|---------|-----------|--------------|
| **Setup** | Easy - just need API key | Complex - need to install & run server |
| **Cost** | Free tier available, then pay per use | 100% Free |
| **Speed** | Fast (cloud servers) | Fast (if good GPU) |
| **Privacy** | Data sent to Google | 100% Local |
| **Internet** | Required | Not required |
| **GPU** | Not needed | RTX 4070 recommended |
| **Model** | gemini-2.0-flash-exp | ministral-3:3b |

## Setup for Gemini Version

### 1. Get Gemini API Key

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

### 2. Set Environment Variable

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Or set it permanently:**
1. Search "Environment Variables" in Windows
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `GEMINI_API_KEY`
5. Variable value: Your API key
6. Click OK

### 3. Install Dependencies

```cmd
pip install -r requirements_gemini.txt
```

### 4. Run the Pipeline

```cmd
python Pipeline_Gemini.py
```

## Usage

### Input
The pipeline reads from:
```
MEA scraped\mea_power_outages_page_001.csv
```

### Output
Saves results to:
```
mea_power_outages_page_001_gemini.csv
```

### What It Does

1. **Loads CSV data** - Power outage announcements from MEA
2. **Filters dates** - Removes future dates, separates recent vs historical
3. **Extracts structured data** - Uses Gemini to extract:
   - Start/End times (HH:MM 24h format)
   - Location details (street, soi, village)
   - District (if mentioned)
   - Province (defaults to Bangkok)
4. **Fetches weather data** - Gets historical weather for each district:
   - Max temperature
   - Total rainfall
   - Max wind gusts
5. **Saves enriched data** - CSV with outage events + weather

## Configuration

Edit these variables in `Pipeline_Gemini.py`:

```python
# Model selection
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Fast and accurate
# Other options:
# - "gemini-1.5-flash" (stable)
# - "gemini-1.5-pro" (more capable, slower, costlier)

# Input/Output files
INPUT_FILE = r"path\to\your\input.csv"
OUTPUT_FILE = "output_gemini.csv"
```

## Pricing

**Gemini 2.0 Flash (Experimental):**
- FREE tier: 1500 requests/day
- After free tier: Very cheap (~$0.001 per request)

**Gemini 1.5 Flash:**
- FREE tier: 1500 requests/day
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens

**Gemini 1.5 Pro:**
- FREE tier: 50 requests/day
- Input: $1.25 / 1M tokens
- Output: $5.00 / 1M tokens

For your use case (~20 rows), you'll stay within free tier easily!

## Troubleshooting

### Error: "GEMINI_API_KEY not set"

**Solution:** Set the environment variable:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### Error: "ModuleNotFoundError: No module named 'google.genai'"

**Solution:** Install dependencies:
```cmd
pip install -r requirements_gemini.txt
```

### Error: "API key not valid"

**Solution:**
1. Check your API key is correct
2. Go to https://aistudio.google.com/apikey
3. Create a new key if needed

### Slow performance

**Solution:**
- Gemini should be fast (cloud servers)
- Check your internet connection
- Try switching to `gemini-1.5-flash` if using `gemini-2.0-flash-exp`

## Switching Between Versions

### Use Gemini When:
- ✅ You want quick setup (no Ollama installation)
- ✅ You have internet connection
- ✅ You don't mind data going to Google
- ✅ You're processing small amounts of data (free tier)

### Use Ollama When:
- ✅ You want 100% privacy (local processing)
- ✅ You process lots of data (no API costs)
- ✅ You have a good GPU (RTX 4070)
- ✅ You work offline

## Both Versions Give Same Results!

The output structure is identical - only the LLM backend changes.

## Next Steps

After running the pipeline:

1. **Check the output CSV** - Review extracted data
2. **Validate accuracy** - Compare with original MEA announcements
3. **Process more files** - Update `INPUT_FILE` for other pages
4. **Analyze data** - Use pandas/Excel for insights
