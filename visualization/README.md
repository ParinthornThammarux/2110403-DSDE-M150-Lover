# Urban Issue Forecasting Visualization

Interactive visualization tools for Bangkok Traffy complaint data analysis, including geospatial dashboards and network graph analysis.

## Features

### 1. Interactive Dashboard (`dashboard/app.py`)
- **Geospatial Analysis**: Interactive maps with heatmaps and marker clusters
- **Time Series Analysis**: Historical complaint trends with moving averages
- **Forecasting**: 30-day ahead predictions with confidence intervals
- **Analytics**: District heatmaps, category distributions, resolution time analysis
- **Anomaly Detection**: Identification of unusual complaint patterns

### 2. Interactive Network Analysis Dashboard (`dashboard/network_app.py`) **NEW!**
- **Interactive Network Visualization**: Real-time interactive network graphs with Pyvis
- **Complaint Co-occurrence Network**: Visualizes relationships between complaint types
- **Organization Collaboration Network**: Shows organizations handling similar issues
- **Community Detection**: Visual identification of related clusters
- **Multiple Layouts**: Spring, Kamada-Kawai, Circular, Random layouts
- **Centrality Analysis**: Degree, Betweenness, Closeness, PageRank metrics
- **Degree Distribution**: Statistical analysis with log-log plots
- **Full Customization**: Adjust node sizes, spacing, colors, edge visibility

### 3. Graph Network Analysis Script (`graphs/network_analysis.py`)
- **Batch Processing**: Generate all network visualizations at once
- **Static Outputs**: PNG images with matplotlib
- **Interactive HTML**: Standalone HTML files with Plotly
- **Data Exports**: GraphML, GML, edge list formats

## Prerequisites

- Python 3.11 or higher
- `clean_data.csv` in the project root directory

## Installation

### Step 1: Install Dependencies

From the `visualization/` directory:

```bash
pip install -r requirements.txt
```

Or install globally from project root:

```bash
pip install -r visualization/requirements.txt
```

### Step 2: Verify Data File

Ensure `clean_data.csv` exists in the project root:

```bash
ls ../clean_data.csv
```

If missing, generate it using the `clean_data.ipynb` notebook.

## Usage

### Running the Interactive Dashboard

From the **project root directory**:

```bash
streamlit run visualization/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Dashboard Features:

**Sidebar Filters:**
- Date range selection
- District filter
- Complaint type filter
- Map visualization mode (heatmap/clusters)

**Tabs:**
1. **🗺️ Geospatial Analysis**: Interactive map with district statistics
2. **📊 Time Series & Forecasting**: Historical trends and 30-day predictions
3. **📈 Analytics**: Category distributions, heatmaps, resolution times
4. **🔍 Anomaly Detection**: Unusual complaints based on resolution time

### Running Interactive Network Analysis Dashboard

From the **project root directory**:

```bash
streamlit run visualization/dashboard/network_app.py
```

Opens at `http://localhost:8502` (different port from main dashboard)

#### Network Dashboard Features:

**Network Selection:**
- Complaint Type Co-occurrence Network
- Organization Collaboration Network

**Visualization Controls:**
- Layout algorithms (Spring, Kamada-Kawai, Circular, Random)
- Node sizing by centrality metrics
- Adjustable graph size and spacing
- Edge visibility toggle
- Community detection with color coding

**Analysis Tab:**
- Basic network statistics (nodes, edges, density, diameter)
- Top central nodes by multiple metrics
- Degree distribution plots (linear and log-log)
- Complete centrality analysis table
- Community analysis with members list
- Downloadable CSV exports

### Running Network Analysis Script (Batch Mode)

From the **project root directory**:

```bash
python visualization/graphs/network_analysis.py
```

#### Output Files:

**Static Images (PNG):**
- `visualization/graphs/outputs/complaint_type_co-occurrence_network.png`
- `visualization/graphs/outputs/organization_collaboration_network.png`

**Interactive HTML:**
- `visualization/graphs/outputs/interactive_complaint_network_interactive.html`
- `visualization/graphs/outputs/interactive_organization_network_interactive.html`

**Graph Data Exports:**
- `visualization/graphs/outputs/*.graphml` (GraphML format)
- `visualization/graphs/outputs/*.gml` (GML format)
- `visualization/graphs/outputs/*_edges.txt` (Edge list)

### Integration with Forecasting Model

The dashboard automatically integrates with the forecasting model:

1. If `ml_models/forecasting/outputs/forecast_predictions.csv` exists, it will be used
2. Otherwise, simulated forecast data is generated for demonstration

To use real forecasts:

```bash
# Train the model first
python ml_models/forecasting/train_lstm_model.py

# Then run the dashboard
streamlit run visualization/dashboard/app.py
```

## Project Structure

```
visualization/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── dashboard/
│   └── app.py                   # Streamlit dashboard application
├── graphs/
│   ├── network_analysis.py      # Graph network visualization
│   └── outputs/                 # Generated visualizations
│       ├── *.png                # Static network images
│       ├── *.html               # Interactive visualizations
│       ├── *.graphml            # Graph data exports
│       ├── *.gml
│       └── *_edges.txt
└── ...
```

## Data Format

The visualization tools expect `clean_data.csv` with the following columns:

**Required:**
- `type`: Complaint types (e.g., `{น้ำท่วม,ร้องเรียน}`)
- `organization`: Handling organization
- `timestamp`: Datetime of complaint
- `lat`, `lon`: Coordinates
- `district`, `subdistrict`: Location
- `solve_days`: Days to resolve
- `state_*`: One-hot encoded status columns
- `star_*`: One-hot encoded rating columns

## Troubleshooting

### Issue: "Data file not found"
**Solution:** Ensure `clean_data.csv` is in the project root, not in the `visualization/` directory.

### Issue: "No module named 'streamlit'"
**Solution:** Install requirements: `pip install -r visualization/requirements.txt`

### Issue: Map not displaying
**Solution:**
- Check that `lat` and `lon` columns have valid coordinates
- Ensure data is filtered to Bangkok region (lat ~13.7, lon ~100.5)

### Issue: Graph visualization empty
**Solution:**
- Ensure `type` column has multi-label complaints in `{type1,type2}` format
- Check that there are co-occurring complaint types in the data

### Issue: "Port 8501 already in use"
**Solution:** Stop other Streamlit instances or use a different port:
```bash
streamlit run visualization/dashboard/app.py --server.port 8502
```

## Performance Tips

1. **Large Datasets**: The dashboard limits map markers to 5,000 points for performance
2. **Date Filtering**: Use date range filters to analyze specific time periods
3. **Network Analysis**: Focuses on top 30 complaint types and well-connected organizations
4. **Browser**: Chrome or Firefox recommended for best interactive visualization performance

## Examples

### Dashboard Screenshots

**Key Metrics:**
- Total complaints, average resolution time, completion rate, anomaly rate

**Geospatial View:**
- Heatmap showing complaint density across Bangkok
- Marker clusters for detailed location view

**Time Series:**
- Historical complaint trends with 7-day moving average
- 30-day forecast with confidence intervals

**Analytics:**
- Top 15 complaint types distribution
- District vs month intensity heatmap
- Resolution time by complaint type

### Network Analysis Output

**Complaint Network:**
- Nodes: Complaint types
- Edges: Co-occurrence relationships
- Communities: Related complaint clusters
- Size: Proportional to degree centrality

**Organization Network:**
- Nodes: Government organizations
- Edges: Shared complaint handling
- Communities: Collaborative groups

## Contributing

Part of the **2110403-DSDE-M150-Lover** project for Data Science and Data Engineering course, Chulalongkorn University.

## License

Educational project - Chulalongkorn University

## Support

For issues or questions, please contact the project team or refer to the main project README.
