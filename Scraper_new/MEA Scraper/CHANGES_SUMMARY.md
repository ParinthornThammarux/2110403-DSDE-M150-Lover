# MEA Scraper V2 - Changes Summary

## What Changed

### ✅ Simplified CSV Structure

**Removed Fields:**
- ~~`announcement_title`~~ - Removed (not needed for analysis)
- ~~`notification_date`~~ - Removed (not the actual outage date)
- ~~`scraped_at`~~ - Removed (not needed for analysis)
- ~~`page_number`~~ - Removed (internal scraping detail)

**Changed Fields:**
- `day_of_week` - Now derived from the `outage_date` instead of parsed text
  - Before: Parsed from page text (e.g., "Saturday")
  - After: Calculated from outage_date (e.g., if outage_date is 2025-12-05, day_of_week is "Friday")

**Kept Fields:**
- ✅ `source` - Always "MEA"
- ✅ `announcement_url` - Link to original announcement
- ✅ `day_of_week` - Day name (now calculated from date)
- ✅ `outage_date` - The actual outage date (YYYY-MM-DD format)
- ✅ `outage_data` - All outage information for that day

## Final CSV Format

```csv
source,announcement_url,day_of_week,outage_date,outage_data
MEA,https://www.mea.or.th/.../news/og3We2yuo,Friday,2025-12-05,"Bangkok: The power outage areas are;
08:30 AM – 03:30 PM
- Seri Thai Road, Soi Seri Thai 14
- Sukhumvit Road, Soi 66/1..."
```

## Benefits of This Format

1. **Cleaner** - Only essential fields
2. **Simpler** - Fewer columns to manage
3. **Consistent** - day_of_week always matches outage_date
4. **Focused** - Only data needed for outage analysis

## How to Use

### Scrape first page (testing):
```bash
python Scraper_new/mea_outage_scraper_v2.py --max-pages 1
```

### Scrape all pages:
```bash
python Scraper_new/mea_outage_scraper_v2.py
```

### Output location:
`Scraper_new/data/external/scraped/mea_power_outages_by_day.csv`

## Data Processing

The `outage_data` field contains cleaned text with:
- ✅ Times normalized: "08:30 AM – 03:30 PM"
- ✅ Location names cleaned
- ✅ Excess whitespace removed
- ✅ Footer text removed
- ❌ Still unstructured text (not parsed into columns yet)

### Next Steps for Further Processing:

If you want to parse `outage_data` into structured columns:
1. Extract province/city (Bangkok, Nonthaburi, Samutprakan)
2. Parse time ranges
3. Extract street/Soi names
4. Split multiple locations into separate rows

This can be done in a separate data processing step after scraping.

## Performance

From testing with first page (24 announcements):
- **Records**: ~100 day records
- **Time**: ~100 seconds (~1.7 minutes)
- **Success rate**: ~79% of announcements parsed

## Files

- **Main scraper**: `Scraper_new/mea_outage_scraper_v2.py`
- **Documentation**: `Scraper_new/MEA_SCRAPER_V2_README.md`
- **This summary**: `Scraper_new/CHANGES_SUMMARY.md`
