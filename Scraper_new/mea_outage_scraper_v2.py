"""
MEA Power Outage Scraper V2 - Day-based records
Scrapes power outage notifications from MEA and creates one record per day
Each record contains all outage data for that specific day
Exports data in batches (1 CSV per page) with parallel processing

Author: DSDE M150-Lover Team
"""

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime
import time
import logging
from pathlib import Path
from typing import List, Dict
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
Path('logs').mkdir(exist_ok=True)

import sys

class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mea_outage_scraper_v2.log', encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MEAOutageScraperV2:
    """Scraper for MEA power outage notifications - creates one record per day"""

    def __init__(self, debug: bool = False, max_workers: int = 12):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.output_dir = Path("data/external/scraped")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.max_workers = max_workers
        if debug:
            self.debug_dir = Path("debug")
            self.debug_dir.mkdir(exist_ok=True)

    def scrape_all_outages(self, start_page: int = 1, stop_page: int = None, max_pages: int = None) -> Dict[int, List[Dict]]:
        """
        Scrape all power outage notifications from MEA website
        Returns dictionary mapping page numbers to their day-based records

        Args:
            start_page: Page number to start from (default: 1)
            stop_page: Page number to stop at (inclusive, default: None = scrape until end)
            max_pages: Maximum number of pages to scrape (default: None = all pages)
                       Note: This limits total pages scraped, not the ending page number

        Returns:
            Dictionary with page numbers as keys and lists of day records as values
        """
        logger.info(f"Starting MEA power outage scraper V2 (day-based) with {self.max_workers} workers...")
        if start_page > 1:
            logger.info(f"Starting from page {start_page}")
        if stop_page:
            logger.info(f"Will stop at page {stop_page}")
        elif max_pages:
            logger.info(f"Will scrape maximum {max_pages} pages")

        base_url = "https://www.mea.or.th/en/public-relations/power-outage-notifications/news"
        page_records = {}

        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')

            driver = webdriver.Chrome(options=options)

            page = start_page
            consecutive_failures = 0
            pages_scraped = 0

            while True:
                # Check stop conditions
                if stop_page and page > stop_page:
                    logger.info(f"Reached stop page: {stop_page}")
                    break

                if max_pages and pages_scraped >= max_pages:
                    logger.info(f"Reached maximum pages limit: {max_pages}")
                    break

                try:
                    if page == 1:
                        url = base_url
                    else:
                        url = f"{base_url}?page={page}"

                    logger.info(f"Scraping page {page}: {url}")
                    driver.get(url)
                    time.sleep(3)

                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # Find all outage notification items (links to detail pages)
                    outage_links = self._find_outage_links(soup)

                    if not outage_links:
                        consecutive_failures += 1
                        logger.warning(f"No outage items found on page {page}")

                        if consecutive_failures >= 3:
                            logger.info("No more pages with content found")
                            break

                        page += 1
                        continue

                    consecutive_failures = 0
                    logger.info(f"Found {len(outage_links)} outage announcements on page {page}")

                    # Process announcements in parallel with ThreadPoolExecutor
                    page_day_records = []
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Submit all tasks
                        future_to_link = {}
                        for idx, link in enumerate(outage_links):
                            future = executor.submit(self._scrape_announcement_parallel, link, page, idx)
                            future_to_link[future] = (idx, link)

                        # Collect results as they complete
                        for future in as_completed(future_to_link):
                            idx, link = future_to_link[future]
                            try:
                                day_records = future.result()
                                if day_records:
                                    page_day_records.extend(day_records)
                                    logger.info(f"  ✓ Page {page}, Announcement {idx+1}: Extracted {len(day_records)} day records")
                            except Exception as e:
                                logger.debug(f"Error processing announcement {idx} on page {page}: {str(e)}")

                    # Save this page's records
                    page_records[page] = page_day_records
                    logger.info(f"Page {page} complete: {len(page_day_records)} day records")

                    # Save page batch immediately
                    self.save_page_batch(page, page_day_records)

                    # Increment pages scraped counter
                    pages_scraped += 1

                    # Check if there's a next page
                    has_next = self._has_next_page(soup)
                    if not has_next:
                        logger.info("No next page found, scraping complete")
                        break

                    page += 1
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error on page {page}: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    page += 1
                    continue

            driver.quit()

        except Exception as e:
            logger.error(f"Fatal error in MEA scraper: {str(e)}")

        total_records = sum(len(records) for records in page_records.values())
        logger.info(f"Scraping complete. Total pages: {len(page_records)}, Total day records: {total_records}")
        return page_records

    def _find_outage_links(self, soup: BeautifulSoup) -> List[str]:
        """Find links to outage detail pages"""
        links = []
        seen_urls = set()

        # Find all items in the grid
        items = soup.select('div.col-sm-6.col-md-4.col-lg-4.mb-4')

        for item in items:
            link_elem = item.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if '/news/' in href and href not in seen_urls:
                    if not href.startswith('http'):
                        href = f"https://www.mea.or.th{href}"
                    seen_urls.add(href)
                    links.append(href)

        return links

    def _scrape_announcement(self, url: str, driver, page_num: int, item_idx: int) -> List[Dict]:
        """
        Scrape a single announcement page and extract day-based records (using shared driver)

        Returns:
            List of day records, one per day mentioned in the announcement
        """
        day_records = []

        try:
            driver.get(url)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Get main content
            content_elem = soup.select_one('div.content-description')
            if not content_elem:
                return []

            # Parse content to extract day-based records
            content_html = str(content_elem)
            content_text = content_elem.get_text(separator='\n', strip=True)

            # Find all day headers (e.g., "Saturday, Sep 6, 2025" or "Sunday, Sep 7, 2025")
            day_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})'

            # Split content by days
            day_sections = self._split_content_by_days(content_text, day_pattern)

            for day_info in day_sections:
                # Calculate day of week from the actual outage date
                day_of_week = ""
                if day_info['outage_date']:
                    day_of_week = day_info['outage_date'].strftime('%A')  # Monday, Tuesday, etc.

                day_record = {
                    'source': 'MEA',
                    'announcement_url': url,
                    'day_of_week': day_of_week,
                    'outage_date': day_info['outage_date'],
                    'outage_data': day_info['content']  # Raw data for the day
                }
                day_records.append(day_record)

        except Exception as e:
            logger.debug(f"Error scraping announcement {url}: {str(e)}")

        return day_records

    def _scrape_announcement_parallel(self, url: str, page_num: int, item_idx: int) -> List[Dict]:
        """
        Scrape a single announcement page and extract day-based records (parallel version with own driver)

        Returns:
            List of day records, one per day mentioned in the announcement
        """
        day_records = []

        try:
            # Create a driver for this thread
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-gpu')
            options.add_argument('--log-level=3')

            driver = webdriver.Chrome(options=options)

            driver.get(url)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Get main content
            content_elem = soup.select_one('div.content-description')
            if not content_elem:
                driver.quit()
                return []

            # Parse content to extract day-based records
            content_html = str(content_elem)
            content_text = content_elem.get_text(separator='\n', strip=True)

            # Find all day headers (e.g., "Saturday, Sep 6, 2025" or "Sunday, Sep 7, 2025")
            day_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})'

            # Split content by days
            day_sections = self._split_content_by_days(content_text, day_pattern)

            for day_info in day_sections:
                # Calculate day of week from the actual outage date
                day_of_week = ""
                if day_info['outage_date']:
                    day_of_week = day_info['outage_date'].strftime('%A')  # Monday, Tuesday, etc.

                day_record = {
                    'source': 'MEA',
                    'announcement_url': url,
                    'day_of_week': day_of_week,
                    'outage_date': day_info['outage_date'],
                    'outage_data': day_info['content']  # Raw data for the day
                }
                day_records.append(day_record)

            driver.quit()

        except Exception as e:
            logger.debug(f"Error scraping announcement {url}: {str(e)}")
            try:
                driver.quit()
            except:
                pass

        return day_records

    def _split_content_by_days(self, content_text: str, day_pattern: str) -> List[Dict]:
        """Split content into sections by day"""
        day_sections = []

        # Find all day matches first
        matches = list(re.finditer(day_pattern, content_text, re.IGNORECASE))

        if not matches:
            return []

        # Process each day section
        for i, match in enumerate(matches):
            day_of_week = match.group(1)
            month_str = match.group(2)
            day_num = int(match.group(3))
            year = int(match.group(4))

            # Parse date
            outage_date = self._parse_day_date(month_str, day_num, year)

            # Get content for this day (from after this match to before next match)
            start_pos = match.end()
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content_text)

            day_content = content_text[start_pos:end_pos].strip()

            # Clean the content
            day_content = self._clean_outage_data(day_content)

            if day_content:
                day_sections.append({
                    'day_of_week': day_of_week,
                    'outage_date': outage_date,
                    'content': day_content
                })

        return day_sections

    def _clean_outage_data(self, text: str) -> str:
        """Clean and format outage data text"""
        if not text:
            return text

        # Remove footer/contact info
        stop_phrases = [
            'We sincerely apologize',
            'In case of any inquiries',
            'In the meantime you can also check',
            'Metropolitan Electricity Authority'
        ]

        for phrase in stop_phrases:
            pos = text.find(phrase)
            if pos > 0:
                text = text[:pos]

        # First, handle spaced out time formats: "08. 3 0" -> "08.30"
        # Match patterns like "08. 3 0" or "0 9. 0 0"
        text = re.sub(r'(\d)\s+(\d)\s*\.\s*(\d)\s+(\d)', r'\1\2:\3\4', text)  # "0 9. 0 0" -> "09:00"
        text = re.sub(r'(\d{2})\s*\.\s*(\d)\s+(\d)', r'\1:\2\3', text)  # "08. 3 0" -> "08:30"

        # Fix standalone scattered numbers on separate lines (e.g., "0\n9\n." -> "09")
        text = re.sub(r'(\d)\s*\n\s*(\d)\s*\n\s*\.', r'\1\2.', text)

        # Now fix time formats: "08.30 AM" -> "08:30 AM"
        text = re.sub(r'(\d{1,2})\s*[:.]\s*(\d{2})\s*(AM|PM|am|pm)', r'\1:\2 \3', text)

        # Fix "12. 0 0" type patterns
        text = re.sub(r'(\d{1,2})\.\s*(\d)\s+(\d)\s+(AM|PM)', r'\1:\2\3 \4', text)

        # Fix number ranges with dashes
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', text)

        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.])', r'\1', text)
        text = re.sub(r'([,.])\s*([^\s\n])', r'\1 \2', text)

        # Remove zero-width spaces
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)

        # Fix CamelCase words
        text = re.sub(r'(Soi)([A-Z])', r'\1 \2', text)
        text = re.sub(r'(Road)([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)

        # Clean up lines - remove lines with only single digits or punctuation
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines, single digits, or lines with only punctuation
            if line and len(line) > 2 and not re.match(r'^[\d\W]$', line):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _parse_day_date(self, month_str: str, day: int, year: int) -> datetime:
        """Parse a date from month name, day, and year"""
        months = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

        month = months.get(month_str.lower()[:3])
        if month:
            return datetime(year, month, day)
        return None

    def _has_next_page(self, soup: BeautifulSoup) -> bool:
        """Check if there's a next page"""
        pagination = soup.select('ul.pagination li, div.pagination a')
        return len(pagination) > 0

    def save_page_batch(self, page_num: int, day_records: List[Dict]):
        """Save a single page's records to its own CSV file"""
        if not day_records:
            logger.warning(f"No day records to save for page {page_num}")
            return

        df = pd.DataFrame(day_records)

        # Remove duplicates based on date and announcement URL
        df = df.drop_duplicates(subset=['outage_date', 'announcement_url'], keep='first')

        # Sort by outage date (newest first)
        if 'outage_date' in df.columns:
            df['outage_date'] = pd.to_datetime(df['outage_date'], errors='coerce')
            df = df.sort_values(by='outage_date', ascending=False)

            # Format outage_date as YYYY-MM-DD string
            df['outage_date'] = df['outage_date'].dt.strftime('%Y-%m-%d')

        # Save with page number in filename
        filename = f'mea_power_outages_page_{page_num:03d}.csv'
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] Saved page {page_num} batch: {len(df)} day records to {output_path}")

        return df

    def save_data(self, page_records: Dict[int, List[Dict]], filename: str = 'mea_power_outages_by_day.csv'):
        """Save all day-based records to a combined CSV (optional)"""
        # Flatten all page records
        all_records = []
        for page_num in sorted(page_records.keys()):
            all_records.extend(page_records[page_num])

        if not all_records:
            logger.warning("No day records to save")
            return

        df = pd.DataFrame(all_records)

        # Remove duplicates based on date and announcement URL
        df = df.drop_duplicates(subset=['outage_date', 'announcement_url'], keep='first')

        # Sort by outage date (newest first)
        if 'outage_date' in df.columns:
            df['outage_date'] = pd.to_datetime(df['outage_date'], errors='coerce')
            df = df.sort_values(by='outage_date', ascending=False)

            # Format outage_date as YYYY-MM-DD string
            df['outage_date'] = df['outage_date'].dt.strftime('%Y-%m-%d')

        # Save combined file
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] Saved combined file: {len(df)} day records to {output_path}")

        return df


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='MEA Power Outage Scraper V2 (Day-based with parallel processing)')
    parser.add_argument('--start-page', type=int, default=1,
                       help='Page number to start from (default: 1)')
    parser.add_argument('--stop-page', type=int, default=None,
                       help='Page number to stop at (inclusive, default: scrape until end)')
    parser.add_argument('--max-pages', type=int, default=None,
                       help='Maximum number of pages to scrape (default: all pages)')
    parser.add_argument('--output', type=str, default='mea_power_outages_by_day_combined.csv',
                       help='Output filename for combined file (default: mea_power_outages_by_day_combined.csv)')
    parser.add_argument('--workers', type=int, default=12,
                       help='Number of parallel workers (default: 12)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-combined', action='store_true',
                       help='Skip creating combined CSV file (only create per-page batches)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("MEA POWER OUTAGE SCRAPER V2 (DAY-BASED RECORDS - BATCH MODE)")
    logger.info("="*80)

    # Build page range info
    page_info = []
    if args.start_page > 1:
        page_info.append(f"Start: page {args.start_page}")
    if args.stop_page:
        page_info.append(f"Stop: page {args.stop_page}")
    if args.max_pages:
        page_info.append(f"Max: {args.max_pages} pages")
    if not page_info:
        page_info.append("All pages")

    logger.info(f"Page range: {', '.join(page_info)}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Output mode: {'Page batches only' if args.no_combined else 'Page batches + combined file'}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("="*80)

    # Initialize scraper
    scraper = MEAOutageScraperV2(debug=args.debug, max_workers=args.workers)

    # Scrape outages
    start_time = datetime.now()
    page_records = scraper.scrape_all_outages(
        start_page=args.start_page,
        stop_page=args.stop_page,
        max_pages=args.max_pages
    )

    # Save combined data (optional)
    if page_records:
        total_records = sum(len(records) for records in page_records.values())

        if not args.no_combined:
            scraper.save_data(page_records, filename=args.output)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*80}")
        logger.info(f"Scraping completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total pages scraped: {len(page_records)}")
        logger.info(f"Total day records: {total_records}")
        logger.info(f"Average records per page: {total_records/len(page_records):.1f}")
        logger.info(f"Output directory: {scraper.output_dir}")
        logger.info(f"{'='*80}")
    else:
        logger.warning("No day records were scraped!")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
