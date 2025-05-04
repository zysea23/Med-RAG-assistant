#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

from paper_scraper.scraper import PaperScraper
from paper_scraper.utils.logging_config import setup_logging

# Configure logging
logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="PubMed Literature Search Tool")
    parser.add_argument('--query', '-q', required=True, help='Search query')
    parser.add_argument('--max-results', '-n', type=int, default=100,
                        help='Maximum number of results with valid PDFs (default: 100)')
    parser.add_argument('--output-dir', '-o', default='scrape_output',
                        help='Output directory (default: scrape_output)')
    parser.add_argument('--rate-limit', '-r', type=float, default=0.5,
                        help='Rate limit between requests in seconds (default: 0.1)')
    parser.add_argument('--date-range', '-dr', nargs=2, metavar=('START', 'END'),
                        help='Date range in YYYY/MM/DD format (PubMed only)')
    parser.add_argument('--sort', '-s', choices=['relevance', 'date'], default='relevance',
                        help='Sort order (default: relevance)')
    parser.add_argument('--download-pdfs', '-p', action='store_true',
                        help='Download PDFs when available')
    args = parser.parse_args()
    
    # Initialize scraper and create folders
    scraper = PaperScraper(output_dir=args.output_dir, rate_limit=args.rate_limit)
    metadata_path, pdf_path = scraper.create_query_folder('pubmed', args.query)
    
    valid_papers = []
    total_attempts = 0
    max_attempts = args.max_results * 5 
    
    while len(valid_papers) < args.max_results and total_attempts < max_attempts:
        batch_size = min(100, args.max_results * 2)
        pmids = scraper.search_pubmed(args.query, batch_size, args.date_range, args.sort)
        results = scraper.fetch_pubmed_details(pmids)
        
        for paper in results:
            if len(valid_papers) >= args.max_results:
                break
                    
            if pdf_url := paper.get('full_text_link'):
                try:
                    title = paper.get('title', '')
                    safe_title = ''.join(c.lower() for c in title if c.isalnum() or c.isspace())
                    safe_title = safe_title.replace(' ', '_')[:100]  # Truncate to reasonable length
                    
                    # Fall back to PMID if title processing results in empty string
                    filename = f"{safe_title or paper['pubmed_id']}.pdf"
                    pdf_file = pdf_path / filename
                    
                    scraper.download_pdf(pdf_url, str(pdf_file))
                    valid_papers.append(paper)
                    logger.info(f"Successfully downloaded PDF {len(valid_papers)}/{args.max_results}")
                except Exception as e:
                    logger.warning(f"Failed to download PDF for PMID {paper['pubmed_id']}: {e}")
            total_attempts += 1
            
    # Save metadata of the successfully processed papers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = metadata_path / f"metadata_{timestamp}.json"
    with open(metadata_file, "w") as f:
        json.dump(valid_papers, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

if __name__ == '__main__':
    main()