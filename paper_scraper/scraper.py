import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from time import sleep

import requests
from bs4 import BeautifulSoup

from paper_scraper.parsers.pubmed_parser import (
    get_doi, get_title, get_abstract, get_authors,
    get_journal_info, get_pub_date, get_full_text_link
)
from paper_scraper.downloaders.pdf_downloader import download_pdf, extract_text_from_pdf

logger = logging.getLogger(__name__)

class PaperScraper:
    """
    Class for searching and retrieving scientific literature from PubMed.
    Features:
      - Metadata retrieval and PDF downloading (when available)
      - Rate limiting for API calls
    """
    def __init__(self, output_dir: str = "scrape_output", rate_limit: float = 0.1):
        self.output_dir = output_dir
        self.rate_limit = rate_limit # Time between requests in seconds
        self.unpaywall_email = os.getenv("UNPAYWALL_EMAIL")
        self.core_api_key = os.getenv("CORE_API_KEY")
        os.makedirs(output_dir, exist_ok=True)
        
    def create_query_folder(self, database: str, query: str) -> tuple[Path, Path]:
        """
        Create folder structure for a specific query under the given database.
        Returns paths for metadata and PDF storage.
        """
        query_folder = query.replace(" ","_").replace("/","_")[:20]
        metadata_path = Path(self.output_dir) / database / "metadata" / query_folder
        pdf_path = Path(self.output_dir) / database / "pdf" / query_folder
        
        metadata_path.mkdir(parents=True, exist_ok=True)
        pdf_path.mkdir(parents=True, exist_ok=True)
        return metadata_path, pdf_path
    
    def search_pubmed(self, query: str, max_results: int = 100,
                      date_range: Optional[Tuple[str, str]] = None,
                      sort: str = "relevance") -> List[str]:
        """
        Search PubMed for the given query and return a list of PubMed IDs.
        """
        logger.info(f"Searching PubMed for : {query}")
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Append date filter if provided
        if date_range:
            start, end = date_range
            query = f"{query} AND {start}[PDAT] : {end}[PDAT]"
            
        # Use official NCBI E-utilities API
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results * 2, 100000), # Respect PubMed's max limit
            "retmode": "json",
            "sort": "relevance" if sort == "relevance" else "pub+date",
            # "api_key": os.getenv("NCBI_API_KEY", ""), 
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise RuntimeError(f"PubMed API error: {data['error']}")
                
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.warning("No results found for query")
            else:
                logger.info(f"Found {len(id_list)} results")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch results from PubMed: {e}")
            raise
        return id_list
    
    def fetch_pubmed_details(self, id_list: List[str]) -> List[Dict]:
        """
        Retrieve detailed metadata for a list of PubMed IDs.
        """
        logger.info(f"Fetching details for {len(id_list)} papers...")
        details = []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        for pmid in id_list:
            sleep(self.rate_limit)
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "rettype": "full"
            }
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "xml")
                article_element = soup.find("PubmedArticle")
                
                if not article_element:
                    logger.warning(f"No article data found for PMID {pmid}")
                    continue
                    
                full_text_link = get_full_text_link(article_element, self.unpaywall_email)
                
                article = {
                    "pubmed_id": pmid,
                    "doi": get_doi(article_element),
                    "title": get_title(article_element),
                    "abstract": get_abstract(article_element),
                    "authors": get_authors(article_element),
                    "journal": get_journal_info(article_element),
                    "publication_date": get_pub_date(article_element),
                    "full_text_link": full_text_link
                }
                details.append(article)
                logger.info(f"Successfully processed PMID {pmid}")
            except Exception as e:
                logger.error(f"Error processing PMID {pmid}: {e}")
                continue
        
        logger.info(f"Successfully fetched details for {len(details)} papers")
        return details
    
    def download_pdf(self, pdf_url: str, filename: str) -> str:
        """
        Download a PDF from a given URL and save it to the specified filename.
        """
        return download_pdf(pdf_url, filename)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using pdfplumber.
        """
        return extract_text_from_pdf(pdf_path)