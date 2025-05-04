import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# --- PubMed XML Parsing Functions ---

def get_doi(soup: BeautifulSoup) -> Optional[str]:
    """Get DOI from article."""
    article_ids = soup.find("ArticleIdList") or soup.find("PubmedData")
    if article_ids:
        doi_tag = article_ids.find("ArticleId", {"IdType": "doi"})
        if doi_tag:
            return doi_tag.text.strip()
    return None

def get_title(soup: BeautifulSoup) -> str:
    """Get article title."""
    title_tag = soup.find("ArticleTitle")
    return title_tag.text.strip() if title_tag else ""

def get_abstract(soup: BeautifulSoup) -> str:
    """Get article abstract."""
    abstract = soup.find("Abstract")
    if not abstract:
        return ""
    
    sections = abstract.find_all("AbstractText")
    if not sections:
        return ""
    
    # Handle structured abstracts
    if any(section.get("Label") for section in sections):
        return "\n".join(
            f"{section.get('Label', 'Abstract')}: {section.text.strip()}"
            for section in sections
        )
    
    # Handle simple abstracts
    return " ".join(section.text.strip() for section in sections)

def get_authors(soup: BeautifulSoup) -> List[Dict]:
    """Get article authors."""
    author_list = soup.find("AuthorList")
    if not author_list:
        return []
    
    authors = []
    for author in author_list.find_all("Author"):
        if author.find("CollectiveName"):
            authors.append({
                "collective_name": author.find("CollectiveName").text.strip(),
                "lastname": "",
                "firstname": "",
                "affiliation": ""
            })
        else:
            lastname = author.find("LastName")
            firstname = author.find("ForeName")
            affiliation = author.find("Affiliation")
            
            authors.append({
                "lastname": lastname.text.strip() if lastname else "",
                "firstname": firstname.text.strip() if firstname else "",
                "affiliation": affiliation.text.strip() if affiliation else ""
            })
    
    return authors

def get_journal_info(soup: BeautifulSoup) -> Dict:
    """Get journal information."""
    journal = soup.find("Journal")
    if not journal:
        return {}
    
    return {
        "name": (journal.find("Title").text.strip() if journal.find("Title") 
                else journal.find("ISOAbbreviation").text.strip() if journal.find("ISOAbbreviation")
                else ""),
        "issn": journal.find("ISSN").text.strip() if journal.find("ISSN") else "",
        "volume": journal.find("Volume").text.strip() if journal.find("Volume") else "",
        "issue": journal.find("Issue").text.strip() if journal.find("Issue") else "",
    }

def get_pub_date(soup: BeautifulSoup) -> Optional[str]:
    """Get publication date."""
    pub_date = (
        soup.find("PubDate") or 
        soup.find("DateCompleted") or 
        soup.find("DateRevised")
    )
    
    if not pub_date:
        return None
    
    if pub_date.find("MedlineDate"):
        return pub_date.find("MedlineDate").text.strip()
    
    year = pub_date.find("Year")
    month = pub_date.find("Month")
    day = pub_date.find("Day")
    
    if year:
        date_parts = [year.text.strip()]
        if month:
            date_parts.append(month.text.strip().zfill(2))
            if day:
                date_parts.append(day.text.strip().zfill(2))
        return "-".join(date_parts)
    
    return None

def get_full_text_link(soup: BeautifulSoup, unpaywall_email: Optional[str] = None) -> Optional[str]:
    """
    Try to obtain a full text PDF link.
    First, check for a PMC link; if none, attempt to resolve via DOI.
    """
    pmid = soup.find("Id").text if soup.find("Id") else "Unknown"
    logger.debug(f"Attempting to get full text link for PMID {pmid}")
    
    # Log all available IDs for debugging
    for id_tag in soup.find_all(["ArticleId", "OtherID"]):
        logger.debug(f"Found ID: {id_tag.get('IdType')} = {id_tag.text}")

    # Check for PMC ID in multiple locations
    pmc_id = None
    for id_tag in soup.find_all(["ArticleId", "OtherID"]):
        if id_tag.get("IdType") == "pmc" or id_tag.get("Source") == "PMC":
            pmc_id = id_tag.text.strip().replace("PMC", "")
            logger.debug(f"Found PMC ID: {pmc_id}")
            break

    if pmc_id:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf"
        logger.debug(f"Generated PMC PDF URL: {pdf_url}")
        return pdf_url

    # Try DOI resolution
    doi = get_doi(soup)
    if doi:
        logger.debug(f"Found DOI: {doi}")
        pdf_url = resolve_doi_to_pdf(doi, unpaywall_email)
        if pdf_url:
            logger.debug(f"Resolved DOI to PDF: {pdf_url}")
            return pdf_url
        else:
            logger.debug("Failed to resolve DOI to PDF")

    # Try looking for direct links in the Article element
    article = soup.find("Article")
    if article:
        logger.debug("Searching for links in Article element")
        for link in article.find_all(["ELocationID", "Link"]):
            url = link.get("URL", "") or link.text.strip()
            logger.debug(f"Found link: {url}")
            if url and (".pdf" in url.lower() or "fulltext" in url.lower()):
                return url

    logger.debug("No full text link found through any method")
    return None

def resolve_doi_to_pdf(doi: str, unpaywall_email: Optional[str] = None) -> Optional[str]:
    """
    Resolve a DOI to a direct PDF link via multiple services
    """
    if not doi:
        return None
    
    logger.debug(f"Attempting to resolve DOI: {doi}")
    
    # Check if DOI is from a known publisher
    publisher_patterns = {
        "10.1016": "Elsevier",
        "10.1038": "Nature",
        "10.1093": "Oxford",
        "10.1007": "Springer",
        "10.1111": "Wiley",
        "10.1371": "PLOS",
    }
    
    for pattern, publisher in publisher_patterns.items():
        if pattern in doi:
            logger.debug(f"Identified publisher: {publisher}")
            break
        
    # Try direct publisher PDF links first
    if "10.1016" in doi:  # Elsevier
        pdf_url = f"https://www.sciencedirect.com/science/article/pii/{doi.split('/')[-1]}/pdfft"
        logger.debug(f"Trying Elsevier direct PDF: {pdf_url}")
        try:
            response = requests.head(pdf_url, allow_redirects=True, timeout=10)
            if response.ok and "pdf" in response.headers.get("Content-Type", "").lower():
                return pdf_url
        except Exception as e:
            logger.debug(f"Elsevier PDF attempt failed: {e}")

    # Try Unpaywall
    if unpaywall_email:
        logger.debug("Trying Unpaywall API")
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={unpaywall_email}"
        try:
            response = requests.get(unpaywall_url, timeout=10)
            if response.ok:
                data = response.json()
                logger.debug(f"Unpaywall response: {data.get('best_oa_location')}")
                best_location = data.get("best_oa_location", {})
                if best_location:
                    pdf_url = best_location.get("pdf_url") or best_location.get("url")
                    if pdf_url and (pdf_url.endswith(".pdf") or "pdf" in pdf_url.lower()):
                        return pdf_url
        except Exception as e:
            logger.debug(f"Unpaywall API error: {e}")

    # Try DOI resolution
    try:
        logger.debug("Trying DOI resolution")
        headers = {"Accept": "text/html,application/pdf"}
        response = requests.get(f"https://doi.org/{doi}", headers=headers, allow_redirects=True)
        if response.ok:
            final_url = response.url
            logger.debug(f"DOI resolves to: {final_url}")
            if final_url.endswith(".pdf"):
                return final_url
    except Exception as e:
        logger.debug(f"DOI resolution failed: {e}")

    return None

def extract_pdf_from_html(html: str, base_url: str) -> Optional[str]:
    """
    Attempt to extract a PDF link from HTML content.
    Returns the first valid PDF URL found or None if no PDF link is found.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Expanded patterns for PDF link detection
    pdf_patterns = [
        # Common PDF-related text in links
        lambda x: x and any(pattern in x.lower() for pattern in [
            '.pdf', 
            'pdf',
            'full-text',
            'fulltext',
            'download',
            'article',
            'view',
            'access'
        ]),
        # Common PDF-related classes/IDs
        lambda x: x and any(pattern in x.lower() for pattern in [
            'pdf-link',
            'download-pdf',
            'article-pdf',
            'full-text-pdf'
        ])
    ]
    
    # Find links matching our patterns
    pdf_links = []
    for pattern in pdf_patterns:
        pdf_links.extend(soup.find_all('a', href=pattern))
        pdf_links.extend(soup.find_all('a', class_=pattern))
        pdf_links.extend(soup.find_all('a', id=pattern))
    
    # Check each potential PDF link
    for link in pdf_links:
        href = link.get('href', '')
        if not href:
            continue
            
        # Clean and normalize the URL
        pdf_url = requests.compat.urljoin(base_url, href.strip())
        
        try:
            # Use a timeout to avoid hanging
            head = requests.head(pdf_url, 
                              allow_redirects=True, 
                              timeout=10,
                              headers={'User-Agent': 'Mozilla/5.0'})
            
            # Check both Content-Type and URL patterns
            is_pdf = (
                'pdf' in head.headers.get('Content-Type', '').lower() or
                pdf_url.lower().endswith('.pdf')
            )
            
            if is_pdf:
                logger.info(f"Found valid PDF URL: {pdf_url}")
                return pdf_url
                
        except Exception as e:
            logger.debug(f"Failed to verify PDF URL {pdf_url}: {str(e)}")
            continue
            
    logger.debug("No valid PDF links found in HTML content")
    return None