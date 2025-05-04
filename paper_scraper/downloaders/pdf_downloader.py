import logging
from time import sleep

import requests
import pdfplumber

logger = logging.getLogger(__name__)

def download_pdf(pdf_url: str, filename: str) -> str:
    """
    Download a PDF from a given URL and save it to the specified filename.
    """
    logger.info(f"Downloading PDF from {pdf_url}")
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,*/*",
    }
    session = requests.Session()
    try:
        base_url = '/'.join(pdf_url.split('/')[:3])
        headers["Referer"] = base_url
        session.get(base_url, headers=headers, timeout=10)
    except Exception as e:
        logger.warning(f"Failed to pre-load cookies: {e}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = session.get(pdf_url, headers=headers, allow_redirects=True, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) + 1
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                sleep(wait_time)
            else:
                raise
    
    content_type = response.headers.get('Content-Type', '').lower()
    if 'pdf' not in content_type:
        logger.warning(f"Received non-PDF content ({content_type}) from {pdf_url}")
        raise ValueError(f"Expected PDF content but received {content_type}")
    
    if not response.content.startswith(b'%PDF-'):
        logger.warning(f"Content from {pdf_url} does not appear to be a valid PDF")
        raise ValueError("Downloaded content is not a valid PDF file")
    
    with open(filename, "wb") as f:
        f.write(response.content)
    logger.info(f"PDF saved to {filename}")
    return filename

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    """
    logger.info(f"Extracting text from {pdf_path}")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""
    logger.info(f"Successfully extracted text from {pdf_path}")
    return text