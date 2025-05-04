import logging

def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    Returns a logger instance.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("paper_scraper")