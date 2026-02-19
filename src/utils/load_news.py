"""
Utility to load news data from various sources.

Can be extended to load from GDELT, news APIs, etc.
"""

import json
from typing import Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_news_from_jsonl(file_path: str) -> Dict[str, List[str]]:
    """
    Load news articles from JSONL file.
    
    Expected format:
    {"date": "2024-01-15", "text": "News article text..."}
    
    Returns:
        Dictionary mapping date strings to lists of news articles
    """
    news_data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    date = item.get('date', '')
                    text = item.get('text', '')
                    
                    if date and text:
                        if date not in news_data:
                            news_data[date] = []
                        news_data[date].append(text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
        
        logger.info(f"Loaded {len(news_data)} dates with news articles from {file_path}")
        return news_data
        
    except FileNotFoundError:
        logger.warning(f"News file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading news from {file_path}: {e}")
        return {}


def create_sample_news_data() -> Dict[str, List[str]]:
    """
    Create sample news data for testing.
    
    Returns:
        Dictionary with sample news articles
    """
    return {
        "2024-01-15": [
            "Apple announced record iPhone sales in Q4, exceeding analyst expectations.",
            "Microsoft reports strong cloud growth, Azure revenue up 30% year-over-year."
        ],
        "2024-01-16": [
            "Tesla faces new lawsuit over autopilot safety claims from regulators.",
            "Amazon announces layoffs of 10,000 employees in AWS division."
        ],
        "2024-01-17": [
            "Nvidia partners with major cloud providers to expand AI chip distribution.",
            "Meta announces hiring freeze due to slowing ad revenue growth."
        ]
    }
