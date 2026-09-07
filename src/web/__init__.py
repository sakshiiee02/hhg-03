"""Web extraction and candidate download package."""

from src.web.downloader import CandidateDownloader, DownloadedCandidate
from src.web.extract import PageMetadata, extract_page_metadata

__all__ = [
    "CandidateDownloader",
    "DownloadedCandidate",
    "PageMetadata",
    "extract_page_metadata",
]
