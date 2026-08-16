"""Pluggable search/fetch backends - see base.py for the SearchBackend interface."""

from .base import FetchedPage, Link, SearchBackend, SearchHit, get_search_backend

__all__ = ["SearchBackend", "SearchHit", "FetchedPage", "Link", "get_search_backend"]
