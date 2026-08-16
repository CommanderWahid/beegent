"""Pluggable search/fetch backend - the abstraction a new tool provider implements.

search_explore.py's agent needs exactly two capabilities: search the web, and fetch a
page's content and outbound links. Everything backend-specific (which API, what auth, how
results map onto these shapes) lives behind SearchBackend, so swapping providers means
writing one new class in one new core/search_backends/<name>.py file - it's discovered
automatically because it lives in this directory (see _discover_backends() below),
nothing else to touch. This module never imports or names a concrete backend.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from core import config


@dataclass
class SearchHit:
    title: str
    url: str
    content: str = ""  # snippet, if the backend provides one - "" is a valid, handled answer


@dataclass
class Link:
    text: str
    url: str


@dataclass
class FetchedPage:
    url: str
    text: str
    links: list[Link] = field(default_factory=list)


class SearchBackend(ABC):
    """What search_explore.py's agent needs from a search/fetch provider.

    Raise on failure rather than returning an empty/partial result - explore_angle()
    already treats a raised exception as "this tool call failed" and reports it in the
    tool result; a swallowed error just looks like a legitimately empty search or page.
    """

    name: ClassVar[str]  # short identifier matched against config.SEARCH_BACKEND

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must set a class-level `name`")

    @abstractmethod
    def web_search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        """Search the web. A content snippet is a bonus, not a requirement - callers treat
        a missing one (empty string) as normal."""

    @abstractmethod
    def fetch_page(self, url: str, max_chars: int = 3000, max_links: int = 30) -> FetchedPage:
        """Fetch one page's text and outbound links."""


def _all_subclasses(cls: type) -> Iterator[type]:
    """Every subclass, any depth - not just __subclasses__()'s direct children, so a
    backend author can share code through an intermediate abstract base without breaking
    discovery."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


_registry: dict[str, type[SearchBackend]] | None = None  # populated once, lazily


def _discover_backends() -> dict[str, type[SearchBackend]]:
    """Import every module in this directory once, so each SearchBackend subclass it
    defines registers itself just by existing - no file in this codebase has to name
    another backend by hand. Uses __package__ ("core.search_backends") rather than a
    hardcoded name, so this keeps working even if the top-level package is ever renamed.

    A module that fails to import (e.g. a missing optional dependency for a provider
    nobody's using) is skipped with a log line rather than breaking discovery for every
    other backend; if it happens to be the one actually selected, get_search_backend()'s
    "unknown backend" error still fires, with the real import error visible just above it.
    """
    global _registry
    if _registry is None:
        here = Path(__file__).parent
        for path in here.glob("*.py"):
            if path.stem in ("base", "__init__"):
                continue
            try:
                importlib.import_module(f"{__package__}.{path.stem}")
            except ImportError as exc:
                print(f"  [search_backends] skipping {path.stem}: {exc}")
        _registry = {
            cls.name: cls for cls in _all_subclasses(SearchBackend) if getattr(cls, "name", None)
        }
    return _registry


def get_search_backend() -> SearchBackend:
    """Instantiate the backend named by config.SEARCH_BACKEND.

    Add a new backend by dropping a new core/search_backends/<name>.py file that
    subclasses SearchBackend and sets `name` - it's found just by living in this
    directory and registers itself just by being importable. Nothing in this file names a
    concrete backend, and nothing else needs to change to add one.
    """
    backends = _discover_backends()
    try:
        return backends[config.SEARCH_BACKEND]()
    except KeyError:
        raise ValueError(
            f"unknown SEARCH_BACKEND: {config.SEARCH_BACKEND!r} "
            f"(available: {', '.join(sorted(backends)) or 'none found'})"
        ) from None
