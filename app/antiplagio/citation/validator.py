"""
Citation Validator Module
=========================
Validates the existence and correctness of bibliographic references
by querying external academic APIs asynchronously.

APIs used:
  - CrossRef (DOI, publication metadata)
  - OpenAlex (global literature coverage)
  - Semantic Scholar (CS/AI heavy, good free API)
  - URL check (web sources)

Validation decisions:
  VALID        → Source exists and metadata matches
  PARTIAL      → Source exists but some metadata differs (year, page)
  NOT_FOUND    → Source not found in any API
  UNVERIFIABLE → Non-academic source (book, news, etc.)

Note: ValidationCache is in-memory with 24h TTL.
      For multi-process production deployments, replace with Redis cache.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

try:
    import aiohttp as _aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _aiohttp = None
    _AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed. Citation validation will be unavailable.")

from .detector import BibliographyEntry, CitationStyle


# ─────────────────────────────────────────────
# Validation Models
# ─────────────────────────────────────────────

class ValidationStatus(Enum):
    VALID = "valid"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    UNVERIFIABLE = "unverifiable"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of validating a bibliography entry."""
    entry_key: str
    status: ValidationStatus
    confidence: float                    # 0.0 – 1.0
    source_api: str = ""
    found_title: Optional[str] = None
    found_authors: list = field(default_factory=list)
    found_year: Optional[str] = None
    found_doi: Optional[str] = None
    found_url: Optional[str] = None
    discrepancies: list = field(default_factory=list)
    raw_response: Optional[dict] = None

    @property
    def is_valid(self) -> bool:
        return self.status in (ValidationStatus.VALID, ValidationStatus.PARTIAL)

    @property
    def plagiarism_penalty(self) -> float:
        """Penalty factor for plagiarism score. An invalid citation doesn't reduce risk."""
        penalties = {
            ValidationStatus.VALID: 0.0,
            ValidationStatus.PARTIAL: 0.2,
            ValidationStatus.NOT_FOUND: 0.8,
            ValidationStatus.UNVERIFIABLE: 0.3,
            ValidationStatus.ERROR: 0.4,
        }
        return penalties[self.status]


# ─────────────────────────────────────────────
# Cache (in-memory, replace with Redis for production)
# ─────────────────────────────────────────────

class ValidationCache:
    """TTL cache for validation results (avoids repeating costly API calls)."""

    def __init__(self, ttl_seconds: int = 86400):  # 24h default
        self._cache: dict = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[ValidationResult]:
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            del self._cache[key]
        return None

    def set(self, key: str, result: ValidationResult) -> None:
        self._cache[key] = (result, time.time())

    def _make_key(self, entry: BibliographyEntry) -> str:
        data = f"{entry.authors}{entry.year}{entry.title}{entry.doi}"
        return hashlib.md5(data.encode()).hexdigest()


# ─────────────────────────────────────────────
# API Clients
# ─────────────────────────────────────────────

class CrossRefClient:
    """
    CrossRef API client.
    Rate limit: ~50 req/s with courtesy email (Polite Pool).
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: str = "antiplagio@example.com"):
        self.headers = {"User-Agent": f"AntiPlagioApp/1.0 (mailto:{email})"}

    async def lookup_doi(self, doi: str, session) -> Optional[dict]:
        url = f"{self.BASE_URL}/works/{quote(doi, safe='/')}"
        try:
            async with session.get(
                url, headers=self.headers,
                timeout=_aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {})
        except Exception as e:
            logger.warning("CrossRef DOI lookup failed: %s", e)
        return None

    async def search(
        self, query: str, author: Optional[str] = None,
        year: Optional[str] = None, session=None
    ) -> list:
        params = {
            "query.bibliographic": query,
            "rows": 3,
            "sort": "relevance",
            "select": "DOI,title,author,published,container-title,URL,score"
        }
        if author:
            params["query.author"] = author
        if year:
            params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"

        url = f"{self.BASE_URL}/works"
        try:
            async with session.get(
                url, params=params, headers=self.headers,
                timeout=_aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("items", [])
        except Exception as e:
            logger.warning("CrossRef search failed: %s", e)
        return []

    def parse_work(self, work: dict) -> dict:
        title_parts = work.get("title", [])
        title = title_parts[0] if title_parts else None

        authors_data = work.get("author", [])
        authors = [
            f"{a.get('family', '')}, {a.get('given', '')[:1]}."
            for a in authors_data[:6]
        ]

        pub_date = work.get("published", {})
        date_parts = pub_date.get("date-parts", [[None]])
        year = str(date_parts[0][0]) if date_parts[0] else None

        journal_parts = work.get("container-title", [])
        journal = journal_parts[0] if journal_parts else None

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": work.get("DOI"),
            "journal": journal,
            "url": work.get("URL"),
            "score": work.get("score", 0)
        }


class OpenAlexClient:
    """
    OpenAlex API client.
    No rate limit with email (Polite Pool). 100k req/day free.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: str = "antiplagio@example.com"):
        self.params_base = {"mailto": email}

    async def search(
        self, title: Optional[str] = None, author: Optional[str] = None,
        year: Optional[str] = None, doi: Optional[str] = None, session=None
    ) -> list:
        params = dict(self.params_base)
        params["per-page"] = 3

        if doi:
            params["filter"] = f"doi:{doi}"
        elif title:
            params["search"] = title[:100]
            if year:
                params["filter"] = f"publication_year:{year}"

        url = f"{self.BASE_URL}/works"
        try:
            async with session.get(
                url, params=params,
                timeout=_aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("results", [])
        except Exception as e:
            logger.warning("OpenAlex search failed: %s", e)
        return []

    def parse_work(self, work: dict) -> dict:
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])[:6]
        ]
        journal_data = work.get("primary_location", {}) or {}
        source = journal_data.get("source") or {}

        return {
            "title": work.get("title"),
            "authors": authors,
            "year": str(work.get("publication_year", "")),
            "doi": work.get("doi", "").replace("https://doi.org/", ""),
            "journal": source.get("display_name"),
            "url": work.get("doi") or work.get("landing_page_url"),
        }


class SemanticScholarClient:
    """
    Semantic Scholar API client.
    100 req/5min without API key. Excellent for CS/AI papers.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,authors,year,externalIds,publicationVenue,openAccessPdf"

    async def search(self, query: str, year: Optional[str] = None, session=None) -> list:
        params = {"query": query[:200], "limit": 3, "fields": self.FIELDS}
        if year:
            params["year"] = f"{year}-{year}"

        url = f"{self.BASE_URL}/paper/search"
        try:
            async with session.get(
                url, params=params,
                timeout=_aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
        except Exception as e:
            logger.warning("Semantic Scholar search failed: %s", e)
        return []

    def parse_paper(self, paper: dict) -> dict:
        authors = [a.get("name", "") for a in paper.get("authors", [])[:6]]
        ext_ids = paper.get("externalIds", {})
        venue = paper.get("publicationVenue") or {}

        return {
            "title": paper.get("title"),
            "authors": authors,
            "year": str(paper.get("year", "")),
            "doi": ext_ids.get("DOI"),
            "arxiv": ext_ids.get("ArXiv"),
            "journal": venue.get("name"),
            "url": (paper.get("openAccessPdf") or {}).get("url"),
        }


# ─────────────────────────────────────────────
# Main Validator
# ─────────────────────────────────────────────

class CitationValidator:
    """
    Validates bibliographic references against multiple academic APIs.

    Validation strategy (cascade):
      1. If DOI present → CrossRef exact lookup (high confidence)
      2. If title present → parallel search CrossRef + OpenAlex + Semantic Scholar
      3. If only authors + year → fuzzy CrossRef search
      4. If URL → HTTP availability check
      5. Otherwise → UNVERIFIABLE
    """

    def __init__(
        self,
        crossref_email: str = "antiplagio@example.com",
        cache_ttl: int = 86400,
        max_concurrent: int = 5
    ):
        self.crossref = CrossRefClient(email=crossref_email)
        self.openalex = OpenAlexClient(email=crossref_email)
        self.semantic = SemanticScholarClient()
        self.cache = ValidationCache(ttl_seconds=cache_ttl)
        self.semaphore_limit = max_concurrent

    async def validate_all(self, entries: list) -> dict:
        """Validates all entries in parallel with concurrency limit."""
        if not _AIOHTTP_AVAILABLE:
            return {
                e.key: ValidationResult(
                    entry_key=e.key,
                    status=ValidationStatus.ERROR,
                    confidence=0.0,
                    source_api="none",
                    discrepancies=["aiohttp not installed"]
                )
                for e in entries
            }

        results = {}
        semaphore = asyncio.Semaphore(self.semaphore_limit)

        async with _aiohttp.ClientSession() as session:
            tasks = [
                self._validate_with_semaphore(entry, session, semaphore)
                for entry in entries
            ]
            validated = await asyncio.gather(*tasks, return_exceptions=True)

        for entry, result in zip(entries, validated):
            if isinstance(result, Exception):
                logger.error("Validation error for %s: %s", entry.key, result)
                results[entry.key] = ValidationResult(
                    entry_key=entry.key,
                    status=ValidationStatus.ERROR,
                    confidence=0.0,
                    source_api="none"
                )
            else:
                results[entry.key] = result

        return results

    async def _validate_with_semaphore(self, entry, session, semaphore) -> ValidationResult:
        async with semaphore:
            cache_key = self.cache._make_key(entry)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            result = await self._validate_entry(entry, session)
            self.cache.set(cache_key, result)
            return result

    async def _validate_entry(self, entry, session) -> ValidationResult:
        """Cascade validation strategy."""
        if entry.doi:
            result = await self._validate_by_doi(entry, session)
            if result.status != ValidationStatus.NOT_FOUND:
                return result

        if entry.title:
            result = await self._validate_by_title_parallel(entry, session)
            if result.status != ValidationStatus.NOT_FOUND:
                return result

        if entry.authors and entry.year:
            result = await self._validate_by_author_year(entry, session)
            if result.status != ValidationStatus.NOT_FOUND:
                return result

        if entry.url:
            return await self._validate_url(entry, session)

        return ValidationResult(
            entry_key=entry.key,
            status=ValidationStatus.UNVERIFIABLE,
            confidence=0.3,
            source_api="none",
            discrepancies=["No DOI, no title in academic database, no URL available"]
        )

    async def _validate_by_doi(self, entry, session) -> ValidationResult:
        work_data = await self.crossref.lookup_doi(entry.doi, session)
        if not work_data:
            return ValidationResult(
                entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                confidence=0.0, source_api="crossref"
            )

        parsed = self.crossref.parse_work(work_data)
        discrepancies = self._find_discrepancies(entry, parsed)
        status = ValidationStatus.VALID if not discrepancies else ValidationStatus.PARTIAL
        confidence = 1.0 if not discrepancies else max(0.6, 1.0 - len(discrepancies) * 0.15)

        return ValidationResult(
            entry_key=entry.key, status=status, confidence=confidence,
            source_api="crossref_doi", found_title=parsed["title"],
            found_authors=parsed["authors"], found_year=parsed["year"],
            found_doi=parsed["doi"], found_url=parsed["url"],
            discrepancies=discrepancies, raw_response=parsed
        )

    async def _validate_by_title_parallel(self, entry, session) -> ValidationResult:
        """Parallel search in CrossRef, OpenAlex, and Semantic Scholar."""
        author_str = entry.authors[0].split(',')[0] if entry.authors else None

        cr_results, oa_results, ss_results = await asyncio.gather(
            self.crossref.search(entry.title, author=author_str, year=entry.year, session=session),
            self.openalex.search(title=entry.title, author=author_str, year=entry.year, session=session),
            self.semantic.search(f"{entry.title} {author_str or ''}", year=entry.year, session=session),
            return_exceptions=True
        )

        best_match, best_score, source = None, 0.0, ""

        if isinstance(cr_results, list):
            for work in cr_results[:2]:
                parsed = self.crossref.parse_work(work)
                score = self._similarity_score(entry, parsed)
                if score > best_score:
                    best_score, best_match, source = score, parsed, "crossref"

        if isinstance(oa_results, list):
            for work in oa_results[:2]:
                parsed = self.openalex.parse_work(work)
                score = self._similarity_score(entry, parsed)
                if score > best_score:
                    best_score, best_match, source = score, parsed, "openalex"

        if isinstance(ss_results, list):
            for paper in ss_results[:2]:
                parsed = self.semantic.parse_paper(paper)
                score = self._similarity_score(entry, parsed)
                if score > best_score:
                    best_score, best_match, source = score, parsed, "semantic_scholar"

        if not best_match or best_score < 0.45:
            return ValidationResult(
                entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                confidence=0.0, source_api="all_apis"
            )

        discrepancies = self._find_discrepancies(entry, best_match)
        status = ValidationStatus.VALID if best_score >= 0.80 else ValidationStatus.PARTIAL

        return ValidationResult(
            entry_key=entry.key, status=status, confidence=best_score,
            source_api=source, found_title=best_match.get("title"),
            found_authors=best_match.get("authors", []),
            found_year=best_match.get("year"), found_doi=best_match.get("doi"),
            found_url=best_match.get("url"), discrepancies=discrepancies,
            raw_response=best_match
        )

    async def _validate_by_author_year(self, entry, session) -> ValidationResult:
        author = entry.authors[0].split(',')[0] if entry.authors else ""
        query = f"{author} {entry.year} {entry.raw_text[:100]}"

        results = await self.crossref.search(
            query, author=author, year=entry.year, session=session
        )

        if not results:
            return ValidationResult(
                entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                confidence=0.0, source_api="crossref"
            )

        parsed = self.crossref.parse_work(results[0])
        score = self._similarity_score(entry, parsed)

        if score < 0.35:
            return ValidationResult(
                entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                confidence=0.0, source_api="crossref"
            )

        return ValidationResult(
            entry_key=entry.key, status=ValidationStatus.PARTIAL,
            confidence=score * 0.7,  # reduced confidence for imprecise search
            source_api="crossref_fuzzy", found_title=parsed.get("title"),
            found_authors=parsed.get("authors", []), found_year=parsed.get("year"),
            found_doi=parsed.get("doi")
        )

    async def _validate_url(self, entry, session) -> ValidationResult:
        try:
            async with session.head(
                entry.url, allow_redirects=True,
                timeout=_aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status < 400:
                    return ValidationResult(
                        entry_key=entry.key, status=ValidationStatus.PARTIAL,
                        confidence=0.60, source_api="url_check",
                        found_url=str(resp.url),
                        discrepancies=["Web source verified by availability only"]
                    )
                return ValidationResult(
                    entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                    confidence=0.0, source_api="url_check",
                    discrepancies=[f"URL returned HTTP {resp.status}"]
                )
        except Exception as e:
            return ValidationResult(
                entry_key=entry.key, status=ValidationStatus.NOT_FOUND,
                confidence=0.0, source_api="url_check",
                discrepancies=[f"URL unreachable: {type(e).__name__}"]
            )

    def _similarity_score(self, entry, found: dict) -> float:
        """Similarity score: title 50%, author 30%, year 20%."""
        score = 0.0

        if entry.title and found.get("title"):
            title_sim = self._text_similarity(
                entry.title.lower()[:150], found["title"].lower()[:150]
            )
            score += title_sim * 0.50

        if entry.authors and found.get("authors"):
            entry_lastname = self._normalize(entry.authors[0].split(',')[0])
            found_lastnames = [self._normalize(a.split(',')[0]) for a in found["authors"]]
            author_match = max(
                (self._text_similarity(entry_lastname, fl) for fl in found_lastnames),
                default=0.0
            )
            score += author_match * 0.30

        if entry.year and found.get("year"):
            score += 0.20 if entry.year == found["year"] else 0.0

        return min(score, 1.0)

    def _find_discrepancies(self, entry, found: dict) -> list:
        discrepancies = []

        if entry.year and found.get("year") and entry.year != found["year"]:
            discrepancies.append(
                f"Year mismatch: declared '{entry.year}', found '{found['year']}'"
            )

        if entry.title and found.get("title"):
            title_sim = self._text_similarity(entry.title.lower(), found["title"].lower())
            if title_sim < 0.70:
                discrepancies.append(f"Title differs: {title_sim:.0%} similarity")

        if entry.doi and found.get("doi"):
            if entry.doi.strip('/').lower() != found["doi"].strip('/').lower():
                discrepancies.append("Declared DOI does not match found DOI")

        return discrepancies

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        tokens_a = set(re.findall(r'\w+', a.lower()))
        tokens_b = set(re.findall(r'\w+', b.lower()))
        stopwords = {
            'the', 'a', 'an', 'of', 'in', 'on', 'for', 'and', 'or',
            'el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'e', 'un', 'una'
        }
        tokens_a -= stopwords
        tokens_b -= stopwords
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'ü': 'u'
        }
        return ''.join(replacements.get(c, c) for c in text)
