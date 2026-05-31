"""
Citation Detector Module
========================
Detects and classifies citations in academic text across all major styles.
Handles: APA, MLA, IEEE, Chicago, Vancouver, Harvard, and hybrid documents.

Architecture:
  1. Segment text into ZONES (quoted, paraphrased, original)
  2. Detect citation markers per zone
  3. Locate and parse bibliography/references section
  4. Cross-link inline citations → bibliography entries
"""

import bisect
import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_ACCENT_MAP: dict = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'ü': 'u',
    'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u', 'Ñ': 'n', 'Ü': 'u',
}


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class CitationStyle(Enum):
    APA = "APA"
    MLA = "MLA"
    IEEE = "IEEE"
    CHICAGO = "Chicago"
    VANCOUVER = "Vancouver"
    HARVARD = "Harvard"
    UNKNOWN = "Unknown"


class ZoneType(Enum):
    DIRECT_QUOTE = "direct_quote"        # "texto entre comillas" con cita
    BLOCK_QUOTE = "block_quote"          # Cita en bloque (sangría)
    PARAPHRASE = "paraphrase"            # Idea ajena reformulada con cita
    ORIGINAL = "original"               # Contenido propio del autor
    BIBLIOGRAPHY = "bibliography"        # Sección de referencias


@dataclass
class CitationMarker:
    """Un marcador de cita encontrado en el texto (inline)."""
    raw_text: str
    start_pos: int
    end_pos: int
    style: CitationStyle
    author: Optional[str] = None
    year: Optional[str] = None
    page: Optional[str] = None
    number: Optional[int] = None        # Para IEEE [1], Vancouver (1)
    confidence: float = 1.0


@dataclass
class BibliographyEntry:
    """Una entrada de bibliografía parseada."""
    raw_text: str
    key: str                             # Clave normalizada para linking
    style: CitationStyle
    authors: list = field(default_factory=list)
    year: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    number: Optional[int] = None        # Para citas numeradas


@dataclass
class TextZone:
    """Un segmento del texto clasificado por tipo."""
    text: str
    start_pos: int
    end_pos: int
    zone_type: ZoneType
    citation_markers: list = field(default_factory=list)
    linked_bibliography: list = field(default_factory=list)
    has_valid_citation: bool = False
    plagiarism_risk: float = 1.0        # 0.0 = sin riesgo, 1.0 = alto riesgo


@dataclass
class CitationAnalysisResult:
    """Resultado completo del análisis de citas de un documento."""
    zones: list
    bibliography: list
    dominant_style: CitationStyle
    inline_citations: list
    orphan_citations: list              # Citas sin entrada en bibliografía
    uncited_bibliography: list          # Entradas no referenciadas en texto
    citation_coverage: float            # % de zonas no-originales con cita válida
    style_consistency: float            # % de consistencia en estilo de citas


# ─────────────────────────────────────────────
# Patrones Regex por Estilo
# ─────────────────────────────────────────────

class CitationPatterns:
    """
    Patrones compilados para cada estilo de citación.
    Ordered de mayor a menor especificidad para evitar falsos positivos.
    """

    # APA: (Autor, 2020) | (Autor & Otro, 2020, p. 45) | (Autor et al., 2020)
    APA_INLINE = re.compile(
        r'\(([A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z\-\']+(?:\s+(?:&|and|y)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+)?'
        r'(?:\s+et\s+al\.)?),\s*((?:19|20)\d{2}[a-z]?)(?:,\s*p+p?\.\s*(\d+(?:\-\d+)?))?\)',
        re.UNICODE
    )

    # APA narrativo: Autor (2020) afirma que...
    APA_NARRATIVE = re.compile(
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z\-\']+(?:\s+(?:&|and|y)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+)?'
        r'(?:\s+et\s+al\.)?)\s+\(((?:19|20)\d{2}[a-z]?)(?:,\s*p+p?\.\s*(\d+(?:\-\d+)?))?\)',
        re.UNICODE
    )

    # MLA: (Apellido número_página) sin coma
    MLA_INLINE = re.compile(
        r'\(([A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+(?:\s+(?:and|y)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+)?'
        r'(?:\s+et\s+al\.)?)\s+(\d+(?:\-\d+)?)\)',
        re.UNICODE
    )

    # APA multi-citation: (Author, Year; Author & Other, Year; ...)
    # Finds parentheticals containing semicolons for split-processing
    APA_MULTI_PAREN = re.compile(
        r'\(([A-ZÁÉÍÓÚÑ][^)]+;\s*[A-ZÁÉÍÓÚÑ][^)]+)\)',
        re.UNICODE
    )

    # IEEE / Vancouver: [1] | [1,2] | [1-3]
    IEEE_INLINE = re.compile(
        r'\[(\d+(?:\s*[,\-]\s*\d+)*)\]'
    )

    # Vancouver bracket: (1) in bibliographic context
    VANCOUVER_BRACKET = re.compile(
        r'\((\d+(?:\s*[,\-]\s*\d+)*)\)(?=[\s,\.\;])'
    )

    # Chicago: superscript footnotes ¹²³ or ^1 or [^1]
    CHICAGO_FOOTNOTE = re.compile(
        r'(?:[¹²³⁴-⁹]|\^\d+|\[\^(\d+)\])'
    )

    # Harvard: like APA with British variants
    HARVARD_INLINE = re.compile(
        r'\(([A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z\-\']+(?:\s+(?:&|and)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+)?'
        r'(?:\s+et\s+al\.)?),?\s*((?:19|20)\d{2}[a-z]?)(?:[:\s]+p+p?\.\s*(\d+(?:\-\d+)?))?\)',
        re.UNICODE
    )

    # DOI universal
    DOI = re.compile(
        r'\b(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,}/\S+)',
        re.IGNORECASE
    )

    # URL
    URL = re.compile(
        r'https?://[^\s\)\]>\"\']+',
        re.IGNORECASE
    )

    # Direct quote markers (multiple international styles)
    DIRECT_QUOTE = re.compile(
        r'(?:'
        r'"([^"]{10,500})"'
        r'|“([^”]{10,500})”'
        r"|‘([^’]{10,500})’"
        r'|«([^»]{10,500})»'
        r'|„([^“]{10,500})“'
        r')',
        re.DOTALL | re.UNICODE
    )

    # Block quote: indented paragraph
    BLOCK_QUOTE_MARKER = re.compile(
        r'^(?:\s{4,}|\t{1,})(.+?)$',
        re.MULTILINE | re.DOTALL
    )

    # Bibliography/references section header
    BIBLIOGRAPHY_HEADER = re.compile(
        r'^(?:Referencias?|References?|Bibliograf[íi]a|Bibliography|'
        r'Obras\s+[Cc]itadas?|Works?\s+[Cc]ited|Fuentes?|Sources?|'
        r'Literatura\s+[Cc]itada|Notas?\s+y\s+Referencias?)\s*$',
        re.MULTILINE | re.IGNORECASE
    )


# ─────────────────────────────────────────────
# Text Segmenter
# ─────────────────────────────────────────────

class TextSegmenter:
    """
    Divides text into semantic zones before citation analysis.
    Falls back to rule-based segmentation when spaCy is unavailable.
    """

    def __init__(self):
        self.patterns = CitationPatterns()

    def segment(self, text: str) -> tuple:
        """
        Segments text and extracts the bibliography section.

        Returns:
            (zones, body_text, bibliography_text)
        """
        body_text, bibliography_text = self._split_bibliography(text)

        zones = []
        pos = 0

        for match in self.patterns.DIRECT_QUOTE.finditer(body_text):
            if match.start() > pos:
                preceding = body_text[pos:match.start()]
                if preceding.strip():
                    zones.append(TextZone(
                        text=preceding,
                        start_pos=pos,
                        end_pos=match.start(),
                        zone_type=ZoneType.ORIGINAL
                    ))

            quoted_text = next(g for g in match.groups() if g is not None)
            zones.append(TextZone(
                text=quoted_text,
                start_pos=match.start(),
                end_pos=match.end(),
                zone_type=ZoneType.DIRECT_QUOTE
            ))
            pos = match.end()

        if pos < len(body_text):
            remaining = body_text[pos:]
            if remaining.strip():
                zones.append(TextZone(
                    text=remaining,
                    start_pos=pos,
                    end_pos=len(body_text),
                    zone_type=ZoneType.ORIGINAL
                ))

        if not zones:
            zones.append(TextZone(
                text=body_text,
                start_pos=0,
                end_pos=len(body_text),
                zone_type=ZoneType.ORIGINAL
            ))

        if bibliography_text.strip():
            zones.append(TextZone(
                text=bibliography_text,
                start_pos=len(body_text),
                end_pos=len(text),
                zone_type=ZoneType.BIBLIOGRAPHY
            ))

        return zones, body_text, bibliography_text

    def _split_bibliography(self, text: str) -> tuple:
        """Separates body text from the references section."""
        match = CitationPatterns.BIBLIOGRAPHY_HEADER.search(text)
        if match:
            return text[:match.start()].strip(), text[match.start():].strip()
        return text, ""


# ─────────────────────────────────────────────
# Citation Detector (main)
# ─────────────────────────────────────────────

class CitationDetector:
    """
    Main citation detection engine.

    Pipeline:
        text → segment → detect_inline_citations → parse_bibliography
             → cross_link → score_zones → CitationAnalysisResult
    """

    def __init__(self):
        self.patterns = CitationPatterns()
        self.segmenter = TextSegmenter()

    def analyze(self, text: str) -> CitationAnalysisResult:
        """Full citation analysis of a text."""
        zones, body_text, bibliography_text = self.segmenter.segment(text)

        all_inline_citations = []
        for zone in zones:
            if zone.zone_type != ZoneType.BIBLIOGRAPHY:
                citations = self._detect_inline_citations(zone.text, zone.start_pos)
                zone.citation_markers = citations
                all_inline_citations.extend(citations)

        bibliography = self._parse_bibliography(bibliography_text)
        dominant_style = self._detect_dominant_style(all_inline_citations, bibliography)

        orphan_citations, uncited_bibliography = self._cross_link(
            all_inline_citations, bibliography, zones
        )

        self._score_zones(zones)

        citation_coverage = self._compute_citation_coverage(zones)
        style_consistency = self._compute_style_consistency(all_inline_citations)

        return CitationAnalysisResult(
            zones=zones,
            bibliography=bibliography,
            dominant_style=dominant_style,
            inline_citations=all_inline_citations,
            orphan_citations=orphan_citations,
            uncited_bibliography=uncited_bibliography,
            citation_coverage=citation_coverage,
            style_consistency=style_consistency
        )

    def _detect_inline_citations(self, text: str, offset: int = 0) -> list:
        """Detects all citation markers in a text fragment."""
        citations = []

        # APA multi-citation: (Author, Year; Author, Year; ...)
        # Process these first so individual APA patterns don't miss them
        _multi_spans = set()
        for m in self.patterns.APA_MULTI_PAREN.finditer(text):
            _multi_spans.add((m.start(), m.end()))
            parts = re.split(r';\s*', m.group(1))
            for part in parts:
                part = part.strip()
                # Try to parse "Author, Year" or "Author & Other, Year"
                sub = re.match(
                    r'([A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z\-\']+(?:\s+(?:&|and|y)\s+'
                    r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+)?(?:\s+et\s+al\.)?)'
                    r',\s*((?:19|20)\d{2}[a-z]?)(?:,\s*p+p?\.\s*(\d+(?:\-\d+)?))?$',
                    part, re.UNICODE
                )
                if sub:
                    citations.append(CitationMarker(
                        raw_text=f"({part})",
                        start_pos=m.start() + offset,
                        end_pos=m.end() + offset,
                        style=CitationStyle.APA,
                        author=sub.group(1),
                        year=sub.group(2),
                        page=sub.group(3),
                        confidence=0.92
                    ))

        # APA parenthetical (skip spans already handled as multi-citations)
        for m in self.patterns.APA_INLINE.finditer(text):
            if (m.start(), m.end()) in _multi_spans:
                continue
            citations.append(CitationMarker(
                raw_text=m.group(0),
                start_pos=m.start() + offset,
                end_pos=m.end() + offset,
                style=CitationStyle.APA,
                author=m.group(1),
                year=m.group(2),
                page=m.group(3),
                confidence=0.95
            ))

        # APA narrative (skip if already captured)
        existing_spans = {(c.start_pos, c.end_pos) for c in citations}
        for m in self.patterns.APA_NARRATIVE.finditer(text):
            span = (m.start() + offset, m.end() + offset)
            if span not in existing_spans:
                citations.append(CitationMarker(
                    raw_text=m.group(0),
                    start_pos=span[0],
                    end_pos=span[1],
                    style=CitationStyle.APA,
                    author=m.group(1),
                    year=m.group(2),
                    page=m.group(3),
                    confidence=0.90
                ))

        # MLA (skip if overlapping with APA) — O(1) span-set lookup
        _occupied: set = {(c.start_pos, c.end_pos) for c in citations}
        for m in self.patterns.MLA_INLINE.finditer(text):
            m_start = m.start() + offset
            if not any(s <= m_start < e for s, e in _occupied):
                c = CitationMarker(
                    raw_text=m.group(0),
                    start_pos=m_start,
                    end_pos=m.end() + offset,
                    style=CitationStyle.MLA,
                    author=m.group(1),
                    page=m.group(2),
                    confidence=0.80
                )
                citations.append(c)
                _occupied.add((c.start_pos, c.end_pos))

        # IEEE
        for m in self.patterns.IEEE_INLINE.finditer(text):
            nums = re.findall(r'\d+', m.group(1))
            for num in nums:
                citations.append(CitationMarker(
                    raw_text=m.group(0),
                    start_pos=m.start() + offset,
                    end_pos=m.end() + offset,
                    style=CitationStyle.IEEE,
                    number=int(num),
                    confidence=0.98
                ))

        # Chicago footnotes
        for m in self.patterns.CHICAGO_FOOTNOTE.finditer(text):
            citations.append(CitationMarker(
                raw_text=m.group(0),
                start_pos=m.start() + offset,
                end_pos=m.end() + offset,
                style=CitationStyle.CHICAGO,
                confidence=0.85
            ))

        return citations

    def _parse_bibliography(self, bib_text: str) -> list:
        """Parses bibliography entries, auto-detecting style."""
        if not bib_text.strip():
            return []

        entries = []
        raw_entries = self._split_bibliography_entries(bib_text)

        for idx, raw in enumerate(raw_entries):
            raw = raw.strip()
            if not raw or len(raw) < 20:
                continue
            entry = self._parse_single_entry(raw, idx + 1)
            if entry:
                entries.append(entry)

        return entries

    def parse_bibliography(self, bib_text: str) -> list:
        """Public wrapper for _parse_bibliography."""
        return self._parse_bibliography(bib_text)

    def _split_bibliography_entries(self, bib_text: str) -> list:
        """Splits bibliography block into individual entries."""
        lines = bib_text.split('\n')
        content_lines = []
        skip_header = True

        for line in lines:
            if skip_header and CitationPatterns.BIBLIOGRAPHY_HEADER.match(line.strip()):
                skip_header = False
                continue
            if not skip_header or content_lines:
                content_lines.append(line)

        text = '\n'.join(content_lines)

        # Strategy 1: numbered entries [1] or 1.
        numbered = re.split(r'(?:^|\n)(?:\[?\d+\]?\.?\s+)', text)
        if len(numbered) > 2:
            return [e for e in numbered if e.strip()]

        # Strategy 2: blank-line separated
        blank_separated = re.split(r'\n\s*\n', text)
        if len(blank_separated) > 1:
            return [e for e in blank_separated if e.strip()]

        # Strategy 3: one entry per line
        return [line for line in text.split('\n') if line.strip() and len(line.strip()) > 20]

    def _parse_single_entry(self, raw: str, number: int) -> Optional[BibliographyEntry]:
        """Parses a single bibliography entry."""
        doi_match = CitationPatterns.DOI.search(raw)
        doi = doi_match.group(1) if doi_match else None

        url_match = CitationPatterns.URL.search(raw)
        url = url_match.group(0) if url_match else None

        year_match = re.search(r'\b((?:19|20)\d{2})\b', raw)
        year = year_match.group(1) if year_match else None

        authors = self._extract_authors(raw)

        title_match = re.search(r'[“”"]([^"“”]{10,200})[“”"]', raw)
        if not title_match:
            title_match = re.search(r'\*([^*]{10,200})\*', raw)
        title = title_match.group(1) if title_match else None

        journal_patterns = [
            r'(?:In|En)\s+([A-ZÁÉÍÓÚÑ][^,\.]{5,60}),',
            r'([A-ZÁÉÍÓÚÑ][^,\.]{5,60}),\s+\d+\(',
        ]
        journal = None
        for pat in journal_patterns:
            j_match = re.search(pat, raw, re.UNICODE)
            if j_match:
                journal = j_match.group(1).strip()
                break

        key = self._generate_entry_key(authors, year, number)
        style = self._detect_entry_style(raw, authors, year)

        pages_match = re.search(r'pp?\.\s*(\d+(?:\s*[-–]\s*\d+)?)', raw, re.IGNORECASE)
        pages = pages_match.group(1) if pages_match else None

        return BibliographyEntry(
            raw_text=raw,
            key=key,
            style=style,
            authors=authors,
            year=year,
            title=title,
            journal=journal,
            doi=doi,
            url=url,
            pages=pages,
            number=number
        )

    def _extract_authors(self, raw: str) -> list:
        """Extracts authors from a bibliography entry text."""
        apa_authors = re.findall(
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+,\s+[A-Z]\.(?:\s+[A-Z]\.)?)',
            raw, re.UNICODE
        )
        if apa_authors:
            return apa_authors[:6]

        mla_authors = re.findall(
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ\-\']+,\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)',
            raw, re.UNICODE
        )
        if mla_authors:
            return mla_authors[:6]

        return []

    def _generate_entry_key(self, authors: list, year: Optional[str], number: int) -> str:
        """Generates a normalized key for citation→bibliography matching."""
        if authors and year:
            first_author_lastname = authors[0].split(',')[0].strip().lower()
            first_author_lastname = ''.join(
                _ACCENT_MAP.get(c, c) for c in first_author_lastname
            )
            return f"{first_author_lastname}_{year}"
        return f"ref_{number}"

    def _detect_entry_style(self, raw: str, authors: list, year: Optional[str]) -> CitationStyle:
        """Infers the style of a bibliography entry."""
        if re.search(r'[A-Z][a-záéíóú]+,\s+[A-Z]\.\s+\((?:19|20)\d{2}\)', raw):
            return CitationStyle.APA
        if authors and not re.search(r'\((?:19|20)\d{2}\)', raw):
            return CitationStyle.MLA
        if re.match(r'^\[?\d+\]?\.?\s+[A-Z]', raw.strip()):
            return CitationStyle.IEEE
        if year:
            return CitationStyle.APA
        return CitationStyle.UNKNOWN

    def _detect_dominant_style(self, citations: list, bibliography: list) -> CitationStyle:
        """Determines the dominant citation style of the document."""
        if not citations and not bibliography:
            return CitationStyle.UNKNOWN

        style_counts: defaultdict = defaultdict(float)
        for c in citations:
            style_counts[c.style] += 1.0
        for b in bibliography:
            style_counts[b.style] += 0.5

        if not style_counts:
            return CitationStyle.UNKNOWN

        return max(style_counts, key=style_counts.get)

    def _cross_link(self, citations: list, bibliography: list, zones: list) -> tuple:
        """
        Links inline citations to bibliography entries.
        Returns (orphan_citations, uncited_bibliography).

        Complexity: O(n+m) via pre-indexed lookups on the common path;
        falls back to O(m) substring scan only for fuzzy author matches.
        """
        # Pre-index bibliography
        _by_number: dict = {}
        _by_lastname: dict = defaultdict(list)  # normalized_lastname -> [entry]
        for entry in bibliography:
            if entry.number is not None:
                _by_number[entry.number] = entry
            if entry.authors:
                ln = self._normalize_name(entry.authors[0].split(',')[0])
                _by_lastname[ln].append(entry)

        # Pre-index zones by start_pos for O(log m) lookups
        _zone_starts = [z.start_pos for z in zones]

        def _mark_zone(citation, entry):
            idx = bisect.bisect_right(_zone_starts, citation.start_pos) - 1
            if idx >= 0:
                zone = zones[idx]
                if zone.start_pos <= citation.start_pos < zone.end_pos:
                    if entry not in zone.linked_bibliography:
                        zone.linked_bibliography.append(entry)
                    zone.has_valid_citation = True

        def _find_entry(citation):
            # Fast path: numeric (IEEE / Vancouver)
            if citation.number is not None:
                return _by_number.get(citation.number)

            if not citation.author:
                return None

            cite_ln = self._normalize_name(
                citation.author.split(',')[0].split('&')[0].strip()
            )

            # Fast path: exact lastname match
            for entry in _by_lastname.get(cite_ln, []):
                if not citation.year or not entry.year or citation.year == entry.year:
                    return entry

            # Slow path: substring lastname match (rare — only when exact fails)
            for entry in bibliography:
                if not entry.authors:
                    continue
                entry_ln = self._normalize_name(entry.authors[0].split(',')[0])
                if cite_ln in entry_ln or entry_ln in cite_ln:
                    if not citation.year or not entry.year or citation.year == entry.year:
                        return entry

            return None

        linked_bib_keys = set()
        orphan_citations = []

        for citation in citations:
            found = _find_entry(citation)
            matched = found is not None

            if found is not None:
                _mark_zone(citation, found)
                linked_bib_keys.add(found.key)

            # EC-05: IEEE/Vancouver/Chicago without bibliography → inline citation
            # is self-contained (numbered reference system, no entry needed).
            if not matched and citation.style in (
                CitationStyle.IEEE, CitationStyle.CHICAGO, CitationStyle.VANCOUVER
            ):
                idx = bisect.bisect_right(_zone_starts, citation.start_pos) - 1
                if idx >= 0:
                    zone = zones[idx]
                    if zone.start_pos <= citation.start_pos < zone.end_pos:
                        zone.has_valid_citation = True
                matched = True

            if not matched:
                orphan_citations.append(citation)

        uncited_bibliography = [b for b in bibliography if b.key not in linked_bib_keys]
        return orphan_citations, uncited_bibliography

    @staticmethod
    def _normalize_name(name: str) -> str:
        return ''.join(_ACCENT_MAP.get(c, c) for c in name.lower().strip())

    def _score_zones(self, zones: list) -> None:
        """
        Calculates plagiarism risk per zone.

        DIRECT_QUOTE + valid citation  → 0.05 (format error only)
        DIRECT_QUOTE + no citation     → 0.95
        PARAPHRASE + valid citation    → 0.15
        PARAPHRASE + no citation       → 0.80
        ORIGINAL                       → 0.10 base (modulated by Qdrant/SerpAPI)
        BIBLIOGRAPHY                   → 0.00
        """
        for zone in zones:
            if zone.zone_type == ZoneType.BIBLIOGRAPHY:
                zone.plagiarism_risk = 0.0
            elif zone.zone_type == ZoneType.DIRECT_QUOTE:
                zone.plagiarism_risk = 0.05 if zone.has_valid_citation else 0.95
            elif zone.zone_type == ZoneType.PARAPHRASE:
                zone.plagiarism_risk = 0.15 if zone.has_valid_citation else 0.80
            elif zone.zone_type == ZoneType.ORIGINAL:
                zone.plagiarism_risk = 0.05 if zone.has_valid_citation else 0.10
            else:
                zone.plagiarism_risk = 0.10

    def _compute_citation_coverage(self, zones: list) -> float:
        """Percentage of non-original zones that have a valid citation."""
        non_original = [z for z in zones if z.zone_type in (
            ZoneType.DIRECT_QUOTE, ZoneType.PARAPHRASE
        )]
        if not non_original:
            return 1.0
        cited = sum(1 for z in non_original if z.has_valid_citation)
        return cited / len(non_original)

    def _compute_style_consistency(self, citations: list) -> float:
        """Calculates how consistent the citation style is across the document."""
        if not citations:
            return 1.0
        styles = [c.style for c in citations if c.style != CitationStyle.UNKNOWN]
        if not styles:
            return 0.0
        return Counter(styles).most_common(1)[0][1] / len(styles)


# ── Module-level singleton (P-09) ─────────────────────────────────
# Both flask_routes and full_analysis import this factory so the spaCy
# model is loaded exactly once at gunicorn preload, shared via CoW.
_shared_detector: Optional[CitationDetector] = None


def get_citation_detector() -> CitationDetector:
    global _shared_detector
    if _shared_detector is None:
        _shared_detector = CitationDetector()
    return _shared_detector
