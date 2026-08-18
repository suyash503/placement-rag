"""Pull structured constraints out of a natural-language question.

An embedding of "companies paying more than 10 LPA" sits close to an embedding of
"companies paying more than 5 LPA" — the vector carries the topic, not the
threshold. Numeric and categorical constraints therefore have to be lifted out of
the text and pushed down to the database as a filter; no amount of retrieval
tuning fixes it otherwise.

Regex handles the common phrasings cheaply and deterministically. Anything it
misses can fall back to a single structured LLM call.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.logging import get_logger

log = get_logger("rag.router")

BRANCH_ALIASES = {
    "Computer Engineering": ["cs", "cse", "computer", "comps", "computer science", "computer engineering"],
    "Information Technology": ["it", "info tech", "information technology"],
    "Electronics & Telecommunication": ["entc", "extc", "telecom", "telecommunication", "e&tc"],
    "Electronics": ["electronics", "etrx"],
    "Mechanical": ["mech", "mechanical"],
    "Civil": ["civil"],
    "Electrical": ["electrical", "eee"],
    "Production": ["production"],
    "Instrumentation": ["instrumentation", "instru"],
    "Chemical": ["chemical", "chem"],
}

AGGREGATE_HINTS = (
    "how many", "which years", "what years", "all the years", "list all",
    "profile of", "tell me about", "overview", "summary", "how often",
    "every year", "across years", "total number",
)

_NUM = r"(\d+(?:\.\d+)?)"
_ABOVE = re.compile(rf"(?:above|over|more than|greater than|higher than|at least|minimum|>=?)\s*{_NUM}\s*(?:lpa|lakhs?|l\b)", re.I)
_BELOW = re.compile(rf"(?:below|under|less than|lower than|at most|upto|up to|<=?)\s*{_NUM}\s*(?:lpa|lakhs?|l\b)", re.I)
_BETWEEN = re.compile(rf"between\s*{_NUM}\s*(?:lpa|lakhs?)?\s*(?:and|to|-)\s*{_NUM}\s*(?:lpa|lakhs?)", re.I)
_PLUS = re.compile(rf"{_NUM}\s*\+\s*(?:lpa|lakhs?)", re.I)
_YEAR = re.compile(r"\b(20(?:1[4-8]))\b")
_YEAR_RANGE = re.compile(r"\b(20(?:1[4-8]))\s*(?:to|-|and|until|till)\s*(20(?:1[4-8]))\b")
_CGPA_HAVE = re.compile(rf"(?:i\s+have|my|with|having|got|scored)\s*(?:a\s*)?{_NUM}\s*(?:cgpa|gpa)", re.I)
_CGPA_HAVE_ALT = re.compile(rf"(?:cgpa|gpa)\s*(?:of|is|=)?\s*{_NUM}", re.I)
_NO_BACKLOG = re.compile(r"\b(no|zero|without)\s+backlogs?\b", re.I)
_WITH_BACKLOG = re.compile(r"\b(with|have|having|active)\s+backlogs?\b", re.I)


@dataclass
class ParsedQuery:
    original: str
    semantic_query: str
    filters: dict[str, Any] = field(default_factory=dict)
    prefer_doc_type: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_pre_filter(self) -> dict[str, Any] | None:
        return self.filters or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_query": self.semantic_query,
            "filters": self.filters,
            "prefer_doc_type": self.prefer_doc_type,
            "notes": self.notes,
        }


def _merge_range(filters: dict, key: str, op: str, value: float) -> None:
    filters.setdefault(key, {})
    filters[key][op] = value


def parse_query(question: str, known_companies: set[str] | None = None) -> ParsedQuery:
    text = question.strip()
    low = text.lower()
    filters: dict[str, Any] = {}
    notes: list[str] = []

    if m := _BETWEEN.search(low):
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        _merge_range(filters, "package_lpa", "$gte", lo)
        _merge_range(filters, "package_lpa", "$lte", hi)
        notes.append(f"package between {lo} and {hi} LPA")
    else:
        if m := (_ABOVE.search(low) or _PLUS.search(low)):
            value = float(m.group(1))
            _merge_range(filters, "package_lpa", "$gte", value)
            notes.append(f"package >= {value} LPA")
        if m := _BELOW.search(low):
            value = float(m.group(1))
            _merge_range(filters, "package_lpa", "$lte", value)
            notes.append(f"package <= {value} LPA")

    if m := _YEAR_RANGE.search(low):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        _merge_range(filters, "year", "$gte", lo)
        _merge_range(filters, "year", "$lte", hi)
        notes.append(f"year {lo}-{hi}")
    else:
        years = sorted({int(y) for y in _YEAR.findall(low)})
        if len(years) == 1:
            filters["year"] = {"$eq": years[0]}
            notes.append(f"year = {years[0]}")
        elif years:
            filters["year"] = {"$in": years}
            notes.append(f"year in {years}")

    branches = [
        canonical
        for canonical, aliases in BRANCH_ALIASES.items()
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases)
    ]
    if branches:
        filters["branches"] = {"$in": branches}
        notes.append(f"branch in {branches}")

    # "I have 7.2 CGPA" is an eligibility question: keep companies whose cutoff the
    # student clears, not companies whose cutoff happens to equal 7.2.
    cgpa_match = _CGPA_HAVE.search(low) or _CGPA_HAVE_ALT.search(low)
    if cgpa_match:
        value = float(cgpa_match.group(1))
        if 0 < value <= 10:
            filters["cgpa_cutoff"] = {"$lte": value}
            notes.append(f"cutoff <= {value} (student's CGPA)")

    if _NO_BACKLOG.search(low):
        notes.append("student has no backlogs (no filter needed)")
    elif _WITH_BACKLOG.search(low):
        filters["active_backlogs"] = {"$eq": "Allowed"}
        notes.append("company allows active backlogs")

    # Both granularities are indexed, and they compete on every query. "How many
    # drives did TCS run" needs the profile; "which companies paid over 20 LPA"
    # needs the individual drives, and profiles would crowd them out.
    if any(h in low for h in AGGREGATE_HINTS):
        prefer_doc_type = "company"
        notes.append("aggregate question -> prefer company profile documents")
    else:
        prefer_doc_type = "record"
        notes.append("specific question -> prefer individual drive records")

    if known_companies:
        hit = next((c for c in known_companies if len(c) > 3 and c in low), None)
        if hit:
            notes.append(f"mentions known company '{hit}'")

    return ParsedQuery(
        original=text,
        semantic_query=text,
        filters=filters,
        prefer_doc_type=prefer_doc_type,
        notes=notes,
    )


LLM_EXTRACTION_PROMPT = """Extract structured filters from a student's placement question.

Return ONLY a JSON object with any of these optional keys:
  package_lpa   - object with $gte and/or $lte (annual package in lakhs)
  year          - object with $eq, $in, $gte or $lte (valid years: 2014-2018)
  branches      - object with $in, values from: {branches}
  cgpa_cutoff   - object with $lte, set to the student's own CGPA when they state it

Omit any key the question does not constrain. Return {{}} if nothing is constrained.

Question: {question}
JSON:"""


def llm_extract_filters(question: str, llm) -> dict[str, Any]:
    """Fallback extractor for phrasings the regexes do not cover."""
    prompt = LLM_EXTRACTION_PROMPT.format(
        branches=", ".join(BRANCH_ALIASES), question=question
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        log.warning("llm filter extraction failed: %s", exc)
        return {}
