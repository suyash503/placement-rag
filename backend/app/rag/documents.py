"""Turn enriched CSV rows into the documents that actually get embedded.

Two document types are produced:

* ``record``  — one per placement drive, written as a sentence rather than a comma
  separated row. Sentence-transformers models were trained on prose, and a raw
  ``TCS,Pillai,Pune,2016`` line embeds far more weakly than the same facts spelled out.
* ``company`` — one per distinct company, summarising every drive it ran. Row-level
  documents can only answer row-level questions; "which years did Infosys visit"
  needs a document that has seen all the rows at once.

Every document carries a deterministic id so re-ingesting updates in place instead
of appending duplicates.
"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

# Atlas rejects a query that filters on a path missing from this list, so anything
# query_router.py can emit has to be declared here and the index rebuilt.
FILTERABLE_FIELDS = [
    "company",
    "company_lower",
    "college",
    "region",
    "year",
    "package_lpa",
    "cgpa_cutoff",
    "branches",
    "company_tier",
    "hiring_type",
    "active_backlogs",
    "role",
    "job_location",
    "doc_type",
]


def _doc_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:24]


def _record_text(row: dict) -> str:
    branches = row["allowed_branches"].replace(";", ",")
    return (
        f"{row['company']} recruited at {row['college']} ({row['region']}) in {row['year']} "
        f"for the role of {row['role']}. "
        f"Package offered: {row['package_lpa']} LPA. "
        f"Eligibility: minimum CGPA {row['cgpa_cutoff']}, "
        f"open to {branches} branches, "
        f"minimum {row['tenth_percentage']}% in 10th and {row['twelfth_percentage']}% in 12th. "
        f"Active backlogs: {row['active_backlogs']}. "
        f"Selection process: {row['selection_rounds']}. "
        f"Job location: {row['job_location']}. Hiring type: {row['hiring_type']}."
    )


def _company_text(company: str, rows: list[dict]) -> str:
    years = sorted({int(r["year"]) for r in rows})
    colleges = sorted({r["college"] for r in rows})
    packages = sorted(float(r["package_lpa"]) for r in rows)
    roles = sorted({r["role"] for r in rows})
    branches = sorted({b.strip() for r in rows for b in r["allowed_branches"].split(";")})
    cutoffs = sorted(float(r["cgpa_cutoff"]) for r in rows)

    college_list = ", ".join(colleges[:6]) + (f" and {len(colleges) - 6} more" if len(colleges) > 6 else "")
    package_span = (
        f"{packages[0]} LPA"
        if packages[0] == packages[-1]
        else f"{packages[0]} LPA to {packages[-1]} LPA"
    )

    return (
        f"Company profile: {company}. "
        f"{company} conducted {len(rows)} campus drive(s) across {len(colleges)} college(s) "
        f"in the years {', '.join(str(y) for y in years)}. "
        f"Colleges visited: {college_list}. "
        f"Packages offered ranged from {package_span} (average "
        f"{sum(packages) / len(packages):.2f} LPA). "
        f"Roles hired for: {', '.join(roles)}. "
        f"Branches eligible across drives: {', '.join(branches)}. "
        f"CGPA cutoff ranged from {cutoffs[0]} to {cutoffs[-1]}."
    )


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_documents(rows: list[dict]) -> list[Document]:
    documents: list[Document] = []
    by_company: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        by_company[row["company"]].append(row)

        documents.append(
            Document(
                page_content=_record_text(row),
                metadata={
                    "doc_id": _doc_id("record", row["company"], row["college"], row["year"]),
                    "doc_type": "record",
                    "company": row["company"],
                    "company_lower": row["company"].lower(),
                    "college": row["college"],
                    "region": row["region"],
                    "year": int(row["year"]),
                    "role": row["role"],
                    "package_lpa": float(row["package_lpa"]),
                    "cgpa_cutoff": float(row["cgpa_cutoff"]),
                    "branches": [b.strip() for b in row["allowed_branches"].split(";")],
                    "tenth_percentage": int(row["tenth_percentage"]),
                    "twelfth_percentage": int(row["twelfth_percentage"]),
                    "active_backlogs": row["active_backlogs"],
                    "selection_rounds": [r.strip() for r in row["selection_rounds"].split("->")],
                    "job_location": row["job_location"],
                    "hiring_type": row["hiring_type"],
                    "company_tier": row["company_tier"],
                },
            )
        )

    for company, company_rows in by_company.items():
        packages = [float(r["package_lpa"]) for r in company_rows]
        documents.append(
            Document(
                page_content=_company_text(company, company_rows),
                metadata={
                    "doc_id": _doc_id("company", company),
                    "doc_type": "company",
                    "company": company,
                    "company_lower": company.lower(),
                    "college": sorted({r["college"] for r in company_rows}),
                    "region": sorted({r["region"] for r in company_rows}),
                    "year": sorted({int(r["year"]) for r in company_rows}),
                    "package_lpa": round(max(packages), 2),
                    "cgpa_cutoff": min(float(r["cgpa_cutoff"]) for r in company_rows),
                    "branches": sorted(
                        {b.strip() for r in company_rows for b in r["allowed_branches"].split(";")}
                    ),
                    "drive_count": len(company_rows),
                    "company_tier": company_rows[0]["company_tier"],
                },
            )
        )

    return documents
