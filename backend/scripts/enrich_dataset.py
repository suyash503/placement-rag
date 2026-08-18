"""Build data/placement_enriched.csv from the raw placement export.

The raw export only records which company recruited at which college in which year.
Everything an actual student asks about — package, CGPA cutoff, eligible branches,
selection rounds — is missing, so this script attaches a synthetic but internally
consistent eligibility layer on top of the real rows.

Values are derived deterministically from a hash of (company, college, year), so
re-running always produces the same file and the ingestion stays reproducible.
"""

import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.logging import get_logger  # noqa: E402

log = get_logger("enrich")

JUNK_COMPANIES = {"", "nan", "name of company", "company", "-", "na", "n/a"}

TIER_KEYWORDS = {
    "tier1": [
        "google", "microsoft", "amazon", "goldman", "morgan stanley", "j p morgan",
        "jp morgan", "jpmorgan", "barclays", "bnp paribas", "hsbc", "nomura",
        "credit suisse", "deutsche", "rakuten", "media.net", "directi", "sprinklr",
        "arcesium", "de shaw", "tower research", "optiver", "uber", "flipkart",
        "adobe", "oracle", "vmware", "nvidia", "qualcomm", "micron", "intuit",
        "paypal", "visa inc", "mastercard", "nutanix", "salesforce", "linkedin",
        "dolat", "citi", "ubs", "nomura", "millennium", "graviton",
    ],
    "tier2": [
        "zs associates", "zycus", "persistent", "amdocs", "seclore", "quantiphi",
        "musigma", "mu sigma", "indus valley", "visible alpha", "gep", "nucsoft",
        "protegrity", "nse it", "bitwise", "vistaar", "kotak", "icici", "axis bank",
        "reliance jio", "ericsson", "bosch", "siemens", "philips", "honeywell",
        "schneider", "abb", "john deere", "cummins", "mercedes", "tata motors",
        "mahindra", "sap", "cisco", "nvidia", "pharmeasy", "pharmaeasy", "zomato",
        "swiggy", "ola", "paytm", "razorpay", "browserstack", "postman",
    ],
    "mass": [
        "tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini",
        "tech mahindra", "hcl", "lti", "l&t infotech", "syntel", "atos", "ibm",
        "mphasis", "mindtree", "hexaware", "zensar", "birlasoft", "ntt data",
        "dxc", "virtusa", "sopra", "igate", "polaris", "quinnox", "cybage",
        "datamatics", "3i infotech", "kpit", "harbinger",
    ],
    "core": [
        "l&t", "larsen", "tata projects", "snc lavalin", "deugro", "thermax",
        "godrej", "kirloskar", "forbes marshall", "alfa laval", "sandvik",
        "endurance", "acg", "sanmar", "ril", "reliance", "jsw", "essar", "vedanta",
        "afcons", "shapoorji", "gammon", "voltas", "blue star", "crompton",
    ],
    "govt": ["indian army", "indian navy", "indian air force", "barc", "isro", "drdo", "ongc"],
}

PACKAGE_RANGE = {
    "tier1": (12.0, 44.0),
    "tier2": (6.0, 14.0),
    "mass": (3.2, 4.6),
    "core": (3.5, 8.0),
    "govt": (6.5, 9.0),
    "mid": (3.5, 7.0),
}

IT_BRANCHES = ["Computer Engineering", "Information Technology", "Electronics & Telecommunication"]
CORE_BRANCHES = ["Mechanical", "Civil", "Electrical", "Production", "Instrumentation"]
ALL_BRANCHES = IT_BRANCHES + CORE_BRANCHES + ["Electronics", "Chemical"]

ROLES = {
    "tier1": [
        "Software Development Engineer", "Software Engineer", "Quantitative Analyst",
        "Technology Analyst", "Product Engineer", "Data Scientist",
    ],
    "tier2": [
        "Software Engineer", "Associate Consultant", "Business Analyst",
        "Systems Engineer", "Data Analyst", "Graduate Engineer Trainee",
    ],
    "mass": ["Assistant System Engineer", "Systems Engineer", "Programmer Analyst", "Associate Software Engineer"],
    "core": ["Graduate Engineer Trainee", "Design Engineer", "Site Engineer", "Project Engineer"],
    "govt": ["Technical Entry Scheme Officer", "Graduate Engineer"],
    "mid": ["Software Engineer", "Junior Engineer", "Associate Engineer", "Trainee Engineer"],
}

ROUND_SETS = {
    "tier1": [
        ["Online Coding Test", "Technical Interview 1", "Technical Interview 2", "Hiring Manager Round", "HR Interview"],
        ["Online Assessment", "System Design Round", "Technical Interview", "HR Interview"],
    ],
    "tier2": [
        ["Online Aptitude Test", "Technical Interview", "HR Interview"],
        ["Written Test", "Group Discussion", "Technical Interview", "HR Interview"],
    ],
    "mass": [
        ["Online Aptitude Test", "Technical Interview", "HR Interview"],
        ["Online Test", "Interview"],
    ],
    "core": [
        ["Written Technical Test", "Technical Interview", "HR Interview"],
        ["Aptitude Test", "Group Discussion", "Personal Interview"],
    ],
    "govt": [["Written Examination", "SSB Interview", "Medical Examination"]],
    "mid": [["Aptitude Test", "Technical Interview", "HR Interview"], ["Written Test", "Interview"]],
}

HIRING_TYPES = ["Full-Time", "Full-Time", "Full-Time", "Internship + PPO", "6-Month Internship"]

SOFTWARE_HINTS = (
    "tech", "soft", "system", "solution", "infotech", "data", "lab", "digital",
    "consult", "analytic", "info", "web", "media", "cyber", "cloud", "computing",
    "learn", "edu", "app", "mobil", "network", "logic", "code", "byte", "ware",
)
CORE_HINTS = (
    "engineering", "construction", "project", "industr", "steel", "cement",
    "motor", "auto", "power", "infra", "chemical", "pharma", "energy", "build",
)


LOCATION_POOL = {
    "Mumbai": ["Mumbai", "Navi Mumbai", "Thane", "Pune"],
    "Navi Mumbai": ["Navi Mumbai", "Mumbai", "Pune"],
    "Thane": ["Thane", "Mumbai", "Navi Mumbai"],
    "Pune": ["Pune", "Mumbai", "Bengaluru", "Hyderabad"],
}


def canonical_company(raw: str) -> str:
    """Collapse spacing/punctuation variants so 'L & T Infotech' == 'L&T Infotech'."""
    name = re.sub(r"\s+", " ", raw).strip().strip(".,-")
    name = re.sub(r"\s*&\s*", "&", name)
    name = re.sub(r"\s*\(\s*20\d{2}\s*\)\s*$", "", name).strip()
    return name


def resolve_surface_forms(names: list[str]) -> dict[str, str]:
    """Map every casing variant of a name onto its most common spelling.

    'TCS', 'tcs' and 'Tcs' are one recruiter, but they index as three companies and
    each query then sees a third of the records.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for name in names:
        counts[name.casefold()][name] += 1
    return {
        key: variants.most_common(1)[0][0]
        for key, variants in counts.items()
    }


def company_tier(name: str) -> str:
    low = name.lower()
    for tier, keywords in TIER_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return tier
    return "mid"


def seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_record(company: str, college: str, region: str, year: int) -> dict:
    tier = company_tier(company)
    rng = seeded_rng(company, college, str(year))

    low, high = PACKAGE_RANGE[tier]
    # Packages skew low within a band, so bias the draw instead of sampling uniformly.
    base = low + (high - low) * (rng.random() ** 1.7)
    drift = 1 + 0.055 * (year - 2018)
    package = round(max(2.5, base * drift), 2)

    if tier == "mass":
        cgpa = round(rng.choice([6.0, 6.0, 6.5, 6.5, 7.0]), 1)
    else:
        cgpa = round(min(8.5, 5.8 + package * 0.075 + rng.uniform(-0.35, 0.35)) * 2) / 2

    if tier == "core":
        pool = CORE_BRANCHES
    elif tier == "govt":
        pool = ALL_BRANCHES
    elif tier in ("tier1", "tier2", "mass"):
        pool = IT_BRANCHES
    else:
        name_low = company.lower()
        if any(h in name_low for h in SOFTWARE_HINTS):
            pool = IT_BRANCHES
        elif any(h in name_low for h in CORE_HINTS):
            pool = CORE_BRANCHES
        else:
            pool = IT_BRANCHES if rng.random() < 0.7 else CORE_BRANCHES

    count = min(len(pool), rng.randint(2, 4))
    branches = sorted(rng.sample(pool, count))

    tenth = rng.choice([60, 60, 65, 70, 70, 75])
    twelfth = rng.choice([60, 60, 65, 70, 70, 75])
    if tier == "tier1":
        tenth, twelfth = max(tenth, 70), max(twelfth, 70)

    backlogs = "Not allowed" if tier in ("tier1", "mass") or rng.random() < 0.6 else "Allowed"

    return {
        "company": company,
        "college": college,
        "region": region,
        "year": year,
        "role": rng.choice(ROLES[tier]),
        "package_lpa": package,
        "cgpa_cutoff": cgpa,
        "allowed_branches": "; ".join(branches),
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "active_backlogs": backlogs,
        "selection_rounds": " -> ".join(rng.choice(ROUND_SETS[tier])),
        "job_location": rng.choice(LOCATION_POOL.get(region, ["Mumbai", "Pune"])),
        "hiring_type": rng.choice(HIRING_TYPES),
        "company_tier": tier,
    }


def main() -> int:
    settings = get_settings()
    src, dst = settings.raw_csv, settings.enriched_csv

    if not src.exists():
        log.error("raw dataset not found at %s", src)
        return 1

    with src.open(encoding="utf-8-sig", newline="") as fh:
        raw_rows = list(csv.DictReader(fh))

    cleaned = []
    skipped = 0
    for row in raw_rows:
        company = canonical_company(row.get("name of company") or "")
        college = re.sub(r"\s+", " ", (row.get("college name") or "")).strip()
        region = (row.get("region") or "").strip()
        year_raw = (row.get("year") or "").strip()

        if company.casefold() in JUNK_COMPANIES or not college or not year_raw.isdigit():
            skipped += 1
            continue
        cleaned.append((company, college, region, int(year_raw)))

    company_names = resolve_surface_forms([c for c, _, _, _ in cleaned])
    college_names = resolve_surface_forms([c for _, c, _, _ in cleaned])

    records, seen = [], set()
    for company, college, region, year in cleaned:
        company = company_names[company.casefold()]
        college = college_names[college.casefold()]

        key = (company, college, year)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)

        records.append(build_record(company, college, region, year))

    records.sort(key=lambda r: (r["company"].lower(), r["year"], r["college"]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    companies = {r["company"] for r in records}
    packages = sorted(r["package_lpa"] for r in records)
    log.info("wrote %s", dst)
    log.info("  rows        : %d (skipped %d junk/duplicate)", len(records), skipped)
    log.info("  companies   : %d", len(companies))
    log.info("  colleges    : %d", len({r["college"] for r in records}))
    log.info("  years       : %s", sorted({r["year"] for r in records}))
    log.info(
        "  package LPA : min %.2f | median %.2f | max %.2f",
        packages[0], packages[len(packages) // 2], packages[-1],
    )
    log.info("  >10 LPA     : %d rows", sum(1 for p in packages if p > 10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
