"""Catalog workers: deterministic (no LLM) calls to known open geospatial catalogs.

Two catalogs to start, both auth-free. TODO: this is the seed of a proper
catalog table; the periodic health-check/scout job that keeps it fresh is
explicitly out of scope for this POC.
"""

import requests

import config
from schemas import Candidate

# TODO: replace with pycountry. Hardcoded is fine for a POC.
COUNTRY_ISO3 = {
    "kenya": "KEN",
    "nigeria": "NGA",
    "ethiopia": "ETH",
    "uganda": "UGA",
    "tanzania": "TZA",
    "ghana": "GHA",
    "rwanda": "RWA",
    "somalia": "SOM",
    "south sudan": "SSD",
    "mozambique": "MOZ",
    "bangladesh": "BGD",
    "india": "IND",
}

# geoBoundaries only ever serves administrative boundaries, so calling it for an
# unrelated use case would inject off-topic candidates for free. Cheap keyword
# gate keeps the catalog honest without an LLM call.
_BOUNDARY_WORDS = (
    "boundar",
    "admin",
    "adm0",
    "adm1",
    "adm2",
    "district",
    "county",
    "province",
    "region",
    "subnational",
    "shapefile",
)


def _hdx(country: str, use_case: str) -> list[Candidate]:
    resp = requests.get(
        "https://data.humdata.org/api/3/action/package_search",
        params={"q": f"{country} {use_case}", "rows": 5},
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for pkg in resp.json()["result"]["results"]:
        out.append(
            Candidate(
                url=f"https://data.humdata.org/dataset/{pkg['name']}",
                title=pkg.get("title") or pkg["name"],
                source="catalog:hdx",
                hops=0,
                rationale=(pkg.get("notes") or "HDX dataset").strip().replace("\n", " ")[:200],
            )
        )
    return out


def _geoboundaries(country: str, use_case: str) -> list[Candidate]:
    iso3 = COUNTRY_ISO3.get(country.strip().lower())
    if not iso3:
        print(f"  [catalog:geoboundaries] no ISO3 mapping for {country!r}, skipping")
        return []
    if not any(w in use_case.lower() for w in _BOUNDARY_WORDS):
        print("  [catalog:geoboundaries] use case is not boundary-related, skipping")
        return []

    resp = requests.get(
        f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ALL/",
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    levels = resp.json()
    if isinstance(levels, dict):  # single-level responses come back unwrapped
        levels = [levels]

    # ADM1/ADM2 are the useful ones for most dashboards; keep it to two so one
    # catalog can't dominate the candidate list.
    levels.sort(key=lambda lv: lv.get("boundaryType", "") not in ("ADM1", "ADM2"))
    out = []
    for lv in levels[:2]:
        url = lv.get("staticDownloadLink") or lv.get("gjDownloadURL")
        if not url:
            continue
        out.append(
            Candidate(
                url=url,
                title=f"geoBoundaries {lv.get('boundaryISO')} {lv.get('boundaryType')} - "
                f"{lv.get('boundaryName')}",
                source="catalog:geoboundaries",
                hops=0,
                rationale=(
                    f"Open admin boundaries ({lv.get('boundaryType')}, "
                    f"{lv.get('admUnitCount')} units, license: {lv.get('boundaryLicense')}), "
                    f"source: {lv.get('boundarySource')}"
                ),
            )
        )
    return out


WORKERS = {"catalog:hdx": _hdx, "catalog:geoboundaries": _geoboundaries}


def run_catalog_workers(country: str, use_case: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for name, worker in WORKERS.items():
        try:
            found = worker(country, use_case)
        except Exception as exc:  # one dead catalog must not kill the run
            print(f"  [{name}] failed: {exc}")
            continue
        print(f"  [{name}] {len(found)} hits")
        candidates.extend(found)
    return candidates
