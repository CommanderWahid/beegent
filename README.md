<div align="center">

# <img src="assets/beegent_logo.svg" alt="" height="72" valign="middle" /> Beegent

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: proof of concept](https://img.shields.io/badge/status-proof--of--concept-yellow)

**Open-source multi-agent pipeline that turns a `(country, use_case)` pair into a review-ready list of candidate geospatial data sources.**

</div>

A planner, catalog workers, a search & explore worker, and a triage/critic loop run as five
chained agents to find, evaluate, and hand off candidates — turning a day of manual searching
and tab-switching into a review-ready draft in minutes.

Accelerates the **technical scoping** step of your ingestion process:

```
business scoping -> [technical scoping: THIS TOOL] -> dev pipeline -> validation -> deployment
```

Today that step is manual: search the internet for a country/use-case's data, open a handful
of endpoints, compare them, explore columns, define a source-to-target mapping, and write up
metadata — before the dev pipeline work can even start. This is a working prototype of that
gap-closing step, not a polished system.
