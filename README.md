<div align="center">

# <img src="assets/beegent_logo.svg" alt="" height="38" valign="middle" /> Beegent

### The open-source multi-agent system for geospatial data source discovery.

</div>

Beegent is an open-source **multi-agent discovery system** that automates the
technical-scoping bottleneck in geospatial data ingestion: a planner, catalog
workers, a search & explore worker, and a triage/critic loop work together to
find, evaluate, and hand off candidate data sources — turning a day of manual
searching and tab-switching into a review-ready draft in minutes.

Accelerates the **technical scoping** step of your ingestion process:

```
business scoping -> [technical scoping: THIS TOOL] -> dev pipeline -> validation -> deployment
```

Today that step is manual: search the internet for a country/use-case's data,
open a handful of endpoints, compare them, explore columns, define a
source-to-target mapping, and write up metadata — before the dev pipeline
work can even start. This is a working prototype that runs that step as five
chained agents and outputs a single review-ready handoff package instead of
a day of tab-switching.
