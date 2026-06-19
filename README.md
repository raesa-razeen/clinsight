# CLINSIGHT - Clinical Supply Chain Intelligence System

> Automated anomaly detection · Monte Carlo simulation · LLM-powered intelligence briefings  
> Built for clinical trial supply chain operations

---

## The Problem

Clinical trial supply chains are complex, high-stakes, and data-intensive. A single Phase III trial can span 40+ active studies across 15+ countries, with data flowing daily from multiple source systems - CTMS, IRT, and supply platforms.

Today, supply chain teams spend hours every morning manually reviewing dashboards, investigating data anomalies, cross-referencing tables, and writing status updates. By the time a problem is identified and escalated, valuable response time has already been lost.

Three specific problems drive this:

**Data integrity failures go undetected** - when source systems send inconsistent or incomplete data, nobody knows until a downstream report looks wrong. Investigating why requires manual SQL queries across multiple tables, often involving 3-4 people under time pressure.

**Business anomalies are noticed too late** - an enrollment drop, a depot running critically low on drug supply, a site that hasn't had activity in 3 weeks - these are visible in dashboards only if someone is actively looking. By the time they're escalated, the response window has shrunk dramatically.

**Impact is unknown until it's felt** - when something goes wrong, the immediate question is "how bad is this?" Answering that question currently requires experienced analysts to manually model scenarios. That takes time nobody has.

---

## What CLINSIGHT Does

CLINSIGHT is a four-layer clinical supply chain intelligence system that automates the entire investigation and briefing process.

You run it. It tells you what's wrong, how bad it could get, and exactly what to do about it - in plain English, in seconds.

---

## Architecture
Clinical Supply Data (CTMS · IRT · Supply Systems)

|


LAYER 1 — Data Contract Validator

Cross-system relationship checks

Selective quarantine per study

CLEAR / WARNING / QUARANTINED

|


|                   |

QUARANTINED            CLEAR / WARNING

Alert raised           Proceed to Layer 2

|



LAYER 2 — Business Anomaly Detector

Enrollment drop detection

Site inactivity detection

Inventory shortage detection

Drug expiry risk detection

|



LAYER 3 — Monte Carlo Simulation

10,000 scenarios per anomaly

P10 / P50 / P90 impact projections

Stockout days · Patients at risk · Units wasted

|



LAYER 4 — LLM Intelligence Briefing

Plain English situation summary

Quantified impact statement

Three specific recommended actions

Urgency classification: IMMEDIATE / THIS WEEK / MONITOR

|



Streamlit Dashboard + PDF Report Download

---

## Key Design Decisions

**Selective quarantine** - each study is validated independently. One study with data integrity issues never blocks the remaining 39 from being analysed. This mirrors how real clinical operations must work - one bad study cannot halt the entire portfolio.

**Two-stage anomaly detection** - Layer 1 catches data quality failures before business logic runs on corrupt data. Layer 2 catches business anomalies in clean validated data. These are fundamentally different problem types and require different detection approaches.

**Monte Carlo over point estimates** - clinical supply chain planning under uncertainty cannot rely on single-number forecasts. P10/P50/P90 projections give planners a range of scenarios to plan against, not a false sense of precision.

**LLM for explanation not for logic** - the core detection and simulation logic is deterministic Python. The LLM is used only for the final explanation step - converting structured findings into plain English briefings. This keeps the system reliable and auditable.

---

## What CLINSIGHT Detects

**Layer 1 - Data Contract Violations**
- Study missing from configuration table
- Sites in enrollment with no site profile record
- IRT dispensing count mismatching CTMS enrollment count
- No inventory records for an active study
- Enrollment data stale beyond threshold
- Sites missing geocoordinates

**Layer 2 - Business Anomalies**
- Site enrollment below target threshold
- Site inactive beyond maximum days
- Depot inventory critically below expected levels
- Drug supply approaching expiry window

---

## Monte Carlo Results - Example Output
CLN-004-2025 - INVENTORY SHORTAGE - Risk: CRITICAL
P10 Best Case    : 11.2 days of supply remaining

P50 Most Likely  : 12.9 days of supply remaining

P90 Worst Case   : 15.4 days of supply remaining
STOCKOUT IMMINENT. Emergency resupply required.

---

## Intelligence Briefing - Example Output
STUDY: CLN-004-2025  |  URGENCY: IMMEDIATE
HEADLINE: Critical inventory shortage at Singapore depot.
SITUATION: Depot D002 in Singapore has critically low inventory

of 21 units vs 404 expected - 5.2% utilization. The Mumbai depot

also faces expiry risk with 305 units expiring in 48 days.
IMPACT: Monte Carlo projections indicate stockout in 11.2 to 15.4

days. Expiry risk could result in 312 to 795 units wasted.
RECOMMENDED ACTIONS:

Initiate emergency resupply to Singapore depot.
Review allocation from nearest depot with surplus stock.
Accelerate dispensing at Mumbai depot sites.


---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Generation | Python · Pandas · NumPy |
| Layer 1 Validation | Python · Pandas · Dataclasses |
| Layer 2 Detection | Python · Statistical thresholds |
| Layer 3 Simulation | NumPy · Monte Carlo (10,000 runs) |
| Layer 4 Briefing | LLaMA 3.3 70B via Groq API |
| UI | Streamlit |
| PDF Export | FPDF2 |
| Data Standard | CDISC-aligned synthetic datasets |

---

## Installation

```bash
git clone https://github.com/raesa-razeen/clinsight.git
cd clinsight
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install pandas numpy groq streamlit python-dotenv scipy fpdf2
```

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at [console.groq.com](https://console.groq.com)

Generate synthetic data:
```bash
python data_generator.py
```

---

## Usage

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

Click **Run Full CLINSIGHT Analysis** to execute the complete pipeline.

Download the intelligence report as PDF using the button at the bottom.

---

## Data

CLINSIGHT uses synthetic clinical trial data generated by `data_generator.py` following CDISC-aligned schemas. The synthetic data mirrors the structure of real clinical supply chain platforms - study configuration, site profiles, enrollment records, inventory levels, and IRT dispensing records.

Five anomalies are planted in the synthetic data to demonstrate detection capability:

- Enrollment drop at APAC sites
- Site inactivity beyond 30 days
- Critical inventory shortage at a regional depot
- Drug expiry risk within 60-day window
- IRT dispensing mismatch against CTMS enrollment

---

## Roadmap

**v0.1 - Current**
- Four-layer pipeline: data validation, anomaly detection, Monte Carlo simulation, LLM briefings
- Streamlit dashboard with PDF export
- Selective study quarantine

**v0.2 - In Development**
- Resiliency audit module - vulnerability scoring per study
- Protocol amendment impact analyzer
- Predictive stockout calendar

**v1.0 - Planned**
- Integration layer for live Gold layer data
- Multi-study portfolio dashboard
- Configurable alert thresholds per study type
- API endpoint for external system integration

---

## Author

**Raesa Razeen**  
PhD Research Scholar · Presidency University, Bangalore  
AI Research Engineer · Clinical Supply Chain AI · LLM Systems

[LinkedIn](https://www.linkedin.com/in/raesa-razeen-200260209/) · [GitHub](https://github.com/raesa-razeen) · [HALO Project](https://github.com/raesa-razeen/halo-hallucination-detector)

---

## License

MIT License — open for research and commercial use.
