"""
benchmarks.gdpr_queries
~~~~~~~~~~~~~~~~~~~~~~~~
Twenty hand-labeled retrieval queries over the GDPR (EU 2016/679).

Ground-truth prefixes use the node_path produced by GDPRChunker:
    gdpr.ch{N}.art{M}

A chunk is relevant if its node_path equals the prefix OR starts with
the prefix followed by a dot (covers paragraph and point sub-chunks).
This matches the same is_relevant() contract used in benchmarks/queries.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LegalQuery:
    text: str
    ground_truth_prefixes: list[str]
    notes: str = ""


GDPR_QUERIES: list[LegalQuery] = [
    LegalQuery(
        text="what rights do individuals have to access their personal data",
        ground_truth_prefixes=["gdpr.ch3.art15"],
        notes="Article 15 — Right of access by the data subject",
    ),
    LegalQuery(
        text="how can someone request deletion of their personal data",
        ground_truth_prefixes=["gdpr.ch3.art17"],
        notes="Article 17 — Right to erasure (right to be forgotten)",
    ),
    LegalQuery(
        text="what are the lawful bases for processing personal data",
        ground_truth_prefixes=["gdpr.ch2.art6"],
        notes="Article 6 — Lawfulness of processing",
    ),
    LegalQuery(
        text="what are the conditions for valid consent to data processing",
        ground_truth_prefixes=["gdpr.ch2.art7"],
        notes="Article 7 — Conditions for consent",
    ),
    LegalQuery(
        text="what special categories of data require extra protection",
        ground_truth_prefixes=["gdpr.ch2.art9"],
        notes="Article 9 — Processing of special categories (health, race, religion…)",
    ),
    LegalQuery(
        text="what information must a data controller provide when collecting data",
        ground_truth_prefixes=["gdpr.ch3.art13", "gdpr.ch3.art14"],
        notes="Articles 13 & 14 — Information to be provided to data subjects",
    ),
    LegalQuery(
        text="what are the core principles for processing personal data",
        ground_truth_prefixes=["gdpr.ch2.art5"],
        notes="Article 5 — Principles relating to processing",
    ),
    LegalQuery(
        text="when must a data breach be reported to authorities",
        ground_truth_prefixes=["gdpr.ch4.art33"],
        notes="Article 33 — Notification of breach to supervisory authority",
    ),
    LegalQuery(
        text="when must individuals be notified about a data breach",
        ground_truth_prefixes=["gdpr.ch4.art34"],
        notes="Article 34 — Communication of breach to data subject",
    ),
    LegalQuery(
        text="what are the obligations of data controllers",
        ground_truth_prefixes=["gdpr.ch4.art24"],
        notes="Article 24 — Responsibility of the controller",
    ),
    LegalQuery(
        text="what must a data processing agreement include",
        ground_truth_prefixes=["gdpr.ch4.art28"],
        notes="Article 28 — Processor obligations and contracts",
    ),
    LegalQuery(
        text="when is a data protection impact assessment required",
        ground_truth_prefixes=["gdpr.ch4.art35"],
        notes="Article 35 — DPIA requirement for high-risk processing",
    ),
    LegalQuery(
        text="what are the rules for appointing a data protection officer",
        ground_truth_prefixes=["gdpr.ch4.art37"],
        notes="Article 37 — Designation of DPO",
    ),
    LegalQuery(
        text="what are the tasks and duties of a data protection officer",
        ground_truth_prefixes=["gdpr.ch4.art39"],
        notes="Article 39 — Tasks of the DPO",
    ),
    LegalQuery(
        text="what are the rules for transferring data outside the European Union",
        ground_truth_prefixes=["gdpr.ch5.art44", "gdpr.ch5.art46", "gdpr.ch5.art49"],
        notes="Articles 44, 46, 49 — Third country transfer requirements",
    ),
    LegalQuery(
        text="what fines can be imposed for GDPR violations",
        ground_truth_prefixes=["gdpr.ch8.art83"],
        notes="Article 83 — Administrative fines up to €20M or 4% global turnover",
    ),
    LegalQuery(
        text="what is the right to data portability",
        ground_truth_prefixes=["gdpr.ch3.art20"],
        notes="Article 20 — Right to data portability",
    ),
    LegalQuery(
        text="how is personal data defined under GDPR",
        ground_truth_prefixes=["gdpr.ch1.art4"],
        notes="Article 4 — Definitions (personal data, processing, controller, etc.)",
    ),
    LegalQuery(
        text="what is the right to restrict processing of personal data",
        ground_truth_prefixes=["gdpr.ch3.art18"],
        notes="Article 18 — Right to restriction of processing",
    ),
    LegalQuery(
        text="what are the rights of individuals regarding automated decision making",
        ground_truth_prefixes=["gdpr.ch3.art22"],
        notes="Article 22 — Automated individual decision-making and profiling",
    ),
]


def is_relevant(node_path: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if node_path == p or node_path.startswith(p + "."):
            return True
    return False
