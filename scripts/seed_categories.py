"""Seed standard document categories for DoD IT / MHS environment with NARA GRS mapping."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.metadata import MetadataStore

CATEGORIES = [
    # IT Operations & Support
    {"name": "it_sops", "description": "Standard Operating Procedures for IT systems and services", "grs": "3.1",
     "acl_groups": ["it_support", "devops"], "keywords": ["server", "restart", "deploy", "backup", "restore", "procedure", "sop"]},

    {"name": "it_policies", "description": "IT security policies, ATO packages, STIG compliance, RMF documentation", "grs": "3.2",
     "acl_groups": ["it_support", "cybersecurity", "executives"], "keywords": ["stig", "ato", "cybersecurity", "compliance", "nist", "rmf", "security", "policy"]},

    {"name": "it_runbooks", "description": "Troubleshooting guides, incident response procedures, known error database", "grs": "5.8",
     "acl_groups": ["it_support", "devops"], "keywords": ["outage", "troubleshoot", "incident", "fix", "workaround", "error", "runbook"]},

    {"name": "network_docs", "description": "Network diagrams, configurations, topology, firewall rules", "grs": "6.3",
     "acl_groups": ["it_support", "devops", "cybersecurity"], "keywords": ["network", "vlan", "firewall", "switch", "router", "dns", "topology", "diagram"]},

    {"name": "system_architecture", "description": "System design documents, data flows, integration specs, API documentation", "grs": "6.3",
     "acl_groups": ["it_support", "devops", "engineering"], "keywords": ["architecture", "diagram", "integration", "interface", "api", "design", "dataflow"]},

    {"name": "change_management", "description": "Change requests, RFC records, approval documentation, implementation plans", "grs": "3.1",
     "acl_groups": ["it_support", "devops", "executives"], "keywords": ["change", "rfc", "approval", "implementation", "rollback", "cab", "change request"]},

    # MHS / Medical System
    {"name": "clinical_systems", "description": "MHS Genesis, CHCS, AHLTA, electronic health record system documentation", "grs": "6.3",
     "acl_groups": ["it_support", "clinical", "medical"], "keywords": ["genesis", "chcs", "ahlta", "ehr", "electronic health record", "mhs", "clinical"]},

    {"name": "medical_sops", "description": "Clinical IT support procedures, workstation setup, badge/PIV access for medical staff", "grs": "3.1",
     "acl_groups": ["it_support", "clinical", "medical"], "keywords": ["patient", "clinical", "workstation", "badge", "piv", "medical", "provider"]},

    {"name": "hipaa_compliance", "description": "HIPAA policies, PII/PHI handling procedures, breach notification, privacy impact assessments", "grs": "4.2",
     "acl_groups": ["cybersecurity", "compliance", "medical", "executives"], "keywords": ["hipaa", "phi", "pii", "privacy", "breach", "disclosure", "protected health"]},

    {"name": "medical_devices", "description": "Biomedical device integration documentation, HL7/DICOM interfaces, device inventory", "grs": "6.3",
     "acl_groups": ["it_support", "medical", "biomedical"], "keywords": ["biomedical", "device", "integration", "hl7", "dicom", "medical device", "biomed"]},

    # DoD / Organizational
    {"name": "dha_directives", "description": "DHA directives, DoDI, policy memorandums, official issuances", "grs": "5.7",
     "acl_groups": ["executives", "compliance"], "keywords": ["directive", "dodi", "dha", "memorandum", "policy", "issuance", "instruction"]},

    {"name": "contracts", "description": "Contract documents, SOWs, PWS, RFIs, CDRLs, deliverables", "grs": "1.1",
     "acl_groups": ["contracts", "executives"], "keywords": ["contract", "sow", "pws", "rfi", "deliverable", "cdrl", "acquisition", "procurement"]},

    {"name": "training", "description": "Training materials, user guides, tutorials, onboarding documentation", "grs": "2.6",
     "acl_groups": ["it_support", "medical", "engineering"], "keywords": ["training", "guide", "tutorial", "onboarding", "course", "certification", "user guide"]},

    {"name": "meeting_notes", "description": "Meeting transcripts, minutes, action items, briefing slides", "grs": "5.1",
     "acl_groups": ["it_support", "engineering", "executives"], "keywords": ["meeting", "minutes", "standup", "brief", "action item", "agenda", "transcript"]},

    {"name": "after_action", "description": "After action reports, lessons learned, hotwash documentation", "grs": "5.1",
     "acl_groups": ["it_support", "engineering", "executives"], "keywords": ["aar", "lessons learned", "hotwash", "debrief", "after action", "improvement"]},

    {"name": "budget_finance", "description": "Budget documents, funding requests, resource allocation, fiscal records", "grs": "1.1",
     "acl_groups": ["finance", "executives"], "keywords": ["budget", "funding", "allocation", "fiscal", "obligation", "expenditure", "cost"]},

    {"name": "continuity_planning", "description": "COOP plans, disaster recovery, business continuity documentation", "grs": "5.3",
     "acl_groups": ["it_support", "executives", "cybersecurity"], "keywords": ["coop", "disaster recovery", "continuity", "failover", "backup site", "dr plan"]},

    {"name": "security_management", "description": "Physical and information security management, access control, clearance records", "grs": "5.6",
     "acl_groups": ["cybersecurity", "executives"], "keywords": ["security", "access", "clearance", "badge", "physical security", "classification"]},
]


async def main():
    store = MetadataStore()
    await store.init()

    created = 0
    skipped = 0
    for cat in CATEGORIES:
        existing = await store.get_category(cat["name"])
        if existing:
            print(f"  SKIP: {cat['name']} (already exists)")
            skipped += 1
            continue
        await store.add_category(
            name=cat["name"],
            description=cat["description"],
            acl_groups=cat["acl_groups"],
            routing_keywords=cat["keywords"],
            grs_number=cat["grs"],
        )
        print(f"  Created: {cat['name']} (GRS {cat['grs']})")
        created += 1

    print(f"\nDone. Created {created}, skipped {skipped}.")

    # Seed ACL groups
    ACL_GROUPS = [
        {"name": "it_support", "display_name": "IT Support", "description": "IT help desk and support staff"},
        {"name": "devops", "display_name": "DevOps", "description": "DevOps and infrastructure team"},
        {"name": "cybersecurity", "display_name": "Cybersecurity", "description": "Information security team"},
        {"name": "engineering", "display_name": "Engineering", "description": "Software and systems engineering"},
        {"name": "finance", "display_name": "Finance", "description": "Finance and budget team"},
        {"name": "executives", "display_name": "Executives", "description": "Leadership and executives"},
        {"name": "compliance", "display_name": "Compliance", "description": "Compliance and regulatory"},
        {"name": "contracts", "display_name": "Contracts", "description": "Contracts and procurement"},
        {"name": "clinical", "display_name": "Clinical", "description": "Clinical and medical staff"},
        {"name": "medical", "display_name": "Medical", "description": "Medical systems and support"},
        {"name": "biomedical", "display_name": "Biomedical", "description": "Biomedical device engineering"},
    ]

    print("\nSeeding ACL groups...")
    for g in ACL_GROUPS:
        await store.add_acl_group(**g)
        print(f"  {g['name']}: {g['display_name']}")

    print(f"Done. {len(ACL_GROUPS)} ACL groups seeded.")


if __name__ == "__main__":
    asyncio.run(main())
