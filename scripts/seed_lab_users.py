"""Generate lab test user tokens for testing ACL scenarios without AD.

Run this script to get pre-built JWT tokens for different user/group combos.
Use the tokens in API calls or paste them into Open WebUI.

Usage:
    python scripts/seed_lab_users.py
    python scripts/seed_lab_users.py --expiry 4320   # 3-day tokens
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.jwt import create_token

# Lab test users with different department access.
# Prefer editing personas in Admin → Settings → Security; these defaults
# match src.db.metadata.DEFAULT_PERSONAS for offline token generation.
try:
    from src.db.metadata import DEFAULT_PERSONAS
    LAB_USERS = [
        {
            "username": p["name"],
            "role": p.get("role") or p["name"],
            "groups": list(p.get("groups") or []),
        }
        for p in DEFAULT_PERSONAS
    ]
except Exception:
    LAB_USERS = [
        {
            "username": "mike",
            "role": "Finance Manager",
            "groups": ["finance", "executives"],
        },
        {
            "username": "bob",
            "role": "IT Support Engineer",
            "groups": ["it_support", "devops"],
        },
        {
            "username": "sarah",
            "role": "Software Engineer",
            "groups": ["engineering"],
        },
        {
            "username": "alice",
            "role": "Compliance Officer",
            "groups": ["finance", "it_support", "engineering", "executives", "compliance"],
        },
        {
            "username": "dave",
            "role": "Intern (limited access)",
            "groups": ["engineering"],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate lab user tokens")
    parser.add_argument("--expiry", type=int, default=1440, help="Token expiry in minutes (default: 1440 = 24h)")
    parser.add_argument("--json", action="store_true", help="Output as JSON for scripting")
    args = parser.parse_args()

    results = []

    for user in LAB_USERS:
        token = create_token(
            username=user["username"],
            groups=user["groups"],
            expiration_minutes=args.expiry,
        )
        results.append({**user, "token": token})

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 70)
    print("LAB TEST USER TOKENS")
    print(f"Expiry: {args.expiry} minutes ({args.expiry / 60:.0f} hours)")
    print("=" * 70)

    for user in results:
        print(f"\n--- {user['username']} ({user['role']}) ---")
        print(f"Groups: {', '.join(user['groups'])}")
        print(f"Token:  {user['token']}")

    print("\n" + "=" * 70)
    print("QUICK TEST COMMANDS (using curl):")
    print("=" * 70)

    # Use first user for examples
    mike = results[0]
    bob = results[1]

    print(f"""
# Query as Mike (finance access):
curl -X POST http://localhost:8080/api/v1/query \\
  -H "Authorization: Bearer {mike['token']}" \\
  -H "X-API-Key: dev-key-1" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What is the expense policy?"}}'

# Query as Bob (IT access — should NOT see finance docs):
curl -X POST http://localhost:8080/api/v1/query \\
  -H "Authorization: Bearer {bob['token']}" \\
  -H "X-API-Key: dev-key-1" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What is the expense policy?"}}'

# List documents visible to Mike:
curl http://localhost:8080/api/v1/documents \\
  -H "Authorization: Bearer {mike['token']}" \\
  -H "X-API-Key: dev-key-1"

# Upload a document with finance ACL:
curl -X POST http://localhost:8080/api/v1/ingest \\
  -H "Authorization: Bearer {mike['token']}" \\
  -H "X-API-Key: dev-key-1" \\
  -F "file=@test_fixtures/sample.pdf" \\
  -F 'acl_groups=["finance", "executives"]'
""")


if __name__ == "__main__":
    main()
