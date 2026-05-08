"""Generate a random API key and print it."""
import secrets

def main():
    key = secrets.token_urlsafe(32)
    print(f"Generated API key: {key}")
    print(f"\nAdd to your .env file:")
    print(f"  API_KEYS=...existing-keys...,{key}")

if __name__ == "__main__":
    main()
