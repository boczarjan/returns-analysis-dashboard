from __future__ import annotations

from getpass import getpass

from returns_dashboard.auth import DEFAULT_PASSWORD_HASH_PATH, write_password_hash


def main() -> None:
    password = getpass("Nowe haslo do aplikacji: ")
    repeated_password = getpass("Powtorz haslo: ")

    if not password:
        raise SystemExit("Haslo nie moze byc puste.")

    if password != repeated_password:
        raise SystemExit("Hasla nie sa takie same.")

    hash_path = write_password_hash(password)
    print(f"Zapisano hash hasla w: {hash_path}")


if __name__ == "__main__":
    main()
