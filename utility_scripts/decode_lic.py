from cryptography.fernet import Fernet
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Encrypt a folder with Fernet encryption."
    )
    parser.add_argument("--license", help="License file to decrypt")
    args = parser.parse_args()
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
    if ENCRYPTION_KEY is None:
        print("ENCRYPTION_KEY environment variable not set")
        return
    with open(args.license, "r") as file:
        encrypted_license = file.read()
    decrypted_license = Fernet(ENCRYPTION_KEY.encode()).decrypt(encrypted_license.encode()).decode()
    print(decrypted_license)

if __name__ == "__main__":
    main()