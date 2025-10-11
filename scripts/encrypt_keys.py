import getpass
from pathlib import Path

from decouple import Config, RepositoryEnv

from ..config.logger import logger
from ..utils.security import KeyEncryptor


def run():
    env_file = Path(".env")
    if not env_file.exists():
        logger.error(".env file not found. Please create it first.")
        return

    password = getpass.getpass("Enter a password to encrypt your API keys: ")
    if not password:
        logger.error("Password cannot be empty.")
        return

    encryptor = KeyEncryptor(password)
    config = Config(RepositoryEnv(str(env_file)))

    keys_to_encrypt = [
        "CRYPTOPANIC_KEY",
        "TELEGRAM_BOT_TOKEN",
        "ALPHA_VANTAGE_KEY",
        "COINDESK_API_KEY",
        "COINGECKO_KEY",
        "MESSARI_API_KEY",
        "SENTRY_DSN",
    ]

    new_env_content = ""
    updated_keys = set()

    with open(env_file, "r") as f:
        for line in f:
            key_in_line = line.split("=")[0].strip()
            if key_in_line in keys_to_encrypt:
                value = config(key_in_line, default="")
                if value and isinstance(value, str):
                    encrypted_value = encryptor.encrypt(value)
                    new_env_content += f"ENCRYPTED_{key_in_line}={encrypted_value}\n"
                    logger.info(f"Encrypted {key_in_line}")
                    updated_keys.add(key_in_line)
                else:
                    new_env_content += line
            else:
                new_env_content += line

    new_env_content += f'\nENCRYPTION_PASSWORD="{password}"\n'

    with open(env_file, "w") as f:
        f.write(new_env_content)

    logger.info("Encryption complete. .env file has been updated.")
    logger.warning("Please remove the original plain-text keys if they are no longer needed.")
    logger.info(f"Keys updated: {', '.join(updated_keys)}")


if __name__ == "__main__":
    run()
    
