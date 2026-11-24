"""Clear only the dedicated local PostgreSQL Skill database public schema."""

import argparse

from mindmemos_skill.config.loader import read_skill_config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


def main():
    """Print the target or explicitly reset its schema after local safety checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mindmemos_skill/dev.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = read_skill_config(args.config).service
    spec = config.persistence.database
    if spec.class_name != "postgres":
        raise ValueError("This command supports the dedicated local PostgreSQL database only")
    options = spec.config.model_dump() if hasattr(spec.config, "model_dump") else spec.config
    url = make_url(options["url"])
    if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "mindmemos_skill":
        raise ValueError("Refusing to reset a nonlocal or nondedicated Skill database")
    print(f"Target: {url.host}:{url.port}/{url.database}; schema: public")
    if not args.execute:
        print("Preview only. Stop Skill API/workers, then add --execute to clear all Skill tables.")
        return
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("Skill schema cleared. Restart Skill service to recreate tables.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
