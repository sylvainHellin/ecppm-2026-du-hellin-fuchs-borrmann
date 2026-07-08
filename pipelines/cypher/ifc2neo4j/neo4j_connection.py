"""
Neo4j connection setup.

Reads credentials from a .env file (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD)
or falls back to a localhost bolt connection.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from neomodel import config, db


def _load_neo4j_config_from_env():
    """Search for a .env file upwards from cwd and return connection params."""
    search_paths = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path(__file__).parent.parent.parent,
    ]

    for search_path in search_paths:
        env_file = search_path / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_username = os.getenv("NEO4J_USERNAME")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if neo4j_uri and neo4j_username and neo4j_password:
                print(f"[Neo4j] Found .env file at: {env_file}")
                return {
                    "uri": neo4j_uri,
                    "username": neo4j_username,
                    "password": neo4j_password,
                }

    return None


class Neo4jConnection:
    """
    Configures neomodel's DATABASE_URL so that all subsequent OGM / db.cypher_query
    calls use the correct Neo4j endpoint.

    Usage::

        Neo4jConnection()                           # auto-detect from .env or localhost
        Neo4jConnection(password="my-secret")       # explicit localhost credentials
    """

    def __init__(
        self,
        username: str = "neo4j",
        password: str = "password",
        hostname: str = "localhost",
        port: int = 7687,
    ):
        env_config = _load_neo4j_config_from_env()

        if env_config:
            host = (
                env_config["uri"]
                .replace("neo4j+s://", "")
                .replace("neo4j://", "")
            )
            bolt_uri = f"bolt+ssc://{env_config['username']}:{env_config['password']}@{host}:7687"
            config.DATABASE_URL = bolt_uri
            print(f"[Neo4j] Connected to remote instance: {env_config['uri']}")
        else:
            config.DATABASE_URL = f"bolt://{username}:{password}@{hostname}:{port}"
            print(f"[Neo4j] Using localhost connection: {hostname}:{port}")

    def __getattr__(self, name):
        return getattr(db, name)
