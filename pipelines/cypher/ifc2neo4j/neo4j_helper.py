"""Thin helpers for batched Cypher execution."""

from neomodel import db


class Neo4jHelper:

    @staticmethod
    def bulk_cypher_query(query: str, rows: list, batch_size: int):
        """Execute *query* in chunks of *batch_size* using UNWIND $batch."""
        total = len(rows)
        num_batches = (total + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, total, batch_size), 1):
            batch = rows[i : i + batch_size]
            print(f"  Batch {batch_idx}/{num_batches} ({min(i + batch_size, total)}/{total} rows)")
            db.cypher_query(query, {"batch": batch})

    @staticmethod
    def truncate_db(batch_size: int):
        """Delete all nodes and relationships (requires APOC plugin)."""
        count = db.cypher_query("MATCH (n) RETURN COUNT(n)")[0][0][0]
        print(f"Deleting {count} nodes.")
        query = """
            CALL apoc.periodic.commit("
            MATCH (n)
            WITH n LIMIT $limit
            DETACH DELETE n
            RETURN count(*)
            ", {limit:$batch_size});
        """
        db.cypher_query(query, {"batch_size": batch_size})
