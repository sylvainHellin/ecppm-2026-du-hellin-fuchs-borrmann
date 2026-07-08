"""
Parse an IFC file and store its complete structure as a labeled property graph
in Neo4j.

Usage::

    from ifc2neo4j import Neo4jConnection, IfcToNeo4j

    Neo4jConnection(password="my-secret")       # configure once
    IfcToNeo4j().run("model.ifc", timestamp="v1")
"""

import ifcopenshell
from neomodel import db

from .neo4j_helper import Neo4jHelper


class IfcToNeo4j:
    """Convert an IFC STEP file into a Neo4j graph."""

    # ------------------------------------------------------------------ #
    #  public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, ifc_path: str, timestamp: str, batch_size: int = 20000):
        """
        Load *ifc_path*, classify every IFC entity, and bulk-insert the
        resulting nodes and edges into Neo4j.

        Parameters
        ----------
        ifc_path : str
            Path to the ``.ifc`` file.
        timestamp : str
            Arbitrary version label attached to every node so that
            multiple model snapshots can coexist in one database.
        batch_size : int
            Number of rows per ``UNWIND`` batch sent to Neo4j.
        """
        helper = Neo4jHelper()

        print("Loading IFC model.")
        model = ifcopenshell.open(ifc_path)

        # ---- classify entities ---------------------------------------- #
        primary_entities = (
            model.by_type("IfcObjectDefinition")
            + model.by_type("IfcPropertyDefinition")
        )
        connection_entities = model.by_type("IfcRelationship")
        prim_conn_ids = {e.id() for e in primary_entities + connection_entities}
        secondary_entities = [
            e for e in model if e.id() != 0 and e.id() not in prim_conn_ids
        ]

        # ---- build skeleton dicts ------------------------------------- #
        primary_nodes = [
            {
                "GlobalId": e.GlobalId,
                "EntityType": e.is_a(),
                "p21_id": f"#{e.id()}",
                "timestamp": timestamp,
            }
            for e in primary_entities
        ]

        connection_nodes = [
            {
                "GlobalId": e.GlobalId,
                "EntityType": e.is_a(),
                "p21_id": f"#{e.id()}",
                "timestamp": timestamp,
            }
            for e in connection_entities
        ]

        secondary_nodes = [
            {
                "EntityType": e.is_a(),
                "p21_id": f"#{e.id()}",
                "timestamp": timestamp,
            }
            for e in secondary_entities
        ]

        # ---- bulk-create nodes ---------------------------------------- #
        print(f"Creating {len(primary_nodes)} PrimaryNodes.")
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS props
            CREATE (n:PrimaryNode:GenericNode:Node)
            SET n = props
            """,
            primary_nodes,
            batch_size,
        )

        print(f"Creating {len(connection_nodes)} ConnectionNodes.")
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS props
            CREATE (n:ConnectionNode:GenericNode:Node)
            SET n = props
            """,
            connection_nodes,
            batch_size,
        )

        print(f"Creating {len(secondary_nodes)} SecondaryNodes.")
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS props
            CREATE (n:SecondaryNode:GenericNode:Node)
            SET n = props
            """,
            secondary_nodes,
            batch_size,
        )

        # ---- indexes (before bulk lookups for speed) ------------------- #
        print("Creating indexes for faster lookup.")
        db.cypher_query(
            "CREATE INDEX generic_p21_ts IF NOT EXISTS FOR (n:GenericNode) ON (n.p21_id, n.timestamp)"
        )
        db.cypher_query(
            "CREATE INDEX generic_ts IF NOT EXISTS FOR (n:GenericNode) ON (n.timestamp)"
        )
        db.cypher_query("CALL db.awaitIndexes()")

        # ---- extract attributes & relationships ----------------------- #
        print("Collecting attributes and relationships.")
        props_map: dict = {}
        relationships: list = []
        related_nodes: set = set()
        inline_patterns: list = []

        for entity in model:
            self._process_ifc_attributes(
                entity, timestamp, props_map, relationships, related_nodes, inline_patterns
            )

        # ---- bulk-update primitive attributes ------------------------- #
        print(f"Updating attributes on {len(props_map)} nodes.")
        attributes_list = [
            {"p21_id": p21_id, "timestamp": timestamp, "properties": properties}
            for p21_id, properties in props_map.items()
        ]
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS row
            MATCH (n:GenericNode {p21_id: row.p21_id, timestamp: row.timestamp})
            SET n += row.properties
            """,
            attributes_list,
            batch_size,
        )

        # ---- bulk-create relationships -------------------------------- #
        print(f"Creating {len(relationships)} relationships.")
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS r
            MATCH (a:GenericNode {p21_id: r.source_p21_id, timestamp: r.timestamp})
            MATCH (b:GenericNode {p21_id: r.target_p21_id, timestamp: r.timestamp})
            CREATE (a)-[:rel {rel_type: r.rel_type, list_index: r.list_index}]->(b)
            """,
            relationships,
            batch_size,
        )

        # ---- bulk-create inline nodes --------------------------------- #
        print(f"Creating {len(inline_patterns)} InlineNode patterns.")
        helper.bulk_cypher_query(
            """
            UNWIND $batch AS r
            MATCH (a:GenericNode {p21_id: r.relation.source_p21_id, timestamp: r.props.timestamp})
            CREATE (b:InlineNode:Node)
            SET b = r.props
            CREATE (a)-[:rel {rel_type: r.relation.rel_type, list_index: r.relation.list_index}]->(b)
            """,
            inline_patterns,
            batch_size,
        )

        print("Finished IFC → Neo4j import.")

    # ------------------------------------------------------------------ #
    #  internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _process_ifc_attributes(
        entity: ifcopenshell.entity_instance,
        timestamp: str,
        props_map: dict,
        relationships: list,
        related_nodes: set,
        inline_patterns: list,
    ):
        """Classify every attribute of *entity* into one of the four output collections."""
        p21_id = f"#{entity.id()}"

        def traverse(key, val, list_index=0):
            if isinstance(val, ifcopenshell.entity_instance):
                if val.id() == 0:
                    node_props: dict = {
                        "EntityType": val.is_a(),
                        "timestamp": timestamp,
                    }
                    if hasattr(val, "wrappedValue"):
                        node_props["wrappedValue"] = val.wrappedValue
                    else:
                        for k, v in val.get_info().items():
                            if k in ("id", "type"):
                                continue
                            if isinstance(v, ifcopenshell.entity_instance):
                                if v.id() == 0 and hasattr(v, "wrappedValue"):
                                    node_props[k] = v.wrappedValue
                                else:
                                    node_props[k] = str(v)
                            elif v is not None:
                                node_props[k] = str(v) if isinstance(v, (tuple, list)) else v
                    inline_patterns.append(
                        {
                            "props": node_props,
                            "relation": {
                                "rel_type": key,
                                "list_index": list_index,
                                "source_p21_id": p21_id,
                            },
                        }
                    )
                else:
                    related_p21_id = f"#{val.id()}"
                    relationships.append(
                        {
                            "source_p21_id": p21_id,
                            "target_p21_id": related_p21_id,
                            "timestamp": timestamp,
                            "rel_type": key,
                            "list_index": list_index,
                        }
                    )
                    related_nodes.add(related_p21_id)
            elif isinstance(val, (tuple, list)):
                if any(isinstance(x, ifcopenshell.entity_instance) for x in val):
                    for i, x in enumerate(val):
                        traverse(key, x, list_index=i)
                else:
                    props_map.setdefault(p21_id, {})[key] = str(val)
            elif val is None:
                props_map.setdefault(p21_id, {})[key] = "$"
            else:
                props_map.setdefault(p21_id, {})[key] = val

        info = entity.get_info()
        for key, val in info.items():
            if key in ("GlobalId", "EntityType", "type", "p21_id", "id", "inline_id", "timestamp"):
                continue
            traverse(key, val)
