"""
Neo4j node and relationship schema for IFC graph representation.

Node hierarchy:
    Node (abstract)
    ├── GenericNode (has p21_id)
    │   ├── PrimaryNode     — IfcObjectDefinition + IfcPropertyDefinition (has GlobalId)
    │   ├── ConnectionNode  — IfcRelationship (has GlobalId)
    │   └── SecondaryNode   — all other STEP entities
    └── InlineNode          — anonymous inline entities (e.g. IfcArcIndex, IfcLineIndex)

Edges:
    :rel {rel_type, list_index}   — IFC attribute references between nodes
"""

from neomodel import (
    StructuredRel,
    StringProperty,
    RelationshipTo,
    RelationshipFrom,
    Relationship,
    IntegerProperty,
)
from neomodel.contrib import SemiStructuredNode


class RelProperties(StructuredRel):
    rel_type = StringProperty(required=True)
    list_index = IntegerProperty()


class Node(SemiStructuredNode):
    EntityType = StringProperty(required=True)

    relation_to = RelationshipTo("Node", "rel", model=RelProperties)
    relation_from = RelationshipFrom("Node", "rel", model=RelProperties)
    equivalent_to = Relationship("Node", "equivalent_to")

    timestamp = StringProperty(required=True)


class GenericNode(Node):
    p21_id = StringProperty(required=True)


class PrimaryNode(GenericNode):
    GlobalId = StringProperty(unique_index=True, required=True)

    def __repr__(self):
        return f"PrimaryNode(GlobalId='{self.GlobalId}', EntityType='{self.EntityType}', timestamp='{self.timestamp}')"


class SecondaryNode(GenericNode):
    def __repr__(self):
        return f"SecondaryNode(EntityType='{self.EntityType}', timestamp='{self.timestamp}')"


class ConnectionNode(GenericNode):
    GlobalId = StringProperty(unique_index=True, required=True)

    def __repr__(self):
        return f"ConnectionNode(GlobalId='{self.GlobalId}', EntityType='{self.EntityType}', timestamp='{self.timestamp}')"


class InlineNode(Node):
    """
    Inline entities with id=0 in IFC, e.g.:
    #92=IFCINDEXEDPOLYCURVE(#91,(IFCLINEINDEX((1,2)),IFCARCINDEX((2,3,4))),.F.);
    """

    def __repr__(self):
        return f"InlineNode(EntityType='{self.EntityType}', timestamp='{self.timestamp}')"
