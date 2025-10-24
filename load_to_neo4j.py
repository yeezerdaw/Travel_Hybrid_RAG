# load_to_neo4j_enhanced.py
import json
import random
from neo4j import GraphDatabase
from tqdm import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def create_constraints(tx):
    """Creates uniqueness constraints on the node IDs."""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def enhance_node_data(node: dict) -> dict:
    """
    Adds missing properties ('category') and simulates realistic data 
    ('rating', 'price_range') to enrich the graph.
    """
    # 1. Use the 'type' field to create the 'category' property.
    node['category'] = node.get('type', 'Unknown')

    # 2. Simulate realistic ratings and price ranges based on the category.
    category = node['category']
    if category in ["Hotel", "Resort"]:
        node['rating'] = round(random.uniform(3.5, 5.0), 1)
        node['price_range'] = random.choice(["$$", "$$$", "$$$$"])
    elif category == "Restaurant":
        node['rating'] = round(random.uniform(3.8, 4.9), 1)
        node['price_range'] = random.choice(["$", "$$", "$$$"])
    elif category == "Activity":
        node['rating'] = round(random.uniform(4.0, 5.0), 1)
        # Price for activities can be represented numerically for easier filtering
        node['price_usd'] = random.choice([10, 15, 20, 25, 50, 75])
    
    # Other types like 'City', 'Region' won't have ratings or prices.
    return node

def upsert_node(tx, node):
    """Merges a node into the graph, setting its labels and properties."""
    labels = [node.get("type", "Unknown"), "Entity"]
    label_cypher = ":".join(labels)
    
    # Filter out properties we don't want to store directly.
    props = {k: v for k, v in node.items() if k not in ("connections",)}
    
    query = (
        f"MERGE (n:{label_cypher} {{id: $id}}) "
        "SET n += $props"
    )
    tx.run(query, id=node["id"], props=props)

def create_relationship(tx, source_id, rel):
    """Merges a relationship between two existing nodes."""
    rel_type = rel.get("relation", "RELATED_TO")
    target_id = rel.get("target")
    if not target_id:
        return
        
    query = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )
    tx.run(query, source_id=source_id, target_id=target_id)

def main():
    """Main function to load and process the data."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with driver.session(database="neo4j") as session:
        session.execute_write(create_constraints)
        
        # --- Node Upsertion with Enhancement ---
        for node in tqdm(nodes, desc="Upserting enhanced nodes"):
            enhanced_node = enhance_node_data(node)
            session.execute_write(upsert_node, enhanced_node)

        # --- Relationship Creation ---
        for node in tqdm(nodes, desc="Creating relationships"):
            for rel in node.get("connections", []):
                session.execute_write(create_relationship, node["id"], rel)
    
    driver.close()
    print("\n✅ Successfully loaded enhanced data into Neo4j.")
    print("   Your graph now contains 'category', 'rating', and 'price_range' properties.")

if __name__ == "__main__":
    main()