# groq_hybrid_chat_ollama_only.py
"""
Hybrid AI Travel Assistant using Groq API and a local Ollama model.
Groq: Used for fast chat completions.
Ollama: Used for local, private text embeddings.
"""

import json
import time
from typing import List, Dict, Optional
from functools import lru_cache
import requests  # For calling local Ollama API

from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
# Ollama configuration is now mandatory for embeddings
try:
    OLLAMA_MODEL = config.OLLAMA_MODEL
    OLLAMA_API_BASE = getattr(config, "OLLAMA_API_BASE", "http://localhost:11434")
except AttributeError:
    print("❌ Configuration Error: 'OLLAMA_MODEL' must be defined in your config.py file.")
    exit(1)

# Groq LLM for chat completions
CHAT_MODEL = "llama-3.3-70b-versatile"
# Alternative Groq models:
# - "llama3-70b-8192" (Large and capable)
# - "llama3-8b-8192" (Fastest)

TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
CACHE_SIZE = 256

# -----------------------------
# Initialize clients
# -----------------------------
print(f"✅ Using local Ollama for embeddings with model: {OLLAMA_MODEL}")
groq_client = Groq(api_key=config.GROQ_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Connect to Neo4j (optional - will work without it)
try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    NEO4J_AVAILABLE = True
    print("✅ Neo4j connected")
except Exception as e:
    print(f"⚠️  Neo4j not available: {e}")
    print("   Running in vector-only mode (Pinecone + Groq)")
    NEO4J_AVAILABLE = False
    driver = None

# -----------------------------
# Caching & Embedding Layer (Ollama only)
# -----------------------------
@lru_cache(maxsize=CACHE_SIZE)
def _get_embedding_cached_ollama(text: str) -> tuple:
    """
    Internal function to get an embedding for a text string from a local Ollama model.
    Results are cached to avoid redundant API calls.
    """
    print(f"📦 Generating Ollama embedding for '{text[:35]}...'")
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError("Ollama API response did not contain an 'embedding' key.")
        return tuple(embedding)
    except Exception as e:
        print(f"❌ Ollama embedding error: {e}")
        # Return a zero-vector on failure to prevent crashes downstream
        return tuple([0.0] * config.PINECONE_VECTOR_DIM)

def embed_text(text: str) -> List[float]:
    """Get embedding for a text string (wrapper for the cached Ollama call)."""
    start_time = time.time()
    embedding_tuple = _get_embedding_cached_ollama(text)
    elapsed = time.time() - start_time
    
    # Heuristic to determine if the result was served from the cache
    if elapsed < 0.01:
        print(f"💾 Cache hit for embedding. ({elapsed:.4f}s)")
    else:
        print(f"   -> Embedding generated in {elapsed:.2f}s")
        
    return list(embedding_tuple)

# -----------------------------
# Groq-powered functions
# -----------------------------
def groq_chat(messages: List[Dict], temperature: float = 0.3, max_tokens: int = 1000) -> str:
    """Call Groq for chat completion - extremely fast."""
    try:
        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return f"Error communicating with Groq: {e}"

def extract_query_intent(user_query: str) -> Dict:
    """Extract structured intent using Groq."""
    intent_prompt = f"""Analyze this travel query and extract structured information:
Query: "{user_query}"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "city": "city name or null",
  "type": "attraction/hotel/activity/city or null",
  "tags": ["tag1", "tag2"],
  "temporal": "time info or null",
  "trip_type": "solo/family/romantic/business or null"
}}

JSON only:"""
    try:
        response_text = groq_chat(
            [{"role": "user", "content": intent_prompt}],
            temperature=0.1,
            max_tokens=200
        )
        # Clean response to ensure it's valid JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```").strip()
        
        intent = json.loads(response_text)
        print(f"🎯 Intent: {intent}")
        return intent
    except (json.JSONDecodeError, IndexError) as e:
        print(f"⚠️  Intent extraction failed to parse JSON: {e}")
        return {}
    except Exception as e:
        print(f"⚠️  Intent extraction failed: {e}")
        return {}

# -----------------------------
# Vector Search
# -----------------------------
def pinecone_query(query_text: str, top_k=TOP_K, filters: Optional[Dict] = None):
    """Query Pinecone index using a locally generated Ollama embedding."""
    vec = embed_text(query_text)
    
    query_params = {
        "vector": vec,
        "top_k": top_k,
        "include_metadata": True,
        "include_values": False
    }
    if filters:
        query_params["filter"] = filters
    
    res = index.query(**query_params)
    print(f"🔍 Pinecone: Found {len(res['matches'])} results")
    return res["matches"]

# -----------------------------
# Graph Context (Optional)
# -----------------------------
def fetch_graph_context(node_ids: List[str], max_per_node=10):
    """Fetch neighboring nodes from Neo4j to add relational context."""
    if not NEO4J_AVAILABLE or not driver:
        print("🕸️  Neo4j: Skipped (not available)")
        return []
    
    facts = []
    try:
        with driver.session() as session:
            q = """
            UNWIND $node_ids AS nid
            MATCH (n:Entity {id: nid})
            OPTIONAL MATCH (n)-[r]->(m:Entity)
            WHERE m.id IS NOT NULL
            RETURN 
                n.id AS source_id, n.name AS source_name,
                type(r) AS rel_type,
                m.id AS target_id, m.name AS target_name
            LIMIT $limit
            """
            result = session.run(q, node_ids=node_ids, limit=max_per_node * len(node_ids))
            for record in result:
                facts.append(record.data())
        print(f"🕸️  Neo4j: Found {len(facts)} relationships")
    except Exception as e:
        print(f"⚠️  Neo4j query error: {e}")
    return facts

# -----------------------------
# Result Ranking & Prompt Building
# -----------------------------
def build_prompt(user_query: str, vector_matches, graph_facts):
    """Build the final prompt for Groq with all retrieved context."""
    system_prompt = """You are a helpful and expert travel assistant. Your goal is to provide specific, actionable recommendations based on the user's query and the context provided.
Guidelines:
1. Synthesize information from the 'Vector Search Results' and 'Knowledge Graph Connections'.
2. Be concise, clear, and directly answer the user's question.
3. When you mention a specific place, cite its ID in brackets, like [hotel_123]. This is very important.
4. If the context is empty or irrelevant, state that you couldn't find specific information but try to provide a general helpful answer."""

    # Format vector search context
    vec_context_str = "\n".join([
        f"- [{match['id']}] {match['metadata'].get('name', 'N/A')} (Type: {match['metadata'].get('type', 'N/A')}, Score: {match.get('score', 0):.2f})"
        for match in vector_matches
    ]) if vector_matches else "No relevant places found."

    # Format graph context
    graph_context_str = "\n".join([
        f"- {fact['source_name']} [{fact['source_id']}] is related to {fact['target_name']} [{fact['target_id']}] via '{fact['rel_type']}'"
        for fact in graph_facts
    ]) if graph_facts else "No direct connections found in the knowledge graph."

    user_content = f"""**User Query:** {user_query}

**Context from Vector Search Results:**
{vec_context_str}

**Context from Knowledge Graph Connections:**
{graph_context_str}

**Your Task:**
Based on all the provided context, give a helpful travel recommendation. Remember to cite sources with their [ID]."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

# -----------------------------
# Main Pipeline
# -----------------------------
def process_query(query: str) -> str:
    """Process a user query through the full RAG pipeline."""
    start_time = time.time()
    print(f"\n{'='*70}\nProcessing query: {query}\n{'='*70}")
    
    # Step 1: Vector search for relevant nodes
    matches = pinecone_query(query, top_k=TOP_K)
    if not matches:
        return "I couldn't find any relevant results for your query. Please try rephrasing it."
    
    # Step 2: Graph expansion for relational context
    match_ids = [m["id"] for m in matches]
    graph_facts = fetch_graph_context(match_ids)
    
    # Step 3: Build a comprehensive prompt
    prompt_messages = build_prompt(query, matches, graph_facts)
    
    # Step 4: Generate a response with Groq's fast LLM
    answer = groq_chat(prompt_messages, temperature=0.4, max_tokens=1500)
    
    elapsed = time.time() - start_time
    print(f"⏱️  Total processing time: {elapsed:.2f}s")
    return answer

# -----------------------------
# Interactive CLI
# -----------------------------
def interactive_chat():
    """Starts an interactive command-line chat session."""
    print("\n" + "="*70)
    print("🚀 HYBRID AI TRAVEL ASSISTANT")
    print("   Powered by: Groq (Chat) + Pinecone (VectorDB) + Neo4j (Graph)")
    print(f"   Embeddings by: Local Ollama ({OLLAMA_MODEL})")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Lightning-fast LLM responses with Groq")
    print("  ✓ Private, local embeddings with Ollama")
    print("  ✓ Semantic search with Pinecone")
    print(f"  ✓ Graph context with Neo4j {'✅' if NEO4J_AVAILABLE else '⚠️ (disabled)'}")
    print("="*70)
    print("\nType your question or 'exit' to quit.\n")
    
    while True:
        try:
            query = input("🌍 Your travel question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("\n✈️  Safe travels! Goodbye!\n")
                break
            
            answer = process_query(query)
            
            print("\n" + "="*70)
            print("✨ ASSISTANT RESPONSE")
            print("="*70)
            print(answer)
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Type 'exit' to quit.\n")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            print("   Please check your API keys, network connection, and service statuses.\n")

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        if driver: 
            driver.close()
            print("Neo4j connection closed.")