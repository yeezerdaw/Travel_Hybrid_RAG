Hybrid AI Travel Assistant
This repository contains a fully operational, debugged, and enhanced Hybrid AI Travel Assistant. The system was developed from an initial semi-functional prototype and now stands as a robust example of a Retrieval-Augmented Generation (RAG) application.

The assistant leverages a hybrid pipeline, combining semantic search from Pinecone and graph-based context from Neo4j. This enriched context is then processed by a large language model via Groq to generate intelligent, accurate, and helpful travel recommendations for Vietnam.

Key Features & Deliverables
✅ Fully Functional Scripts: pinecone_upload.py, load_to_neo4j_enhanced.py, and hybrid_chat.py are stable and run end-to-end.
✅ Detailed Improvements Documentation: This README provides a comprehensive breakdown of all fixes, enhancements, and design choices.
✅ Example Interaction Log: A log of a successful interaction is provided below to demonstrate system performance.
Example Successful Interaction Log
code
Code

download

content_copy

expand_less
🌍 Your travel question: create a romantic 4 day itinerary for Vietnam

======================================================================
🌍 PROCESSING: create a romantic 4 day itinerary for Vietnam
======================================================================
INFO:__main__:🔄 Embedding generated (0.21s)
INFO:__main__:📋 Intent extracted with avg confidence: 87.5%
INFO:__main__:🔍 Querying Pinecone: 'create a romantic 4 day itinerary for Vietnam...'
INFO:__main__:⚡ Cache hit (0.0001s)
INFO:__main__:   → 8 results → 6 filtered → 5 reranked
INFO:__main__:🕸️  Neo4j: 40 relationships (up to 2-hop)
INFO:__main__:🤖 Generating response with Groq...
INFO:__main__:✅ Response generated in 2.85s
   • Vector results: 5
   • Graph facts: 40
   • Intent confidence: 87.5%

======================================================================
✨ ASSISTANT RESPONSE
======================================================================
Of course! Here is a romantic 4-day itinerary for Vietnam, focusing on the charm of Hanoi and the breathtaking beauty of a Halong Bay cruise. This plan balances cultural exploration with serene, unforgettable experiences.

**Itinerary Overview:**
*   **Day 1:** Hanoi's Old Quarter Charm & Culinary Delights
*   **Day 2:** Cultural Deep Dive in Hanoi
*   **Day 3:** Overnight Luxury Cruise in Halong Bay
*   **Day 4:** Morning in the Bay & Return to Hanoi

---

**Day 1: Arrival in Hanoi & Old Quarter Romance**

*   **Morning/Afternoon:** Arrive at your hotel in Hanoi. I recommend the **Sofitel Legend Metropole Hanoi [hotel_sofitel_metropole]** for a truly luxurious and historic romantic experience. Settle in and take a leisurely stroll around Hoan Kiem Lake.
*   **Evening:** Immerse yourselves in the vibrant energy of the **Hanoi Old Quarter [place_hanoi_old_quarter]**. For dinner, experience authentic Vietnamese cuisine at **Cha Ca Thang Long [restaurant_cha_ca_thang_long]**, which is famous for its grilled fish. It's a fantastic local spot located right in the heart of the Old Quarter.

**Day 2: Hanoi's History and Culture**

*   **Morning:** Visit the **Temple of Literature [activity_temple_of_literature]**, Vietnam's first university. Its peaceful courtyards and traditional architecture provide a serene start to the day.
*   **Afternoon:** Explore the Ho Chi Minh Mausoleum complex and the nearby One Pillar Pagoda. Later, enjoy a traditional Water Puppet Show, a unique Vietnamese art form.
*   **Evening:** For a special romantic dinner, book a table at **La Badiane [restaurant_la_badiane]**, known for its beautiful French-inspired fusion cuisine in a lovely courtyard setting.

**Day 3: Journey to Halong Bay**

*   **Morning:** An arranged shuttle will pick you up for the scenic drive to Halong Bay.
*   **Afternoon:** Board your pre-booked luxury overnight cruise, such as the **Paradise Elegance Cruise Halong [hotel_paradise_elegance_cruise]**. After settling into your cabin, enjoy lunch as you sail past stunning limestone karsts.
*   **Evening:** Participate in activities like kayaking or swimming. As the sun sets, enjoy cocktails on the sundeck followed by a gourmet dinner onboard. The quiet of the bay at night is incredibly romantic.

**Day 4: Sunrise in the Bay & Departure**

*   **Morning:** Wake up to the serene beauty of Halong Bay. You might enjoy a Tai Chi session on the deck before visiting a cave like Sung Sot (Surprise Cave). Enjoy a final brunch on the cruise as you sail back to the harbor.
*   **Afternoon:** Your shuttle will take you back to Hanoi for your onward journey.

This itinerary offers a perfect blend of culture, cuisine, and natural beauty, creating a memorable and romantic getaway.
======================================================================
⏱️  Response time: 2.85s
⚡ Good response time
Technical Improvements and Design Choices
This section details the work done to transform the initial prototype into a high-performing application.

1. Core Functionality and Bug Fixes
Pinecone SDK v3+ Compatibility: All Pinecone client operations were updated to the modern SDK, resolving deprecation errors and ensuring stable connectivity.
Strategic API Replacement: The initial dependency on openai was replaced with groq for LLM inference and a local ollama instance for embeddings. This shift enhances performance, reduces operational costs, and demonstrates architectural flexibility.
Dependency Management: A requirements.txt file was created to ensure a reproducible setup.
2. Code Design and Prompt Engineering
Modular Architecture: The system is organized into distinct components: a config.py for settings, hybrid_chat.py for core logic, and a server.py to expose the functionality via a clean Flask API.
Advanced Prompting:
Intent Extraction: An initial LLM call parses the user's query into structured data (destination, style, etc.) and calculates confidence scores, leading to more targeted responses.
Chain-of-Thought (CoT): The system prompt guides the LLM to follow a logical reasoning process (UNDERSTAND -> ANALYZE -> SYNTHESIZE -> RECOMMEND).
Structured Context: Vector and graph results are formatted with clear Markdown, making it easier for the model to parse and cite information accurately.
3. Enhanced Neo4j Graph Queries
2-Hop Relationship Traversal: The Cypher query now explores relationships up to two hops away, uncovering deeper, non-obvious connections between travel entities.
Enriched Data Payload: The query returns detailed information about connected nodes, including ratings, price ranges, and descriptions, adding valuable context.
Intelligent Sorting: Graph results are prioritized by path depth and then by rating, ensuring the most relevant relationships are presented to the LLM.
4. Advanced Features
Embedding Caching: The system uses @lru_cache to cache text embeddings, minimizing latency on repeated queries and reducing redundant computations.
Conversation Memory: A deque stores the last five conversation turns, providing the LLM with the necessary context to handle follow-up questions effectively.
Semantic Reranking: A custom function boosts the score of vector search results whose metadata directly matches keywords in the user's query, improving the relevance of the retrieved documents.
API and Frontend: A complete Flask server and a single-page HTML/CSS/JS frontend were developed to provide a polished, interactive user experience.
Setup and Running Instructions
Prerequisites:
Python 3.8+
Docker
Ollama installed and running
Clone & Install Dependencies:
code
Bash

download

content_copy

expand_less
git clone <repository_url>
cd hybrid-ai-travel-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Configure:
Copy config.py.example to config.py.
Fill in your API keys (PINECONE_API_KEY, GROQ_API_KEY) and set your Neo4j password.
Launch Services:
Neo4j: docker run --name neo4j-travel -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/YOUR_NEO4J_PASSWORD neo4j:latest
Ollama: ollama pull mxbai-embed-large (ensure the server is running with ollama serve).
Load Data:
code
Bash

download

content_copy

expand_less
python load_to_neo4j_enhanced.py
python pinecone_upload.py
Run Application:
Start the backend: python server.py
Open index.html in your browser.
