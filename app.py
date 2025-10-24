import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the core processing function from your original script
# This is why the file structure is important.
from hybrid_chat import process_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This allows your frontend (on a different "origin") to communicate with this server.
CORS(app)

@app.route('/ask', methods=['POST'])
def ask_assistant():
    """
    API endpoint to receive a query and return the assistant's response.
    """
    # 1. Get the user's query from the incoming JSON request
    data = request.get_json()
    if not data or 'query' not in data:
        logger.warning("Received invalid request: no query provided.")
        return jsonify({'error': 'No query provided in the request.'}), 400

    user_query = data['query']
    logger.info(f"Received query from frontend: \"{user_query}\"")

    try:
        # 2. Call the main processing function from your hybrid_chat script
        assistant_answer = process_query(user_query)
        
        if not assistant_answer:
             raise Exception("The assistant returned an empty response.")
        
        logger.info("Successfully generated a response.")

        # 3. Return the response as JSON
        return jsonify({'answer': assistant_answer})

    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your local network
    # debug=True provides helpful error messages during development
    print("="*70)
    print("🚀 Starting Hybrid AI Travel Assistant Server...")
    print("   Access the frontend by opening the index.html file in your browser.")
    print("   API is listening on http://127.0.0.1:5000")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True)