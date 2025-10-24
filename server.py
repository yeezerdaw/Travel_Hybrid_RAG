from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys

# Import the project pipeline. This will initialize clients on import.
import hybrid_chat

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True) or {}
    query = data.get('query', '')
    if not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Empty query'}), 400

    query = query.strip()
    logger.info(f'Received query: {query[:80]}')

    try:
        # Call the hybrid_chat pipeline
        answer = hybrid_chat.process_query(query)
        return jsonify({'answer': answer})
    except Exception as e:
        logger.exception('Error processing query')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Allow overriding host/port via environment variables for flexibility
    host = os.getenv('HOST', '127.0.0.1')
    try:
        port = int(os.getenv('PORT', '5000'))
    except ValueError:
        logger.error('Invalid PORT environment variable, must be an integer')
        sys.exit(1)

    try:
        logger.info(f"Starting server on http://{host}:{port} (use PORT env to change)")
        app.run(host=host, port=port)
    except OSError as e:
        # Common cause: Address already in use
        if 'Address already in use' in str(e) or getattr(e, 'errno', None) == 98:
            logger.error(f"Port {port} is already in use. Choose a different PORT or stop the process using it.")
            logger.info("To find the process on macOS: lsof -nP -iTCP:{port} | grep LISTEN")
            logger.info("Example: PORT=5001 python3 server.py")
            sys.exit(1)
        else:
            raise
