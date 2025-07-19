from flask import Flask, request, jsonify, send_from_directory
from rag_chain import hybrid_qa

# Create the Flask app
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    print(f"📩 Received question: {question}")

    if not question:
        return jsonify({'answer': 'Please enter a question.'}), 400

    try:
        answer = hybrid_qa(question)
        print(f"✅ Answer: {answer}")
        return jsonify({'answer': answer}), 200
    except Exception as e:
        print(f"❌ Error during QA: {e}")
        return jsonify({'answer': "Oops! Something went wrong. Please try again later."}), 500

# Serve PWA manifest
@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

# Serve the service worker
@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('static', 'service-worker.js')

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

# Catch-all static file handler
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)

# App entry point
if __name__ == '__main__':
    print(" Starting Guardiané Flask app...")
    app.run(debug=False)
