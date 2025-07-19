from flask import Flask, request, jsonify, send_from_directory
from rag_chain import hybrid_qa

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get('question', '').strip()
        print(f"📩 Received question: {question}")

        if not question:
            return jsonify({'answer': 'Please enter a question.'}), 400

        answer = hybrid_qa(question)
        print(f"✅ Answer: {answer}")
        return jsonify({'answer': answer}), 200

    except Exception as e:
        print(f"❌ Error during QA: {e}")
        return jsonify({'answer': "An unexpected error occurred. Please try again later."}), 500

# Serve PWA manifest
@app.route('/manifest.json')
def manifest():
    return send_from_directory(app.static_folder, 'manifest.json')

# Serve the service worker
@app.route('/service-worker.js')
def service_worker():
    return send_from_directory(app.static_folder, 'service-worker.js')

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

# Fallback route for other static assets
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print(" Starting Guardiané Flask app on http://localhost:5000")
    app.run(debug=False)
