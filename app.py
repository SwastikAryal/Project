#!/usr/bin/env python3
"""
Flask Web API for Manual Vector Search Legal RAG
NyayaSathi - Legal Chat Application with MySQL Database
"""

import os
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps

from config import Config
from models import db, User, ChatSession, init_db, get_user_chats, get_user_chat, create_chat, update_chat_title, toggle_chat_pin, delete_chat
from legal_rag_manual import StrictLegalRAG

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = Config.get_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, supports_credentials=True)
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Initializing Manual Vector Search System...")
rag = StrictLegalRAG(os.path.join(BASE_DIR, "manual_vectors.json"))
print("Ready!")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def init_database():
    try:
        if Config.USE_SQLITE:
            if not os.path.exists(Config.SQLITE_DB_PATH):
                init_db(app)
                print(f"Created SQLite database at {Config.SQLITE_DB_PATH}")
            else:
                print(f"Using existing SQLite database at {Config.SQLITE_DB_PATH}")
        else:
            init_db(app)
            print("Connected to MySQL database")
    except Exception as e:
        print(f"Database initialization info: {e}")
        print("Falling back to SQLite...")
        app.config['SQLALCHEMY_DATABASE_URI'] = Config.get_sqlite_uri()
        init_db(app)


init_database()


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template("index.html", user=current_user)
    return render_template("index.html", user=None)


@app.route("/login")
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template("login.html")


@app.route("/register")
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template("register.html")


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    
    user = User(username=username, email=email if email else None)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    login_user(user)
    return jsonify({"success": True, "user": user.to_dict()})


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    login_user(user)
    return jsonify({"success": True, "user": user.to_dict()})


@app.route("/auth/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"success": True})


@app.route("/auth/status")
def auth_status():
    if current_user.is_authenticated:
        return jsonify({"authenticated": True, "user": current_user.to_dict()})
    return jsonify({"authenticated": False})


@app.route("/css/site.css")
def site_css():
    return send_from_directory(os.path.join(BASE_DIR, "static", "css"), "site.css")


@app.route("/Chat/Sessions")
@login_required
def get_sessions():
    chats = get_user_chats(current_user.id)
    pinned = [c.to_dict() for c in chats if c.is_pinned]
    unpinned = [c.to_dict() for c in chats if not c.is_pinned]
    return jsonify({"sessions": {"pinned": pinned, "unpinned": unpinned}})


@app.route("/Chat/Sessions/<int:session_id>")
@login_required
def get_session(session_id):
    chat = get_user_chat(current_user.id, session_id)
    if not chat:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(chat.to_dict())


@app.route("/Chat/Sessions/<int:session_id>", methods=["PUT"])
@login_required
def update_session(session_id):
    data = request.json
    chat = get_user_chat(current_user.id, session_id)
    if not chat:
        return jsonify({"error": "Session not found"}), 404
    
    if 'title' in data:
        update_chat_title(session_id, data['title'])
    if 'isPinned' in data:
        toggle_chat_pin(session_id)
    
    return jsonify({"success": True})


@app.route("/Chat/Sessions/<int:session_id>", methods=["DELETE"])
@login_required
def delete_session(session_id):
    if delete_chat(session_id):
        return jsonify({"success": True})
    return jsonify({"error": "Session not found"}), 404


@app.route("/Chat/Stream", methods=["POST"])
@login_required
def stream_chat():
    data = request.json
    query = data.get("query", "").strip()
    history = data.get("history", [])
    session_id = data.get("sessionId")
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    if len(query) > 1000:
        return jsonify({"error": "Query too long (max 1000 characters)"}), 400
    
    chat = None
    if session_id:
        chat = get_user_chat(current_user.id, session_id)
    
    result = rag.answer(query)
    
    if not chat:
        title = query[:50] + ('...' if len(query) > 50 else '')
        chat = create_chat(current_user.id, title)
    
    chat.add_message('user', query, result.get('sources', []))
    chat.add_message('assistant', result.get('answer', ''), result.get('sources', []))
    db.session.commit()
    
    return jsonify({
        "sessionId": chat.id,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query required"}), 400
        
        if len(query) > 1000:
            return jsonify({"error": "Query too long (max 1000 characters)"}), 400
        
        result = rag.answer(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/search", methods=["POST"])
def search_debug():
    data = request.json
    if not data:
        return jsonify({"error": "Request body required"}), 400
    
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    if len(query) > 1000:
        return jsonify({"error": "Query too long (max 1000 characters)"}), 400
    
    top_k = data.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        top_k = 5
    
    results = rag.retriever.search(query, top_k=top_k)
    
    return jsonify({
        "query": query,
        "algorithm": "manual_cosine_similarity",
        "complexity": "O(N×D + N log N)",
        "results": [
            {
                "rank": r.rank,
                "id": r.id,
                "section": r.section,
                "page": r.page,
                "score": r.score,
                "text": r.text[:300] + "..."
            }
            for r in results
        ]
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    rag.reset()
    return jsonify({"status": "reset"})


@app.route("/api/stats")
def stats():
    user_count = User.query.count() if hasattr(User, 'query') else 0
    chat_count = ChatSession.query.count() if hasattr(ChatSession, 'query') else 0
    
    return jsonify({
        "documents": len(rag.retriever.documents),
        "dimension": rag.retriever.dimension,
        "algorithm": "Brute-force k-NN with manual Cosine Similarity",
        "storage": "MySQL/SQLite Database",
        "database": "MySQL" if not Config.USE_SQLITE else "SQLite",
        "total_users": user_count,
        "total_chats": chat_count
    })


if __name__ == "__main__":
    print("=" * 60)
    print("NyayaSathi Legal RAG Server")
    print("=" * 60)
    print(f"Documents: {len(rag.retriever.documents)}")
    print(f"Dimensions: {rag.retriever.dimension}")
    print(f"Algorithm: Manual Cosine Similarity")
    print(f"Database: {'MySQL' if not Config.USE_SQLITE else 'SQLite'}")
    print("=" * 60)
    print("Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)