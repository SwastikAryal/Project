from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    chats = db.relationship('ChatSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    messages = db.Column(db.Text, default='[]')
    is_pinned = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_messages(self):
        try:
            return json.loads(self.messages) if self.messages else []
        except:
            return []
    
    def set_messages(self, messages_list):
        self.messages = json.dumps(messages_list, ensure_ascii=False)
    
    def add_message(self, role, content, sources=None):
        messages = self.get_messages()
        msg = {'role': role, 'content': content}
        if sources:
            msg['sources'] = sources
        messages.append(msg)
        self.set_messages(messages)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'is_pinned': self.is_pinned,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'messages': self.get_messages()
        }


def init_db(app):
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")


def get_user_chats(user_id):
    return ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()


def get_user_chat(user_id, chat_id):
    return ChatSession.query.filter_by(id=chat_id, user_id=user_id).first()


def create_chat(user_id, title):
    chat = ChatSession(user_id=user_id, title=title)
    db.session.add(chat)
    db.session.commit()
    return chat


def update_chat_title(chat_id, new_title):
    chat = ChatSession.query.get(chat_id)
    if chat:
        chat.title = new_title
        db.session.commit()
    return chat


def toggle_chat_pin(chat_id):
    chat = ChatSession.query.get(chat_id)
    if chat:
        chat.is_pinned = not chat.is_pinned
        db.session.commit()
    return chat


def delete_chat(chat_id):
    chat = ChatSession.query.get(chat_id)
    if chat:
        db.session.delete(chat)
        db.session.commit()
        return True
    return False