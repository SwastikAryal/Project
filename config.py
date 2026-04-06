import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'nyayasathi-secret-key-change-in-production')
    
    # MySQL Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'nyayasathi_db')
    
    @classmethod
    def get_db_uri(cls):
        return f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}?charset=utf8mb4"
    
    # SQLite Fallback (for development without MySQL)
    USE_SQLITE = os.getenv('USE_SQLITE', 'true').lower() == 'true'
    SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nyayasathi.db')
    
    @classmethod
    def get_sqlite_uri(cls):
        return f"sqlite:///{cls.SQLITE_DB_PATH}"
    
    @classmethod
    def get_uri(cls):
        if cls.USE_SQLITE:
            return cls.get_sqlite_uri()
        return cls.get_db_uri()