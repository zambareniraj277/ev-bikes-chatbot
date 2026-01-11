import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

from groq import Groq

def get_db_context(query):
    """Get relevant context from database based on user query"""
    try:
        import psycopg2
        
        # Extract host from JDBC URL
        jdbc_url = os.getenv("SUPABASE_DB_HOST")
        host = jdbc_url.split("//")[1].split(":")[0]
        
        conn = psycopg2.connect(
            host=host,
            database=os.getenv("SUPABASE_DB_NAME"),
            user=os.getenv("SUPABASE_DB_USER"),
            password=os.getenv("SUPABASE_DB_PASSWORD"),
            port=os.getenv("SUPABASE_DB_PORT")
        )
        
        cursor = conn.cursor()
        
        # Simple query to get EV-related data (adjust based on your actual tables)
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cursor.fetchall()
        
        context = f"Available database tables: {[t[0] for t in tables]}"
        
        cursor.close()
        conn.close()
        
        return context
        
    except ImportError:
        return "Database module not installed. Install with: pip install psycopg2-binary"
    except Exception as e:
        return f"Database connection unavailable: {str(e)}"

def get_chatbot():
    # Get API key from environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found! Please create a .env file with your API key.")
    
    client = Groq(api_key=api_key)
    return client
