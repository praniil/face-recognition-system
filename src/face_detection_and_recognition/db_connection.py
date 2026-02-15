import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
def db_connection():
    print("in db connection")
    connection = None

    try:
        connection = psycopg2.connect(
            host = os.getenv("DB_HOST"),
            port = os.getenv("DB_PORT"),
            user = os.getenv("DB_USERNAME"),
            password = os.getenv("DB_PASSWORD"),
            dbname = os.getenv("DB_DATABASE"),
        )
        print("connection established")
        return connection

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None
