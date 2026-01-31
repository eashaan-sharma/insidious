import sqlite3
from datetime import datetime
import os

# Absolute path to database file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "inference.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inference_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        risk_flag TEXT NOT NULL,
        image_path TEXT,
        heatmap_path TEXT
    );
    """)

    conn.commit()
    conn.close()


def insert_inference(prediction, confidence, risk_flag, image_path, heatmap_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO inference_logs (
        timestamp,
        prediction,
        confidence,
        risk_flag,
        image_path,
        heatmap_path
    ) VALUES (?, ?, ?, ?, ?, ?);
    """, (
        datetime.now().isoformat(),
        prediction,
        confidence,
        risk_flag,
        image_path,
        heatmap_path
    ))

    conn.commit()
    conn.close()


# Run once for setup/testing
if __name__ == "__main__":
    create_table()

    insert_inference(
        prediction="Pneumonia",
        confidence=0.87,
        risk_flag="HIGH",
        image_path="data/sample_xray.jpg",
        heatmap_path="data/sample_heatmap.png"
    )

    print("✅ SQLite database and inference_logs table created.")
