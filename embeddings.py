import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json

# تحميل إعدادات الـ .env
load_dotenv()

# إعدادات الداتابيز من الـ .env
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "pool_size": int(os.getenv("DB_POOL_SIZE")),
}

# تحميل نموذج sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')  # نموذج خفيف وسريع

# دالة لتحويل نص إلى embedding
def get_embedding(text: str) -> list:
    try:
        embedding = model.encode(text, convert_to_tensor=False).tolist()
        return embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text}': {e}")
        return []

# دالة للاتصال بالداتابيز
def connect_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

# دالة للتحقق من وجود عمود
def column_exists(table_name: str, column_name: str) -> bool:
    conn = connect_db()
    if conn is None:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = %s AND COLUMN_NAME = %s AND TABLE_SCHEMA = %s
        """, (table_name, column_name, DB_CONFIG["database"]))
        exists = cursor.fetchone()[0] > 0
        return exists
    except Error as e:
        print(f"Error checking column {column_name}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# دالة لإضافة أعمدة جديدة للـ embeddings
def add_embedding_columns():
    conn = connect_db()
    if conn is None:
        return
    
    cursor = conn.cursor()
    try:
        # إضافة عمود name_embedding إذا مش موجود
        if not column_exists("PrintradoProducts", "name_embedding"):
            cursor.execute("""
                ALTER TABLE PrintradoProducts
                ADD name_embedding TEXT
            """)
            print("Added column: name_embedding")
        
        # إضافة عمود category_embedding إذا مش موجود
        if not column_exists("PrintradoProducts", "category_embedding"):
            cursor.execute("""
                ALTER TABLE PrintradoProducts
                ADD category_embedding TEXT
            """)
            print("Added column: category_embedding")
        
        conn.commit()
        print("Embedding columns checked/added successfully.")
    except Error as e:
        print(f"Error adding embedding columns: {e}")
    finally:
        cursor.close()
        conn.close()

# دالة لجلب البيانات وتحديث الـ embeddings
def process_products():
    conn = connect_db()
    if conn is None:
        return
    
    cursor = conn.cursor()
    try:
        # التأكد من وجود الأعمدة قبل التحديث
        if not (column_exists("PrintradoProducts", "name_embedding") and 
                column_exists("PrintradoProducts", "category_embedding")):
            print("Error: Embedding columns are missing. Please add them first.")
            return
        
        # جلب البيانات
        cursor.execute("SELECT product_id, product_name, product_categories FROM PrintradoProducts")
        products = cursor.fetchall()
        
        for product_id, product_name, categories in products:
            # تحويل النصوص إلى embeddings
            name_embedding = get_embedding(product_name)
            category_embedding = get_embedding(categories)
            
            # تحويل الـ embeddings إلى JSON string
            name_embedding_str = json.dumps(name_embedding) if name_embedding else None
            category_embedding_str = json.dumps(category_embedding) if category_embedding else None
            
            # تحديث الجدول
            cursor.execute("""
                UPDATE PrintradoProducts
                SET name_embedding = %s, category_embedding = %s
                WHERE product_id = %s
            """, (name_embedding_str, category_embedding_str, product_id))
            
            print(f"Processed product ID {product_id}: {product_name}")
        
        conn.commit()
    except Error as e:
        print(f"Error processing products: {e}")
    finally:
        cursor.close()
        conn.close()

# الجزء الرئيسي
def main():
    add_embedding_columns()
    process_products()
    print("Processing completed.")

if __name__ == "__main__":
    main()