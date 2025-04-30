import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import mysql.connector
from mysql.connector import Error
import json
import numpy as np
import faiss
import time
from tqdm import tqdm  # لعرض شريط التقدم
import psutil  # لتتبع استهلاك الذاكرة
import requests  # للاتصال بـ Gemini API

# إعداد Logging لتسجيل الأخطاء
logging.basicConfig(
    filename='api.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# تحميل إعدادات الـ .env
load_dotenv()

# إعدادات الداتابيز
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", 5)),
}

# إعدادات Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# مسارات ملفات التخزين
FAISS_INDEX_PATH = "faiss_index.bin"
PRODUCT_INFO_PATH = "product_info.json"

# إنشاء تطبيق FastAPI
app = FastAPI(title="Book Search API with RAG and Chat")

# تحميل نموذج sentence-transformers
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIM = 384  # أبعاد الـ embedding لنموذج all-MiniLM-L6-v2
except Exception as e:
    logger.error(f"Error loading sentence-transformers model: {str(e)}")
    raise

# نموذج Pydantic للـ input
class QueryRequest(BaseModel):
    query: str

# دالة للاتصال بالداتابيز
def connect_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        logger.info("Successfully connected to database")
        return conn
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

# دالة لتحويل نص إلى embedding
def get_embedding(text: str) -> list:
    try:
        embedding = model.encode(text, convert_to_tensor=False).tolist()
        logger.debug(f"Generated embedding for text: {text}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# دالة للحصول على رد من Gemini
def get_gemini_response(query: str, is_chat=True) -> str:
    logger.info("Requesting response from Gemini API")
    try:
        # إنشاء الـ prompt
        if is_chat:
            prompt = (
                f"You are a friendly chatbot assisting users with book recommendations and general conversation. "
                f"The user said: '{query}'. If this is a general greeting or conversational message (e.g., 'Good morning'), "
                f"respond with a friendly, natural reply. If it seems like a request for a book, respond with 'BOOK_REQUEST' to trigger a search. "
                f"Examples: 'Good morning' -> 'Good morning! Ready to find a great book today?' | 'I want a book about coding' -> 'BOOK_REQUEST'"
            )
        else:
            prompt = query  # لتوصيات الكتب، نستخدم الـ prompt مباشرة
        
        # إعداد الطلب لـ Gemini API
        url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        # إرسال الطلب
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # معالجة الاستجابة
        result = response.json()
        if "candidates" not in result or not result["candidates"]:
            logger.error("No valid response from Gemini API")
            raise HTTPException(status_code=500, detail="No valid response from Gemini API")
        
        response_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info("Received response from Gemini API")
        return response_text
    
    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

# دالة للحصول على توصيات من Gemini
def get_gemini_recommendations(query: str, books: list) -> dict:
    logger.info("Requesting recommendations from Gemini API")
    try:
        # صيغة البيانات لكل كتاب
        books_info = [
            {
                "product_id": book["product_id"],
                "product_name": book["product_name"],
                "categories": book["categories"],
                "total_items_sold": book["total_items_sold"]
            }
            for book in books
        ]
        
        # إنشاء الـ prompt لـ Gemini
        prompt = (
            f"Based on the user's query '{query}', provide a short recommendation (1-2 sentences) for each of the following books. "
            f"Each recommendation should explain why the book is suitable for the query, considering its title, categories, and popularity (total items sold). "
            f"Return the recommendations as a JSON object where each key is the product_id and the value is the recommendation text. "
            f"Books: {json.dumps(books_info, indent=2)}"
        )
        
        # استدعاء Gemini
        recommendations_text = get_gemini_response(prompt, is_chat=False)
        # نزيل أي علامات تنسيق (مثل ```json) لو موجودة
        recommendations_text = recommendations_text.strip().strip("```json").strip("```")
        recommendations = json.loads(recommendations_text)
        logger.info("Received recommendations from Gemini API")
        return recommendations
    
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing Gemini response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing Gemini response: {str(e)}")

def build_faiss_index_from_db():
    logger.info("Building FAISS index from database")
    conn = connect_db()
    if conn is None:
        logger.error("Failed to connect to database")
        raise HTTPException(status_code=500, detail="Failed to connect to database")
    
    cursor = conn.cursor()
    try:
        # جلب البيانات مع الـ embeddings وعدد المبيعات
        cursor.execute("""
            SELECT product_id, product_name, product_categories, total_items_sold, 
                   product_url, image_url, name_embedding, category_embedding
            FROM PrintradoProducts
            WHERE name_embedding IS NOT NULL OR category_embedding IS NOT NULL
            LIMIT 1000  # شيلي الـ LIMIT لو عايزة كل البيانات
        """)
        products = cursor.fetchall()
        logger.info(f"Fetched {len(products)} products from database")
        
        # إعداد البيانات لـ FAISS
        embeddings = []
        product_info = []
        for product_id, product_name, categories, total_items_sold, product_url, image_url, name_embedding_str, category_embedding_str in products:
            try:
                name_embedding = json.loads(name_embedding_str) if name_embedding_str else None
                category_embedding = json.loads(category_embedding_str) if category_embedding_str else None
                
                # التحقق من حجم الـ embedding
                if name_embedding:
                    if len(name_embedding) != EMBEDDING_DIM:
                        logger.warning(f"Invalid name_embedding size for product_id {product_id}: {len(name_embedding)}")
                        continue
                    embeddings.append(name_embedding)
                    product_info.append({
                        "id": product_id,
                        "name": product_name,
                        "categories": categories,
                        "total_items_sold": total_items_sold,
                        "product_url": product_url,
                        "image_url": image_url,
                        "type": "name"
                    })
                
                # تعليق: لو عايزة تضيفي category_embedding، الكود تحت ده هو اللي بيضيفها
                if category_embedding:
                    if len(category_embedding) != EMBEDDING_DIM:
                        logger.warning(f"Invalid category_embedding size for product_id {product_id}: {len(category_embedding)}")
                        continue
                    embeddings.append(category_embedding)
                    product_info.append({
                        "id": product_id,
                        "name": product_name,
                        "categories": categories,
                        "total_items_sold": total_items_sold,
                        "product_url": product_url,
                        "image_url": image_url,
                        "type": "category"
                    })
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for product_id {product_id}: {str(e)}")
                continue
        
        if not embeddings:
            logger.error("No valid embeddings found in database")
            raise HTTPException(status_code=404, detail="No valid embeddings found in database")
        
        # فحص الـ embeddings لو فيها NaN أو inf
        logger.info(f"Validating {len(embeddings)} embeddings")
        valid_embeddings = []
        valid_product_info = []
        for emb, info in zip(embeddings, product_info):
            if not np.any(np.isnan(emb)) and not np.any(np.isinf(emb)):
                valid_embeddings.append(emb)
                valid_product_info.append(info)
            else:
                logger.warning(f"Invalid embedding for product_id {info['id']}: contains NaN or inf")
        
        if not valid_embeddings:
            logger.error("No valid embeddings after filtering")
            raise HTTPException(status_code=404, detail="No valid embeddings after filtering")
        
        embeddings = valid_embeddings
        product_info = valid_product_info
        logger.info(f"Validated {len(embeddings)} embeddings")
        
        # إنشاء فهرس FAISS
        try:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)  # فهرس أخف من HNSW
            logger.info("Created FAISS FlatL2 index")
            
            # معالجة الـ embeddings على دفعات مع progress bar وتتبع مفصل
            batch_size = 200  # حجم دفعة أصغر
            total_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size else 0)
            batch_times = []
            
            for i in tqdm(range(0, len(embeddings), batch_size), total=total_batches, desc="Adding embeddings to FAISS"):
                start_time = time.time()
                batch = embeddings[i:i + batch_size]
                batch_len = len(batch)
                try:
                    # تتبع استهلاك الذاكرة
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024  # تحويل إلى ميجابايت
                    
                    batch_np = np.array(batch, dtype=np.float32)
                    faiss.normalize_L2(batch_np)
                    index.add(batch_np)
                    
                    elapsed_time = time.time() - start_time
                    batch_times.append(elapsed_time)
                    
                    logger.info(
                        f"Added batch {i // batch_size + 1}/{total_batches}: "
                        f"{batch_len} embeddings, "
                        f"time={elapsed_time:.2f}s, "
                        f"memory={memory_mb:.2f}MB"
                    )
                    if elapsed_time > 30:  # تحذير لو الدفعة أخدت وقت طويل
                        logger.warning(f"Batch {i // batch_size + 1} took too long: {elapsed_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error adding batch {i // batch_size + 1}: {str(e)}")
            
            # تلخيص عملية التحميل
            total_time = sum(batch_times)
            avg_time_per_batch = total_time / len(batch_times) if batch_times else 0
            logger.info(
                f"Batch loading summary: "
                f"Total embeddings={len(embeddings)}, "
                f"Total batches={total_batches}, "
                f"Total time={total_time:.2f}s, "
                f"Average time per batch={avg_time_per_batch:.2f}s"
            )
            logger.info("FAISS index built successfully")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error building FAISS index: {str(e)}")
        
        # حفظ الفهرس على القرص
        try:
            faiss.write_index(index, FAISS_INDEX_PATH)
            logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving FAISS index: {str(e)}")
        
        # حفظ product_info في ملف JSON
        try:
            with open(PRODUCT_INFO_PATH, 'w') as f:
                json.dump(product_info, f)
            logger.info(f"Saved product info to {PRODUCT_INFO_PATH}")
        except Exception as e:
            logger.error(f"Error saving product info: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving product info: {str(e)}")
        
        return index, product_info
    
    except Error as e:
        logger.error(f"Error fetching data from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching data from database: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        logger.info("Database connection closed")

# دالة لتحميل أو بناء فهرس FAISS
def load_or_build_faiss_index():
    logger.info("Checking for existing FAISS index")
    # التحقق من وجود ملف الفهرس
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(PRODUCT_INFO_PATH):
        try:
            # تحميل الفهرس
            index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
            
            # تحميل product_info
            with open(PRODUCT_INFO_PATH, 'r') as f:
                product_info = json.load(f)
            logger.info(f"Loaded product info from {PRODUCT_INFO_PATH}")
            
            return index, product_info
        except Exception as e:
            logger.error(f"Error loading FAISS index or product info: {e}")
            # لو فشل التحميل، بني الفهرس من الداتابيز
    
    # بناء الفهرس من الداتابيز لو مش موجود أو فيه مشكلة
    return build_faiss_index_from_db()

# دالة للبحث الدلالي مع ترتيب حسب المبيعات
def semantic_search(query_embedding: list, index, product_info, top_k: int = 3, initial_k: int = 10):
    logger.info("Performing semantic search")
    try:
        # تحويل الاستعلام إلى numpy array
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # البحث في FAISS
        distances, indices = index.search(query_embedding, initial_k)
        logger.debug(f"FAISS search returned {len(indices[0])} results")
        
        # تجميع النتايج
        results = []
        seen_ids = set()  # لتجنب تكرار الكتب
        for dist, idx in zip(distances[0], indices[0]):
            product = product_info[idx]
            if product["id"] not in seen_ids:
                seen_ids.add(product["id"])
                results.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "categories": product["categories"],
                    "total_items_sold": product["total_items_sold"],
                    "product_url": product["product_url"],
                    "image_url": product["image_url"],
                    "similarity": float(dist)
                })
        
        # ترتيب النتايج حسب total_items_sold
        results = sorted(results, key=lambda x: x["total_items_sold"], reverse=True)[:top_k]
        logger.info(f"Returning {len(results)} search results")
        
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in semantic search: {str(e)}")

# Endpoint لتحويل الاستعلام إلى vector
@app.post("/query-to-vector")
async def query_to_vector(request: QueryRequest):
    logger.info(f"Received query-to-vector request: {request.query}")
    if not request.query.strip():
        logger.warning("Query is empty")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    embedding = get_embedding(request.query)
    
    return {
        "query": request.query,
        "embedding": embedding,
        "embedding_length": len(embedding)
    }

# Endpoint للبحث الدلالي مع RAG
@app.post("/search")
async def search_books(request: QueryRequest):
    logger.info(f"Received search request: {request.query}")
    if not request.query.strip():
        logger.warning("Query is empty")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # تحويل الاستعلام إلى embedding
    query_embedding = get_embedding(request.query)
    
    # تحميل أو بناء فهرس FAISS
    index, product_info = load_or_build_faiss_index()
    
    # إجراء البحث الدلالي
    results = semantic_search(query_embedding, index, product_info, top_k=3, initial_k=10)
    
    # الحصول على توصيات من Gemini
    recommendations = {}
    if results:
        try:
            recommendations = get_gemini_recommendations(request.query, results)
        except HTTPException as e:
            logger.warning(f"Failed to get Gemini recommendations: {str(e)}")
            recommendations = {book["product_id"]: "Recommendation unavailable" for book in results}
    
    # إضافة التوصيات للنتايج
    for book in results:
        book["recommendation"] = recommendations.get(str(book["product_id"]), "Recommendation unavailable")
    
    return {
        "query": request.query,
        "results": results
    }

# Endpoint للدردشة
@app.post("/chat")
async def chat(request: QueryRequest):
    logger.info(f"Received chat request: {request.query}")
    if not request.query.strip():
        logger.warning("Query is empty")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # تحليل السؤال باستخدام Gemini
    try:
        intent = get_gemini_response(request.query)
        logger.debug(f"Gemini intent: {intent}")
        
        if intent == "BOOK_REQUEST":
            # إذا كان طلب كتاب، استدعي البحث
            search_response = await search_books(request)
            results = search_response["results"]
            
            # صيغ الرد بطريقة ودودة
            if results:
                response_text = "Here are some books I found for you:\n\n"
                for book in results:
                    response_text += (
                        f"**{book['product_name']}** (Categories: {book['categories']})\n"
                        f"{book['recommendation']}\n"
                        f"Check it out: {book['product_url']}\n\n"
                    )
            else:
                response_text = "Sorry, I couldn't find any books matching your query. Try something else?"
        else:
            # رد عام للأسئلة البديهية
            response_text = intent
        
        return {
            "query": request.query,
            "response": response_text
        }
    
    except HTTPException as e:
        logger.error(f"Error in chat processing: {str(e)}")
        raise
from fastapi.middleware.cors import CORSMiddleware

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5500"],  # أضيفي 5500
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# تشغيل الـ API
if __name__ == "__main__":
    import uvicorn
    try:
        logger.info("Starting Uvicorn server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error starting Uvicorn server: {str(e)}")
        raise