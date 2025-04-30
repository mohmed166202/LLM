from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# تحميل إعدادات الـ .env
load_dotenv()

# إنشاء تطبيق FastAPI
app = FastAPI(title="Query to Vector API")

# تحميل نموذج sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# نموذج Pydantic للـ input
class QueryRequest(BaseModel):
    query: str

# دالة لتحويل نص إلى embedding
def get_embedding(text: str) -> list:
    try:
        embedding = model.encode(text, convert_to_tensor=False).tolist()
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Endpoint لتحويل الاستعلام إلى vector
@app.post("/query-to-vector")
async def query_to_vector(request: QueryRequest):
    # التحقق من إن الاستعلام مش فاضي
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # تحويل الاستعلام إلى embedding
    embedding = get_embedding(request.query)
    
    # إرجاع النتيجة كـ JSON
    return {
        "query": request.query,
        "embedding": embedding,
        "embedding_length": len(embedding)
    }

# تشغيل الـ API (للاختبار محليًا)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)