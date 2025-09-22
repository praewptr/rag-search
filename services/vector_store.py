from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
from typing import Optional
from services.client import chroma_client
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Create or get the collection for FAQs
faq_collection = chroma_client.get_or_create_collection("faq")


def add_faq(question, answer):
    try:
        # เช็คว่าคำถามนี้มีอยู่แล้วหรือไม่
        existing = faq_collection.get(ids=[question])
        if existing["ids"]:
            print(f"⚠️  คำถามนี้มีอยู่แล้ว: {question}")
            return False

        embedding = model.encode(question, convert_to_numpy=True)
        faq_collection.add(
            ids=[question],
            metadatas=[{"question": question, "answer": answer}],
            embeddings=[embedding.tolist()],
        )
        print(f"✅ เพิ่ม FAQ สำเร็จ: {question}")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเพิ่ม FAQ: {e}")
        return False


def get_threshold_by_length(text, thresholds=None):
    if thresholds is None:
        thresholds = {
            (0, 5): 0.95,  # เข้มงวดมากสำหรับคำสั้นๆ
            (6, 15): 0.90,  # เข้มงวดมากสำหรับคำปานกลาง
            (16, 1000): 0.85,  # เข้มงวดมากสำหรับคำยาว
        }
    length = len(text.split())
    for (min_len, max_len), thr in thresholds.items():
        if min_len <= length <= max_len:
            return thr
    return 0.92


def find_similar_faq(user_question):
    try:
        # เข้ารหัสคำถามผู้ใช้
        query_emb = model.encode(user_question, convert_to_numpy=True)
        print(f"🔍 Query embedding shape: {query_emb.shape}")

        results = faq_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=1,
            include=["embeddings", "metadatas"],
        )

        if results and results["ids"] and len(results["ids"][0]) > 0:
            matched_question = results["metadatas"][0][0]["question"]
            answer = results["metadatas"][0][0]["answer"]
            matched_emb = np.array(results["embeddings"][0][0], dtype=np.float32)

            similarity = util.cos_sim(query_emb, matched_emb).item()

            thr_user = get_threshold_by_length(user_question)
            thr_db = get_threshold_by_length(matched_question)
            threshold = (thr_user + thr_db) / 2

            print(f"📊 Similarity: {similarity:.4f}, Threshold: {threshold:.4f}")
            print(f"🔍 Matched question: {matched_question}")

            # เช็คว่า similarity อยู่ในช่วงที่ถูกต้อง
            if similarity < -1 or similarity > 1:
                print(f"⚠️  Similarity ผิดปกติ: {similarity}")
                return None

            if similarity >= threshold:
                print("✅ พบคำตอบที่เหมาะสม")
                return answer
            else:
                print(
                    f"❌ Similarity ({similarity:.4f}) ต่ำกว่าค่า threshold ({threshold:.4f})"
                )
                return None
        else:
            print("❌ ไม่มีคำถามในฐานข้อมูล")
            return None
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}")
        return None
