import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from chromadb import Client
from chromadb.config import Settings

# สร้างโฟลเดอร์เก็บข้อมูลถ้ายังไม่มี
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# โหลดโมเดล
print("🔄 กำลังโหลดโมเดล...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ โหลดโมเดลเสร็จสิ้น")

# เชื่อมต่อ ChromaDB
try:
    chroma_client = Client(
        Settings(persist_directory="./chroma_db", is_persistent=True)
    )
    faq_collection = chroma_client.get_or_create_collection("faq_test")
    print("✅ เชื่อมต่อ ChromaDB สำเร็จ")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ ChromaDB: {e}")
    exit(1)


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
            (0, 5): 0.75,    # เพิ่มค่าสำหรับคำสั้นๆ
            (6, 15): 0.80,   # เพิ่มค่าสำหรับคำปานกลาง
            (16, 1000): 0.85, # เพิ่มค่าสำหรับคำยาว
        }
    length = len(text.split())
    for (min_len, max_len), thr in thresholds.items():
        if min_len <= length <= max_len:
            return thr
    return 0.65


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

            print(f"📋 Matched embedding shape: {matched_emb.shape}")
            print(f"🎯 Query embedding shape: {query_emb.shape}")

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


if __name__ == "__main__":
    print("🚀 เริ่มต้นการทดสอบระบบ FAQ")

    # เพิ่มข้อมูล FAQ
    print("\n📝 กำลังเพิ่มข้อมูล FAQ...")
    faqs = [
        (
            "วิธีตัด Join Back Rise",
            "วิธีตัด Join Back Rise ตามคู่มือปฏิบัติการ\n1. ตัดส่วนหลังของกางเกง\n2. วัดขนาด\n3. ตัดตามเส้น\n4. ตรวจสอบความเรียบร้อย",
        ),
        (
            "RAG คืออะไร",
            "RAG คือ Retrieval Augmented Generation เทคนิครวมการค้นหาข้อมูลกับการสร้างคำตอบเพื่อ AI ตอบแม่นยำขึ้น",
        ),
        ("วิธีดูแลรักษากางเกงยีนส์", "ควรซักด้วยน้ำเย็น หลีกเลี่ยงการตากแดดแรง และใช้ผงซักฟอกอ่อนโยน"),
        (
            "How to care for jeans",
            "Wash with cold water, avoid direct sunlight, and use mild detergent",
        ),
        (
            "วิธีเลือกขนาดกางเกง",
            "วัดรอบเอว รอบสะโพก และความยาวขา แล้วเทียบกับตารางไซส์ของแบรนด์",
        ),
    ]

    for question, answer in faqs:
        add_faq(question, answer)

    # ทดสอบค้นหาคำถาม
    print("\n🧪 เริ่มทดสอบการค้นหาคำถาม...")
    test_questions = [
        "ตัด Join Back Rise",  # คล้ายกับคำถามแรก
        "RAG หมายถึงอะไร",  # คล้ายกับคำถามที่สอง
        "วิธีตัดส่วนหลังกางเกง",  # คล้ายกับคำถามแรก
        "How to care for jeans?",  # ภาษาอังกฤษ
        "วิธีดูแลกางเกงยีนส์",  # ภาษาไทย
        "ตั้งค่า Join Back Rise ยังไง",  # ใกล้เคียงแต่อาจไม่ค่อยเหมือน
        "เลือกไซส์กางเกงอย่างไร",  # คล้ายกับคำถามใหม่
        "Python คืออะไร",  # ไม่เกี่ยวข้อง ควรไม่เจอคำตอบ
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"🤔 คำถามที่ {i}: {q}")
        print("-" * 60)
        ans = find_similar_faq(q)
        if ans:
            print(f"💬 คำตอบ: {ans}")
        else:
            print("🔄 → Fallback ไปวิธีอื่น")
