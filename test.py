import os
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Ensure the chroma_db folder exists
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")
    print("Created chroma_db folder.")

# Initialize ChromaDB client
chroma_client = Client(
    Settings(
        persist_directory="./chroma_db",  # Directory to store the database
        anonymized_telemetry=False,
        is_persistent=True,
    )
)

# Get or create a collection
collection = chroma_client.get_or_create_collection(name="Students")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add sample data to the collection
collection.add(
    ids=["1", "2"],
    metadatas=[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}],
    embeddings=[
        model.encode("Alice is 25 years old."),
        model.encode("Bob is 30 years old."),
    ],
)

print("Data added successfully!")
