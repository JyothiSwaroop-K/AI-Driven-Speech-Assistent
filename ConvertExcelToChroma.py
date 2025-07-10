import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

# Load Excel file
EXCEL_FILE = "questions.xlsx"
COLLECTION_NAME = "python_questions"
DB_DIR = "python_quiz_db"

if not os.path.exists(EXCEL_FILE):
    raise FileNotFoundError(f"❌ Excel file '{EXCEL_FILE}' not found.")

# Set up embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize persistent Chroma client
client = chromadb.PersistentClient(path=DB_DIR)

# Check if collection already exists
existing_collections = [col.name for col in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists. Loading existing collection.")
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
else:
    print(f"✅ Creating new collection '{COLLECTION_NAME}'.")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

# Load and clean data
df = pd.read_excel(EXCEL_FILE)

required_columns = {"Question", "Answer", "Difficulty", "ID"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"❌ Excel must contain columns: {required_columns}")

df = df.dropna(subset=["Question", "Answer", "Difficulty", "ID"])
df = df.astype(str)  # Ensure all are strings

# Prepare documents and metadata
documents = df["Answer"].tolist()
metadatas = df[["Difficulty"]].to_dict(orient="records")
ids = df["ID"].tolist()

# Add to collection (check if IDs already exist is optional)
print(f"📥 Adding {len(documents)} items to Chroma collection...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("✅ Data successfully added to Chroma collection.")

