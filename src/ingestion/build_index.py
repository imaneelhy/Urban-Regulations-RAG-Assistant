from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import DATA_RAW_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME
from src.rag.embeddings import SBERTEmbeddings


def load_documents():
    pdf_files = list(DATA_RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in: {DATA_RAW_DIR}")
        print("   → Put your zoning / urban regulations PDFs in that folder.")
        return []

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source_file"] = pdf_path.name
        docs.extend(pdf_docs)
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    print(f"📄 Loading documents from: {DATA_RAW_DIR}")
    docs = load_documents()

    if not docs:
        print("⛔ No documents, cannot build index.")
        return

    print(f"✔ Loaded {len(docs)} pages")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"✔ Created {len(chunks)} chunks")

    print("🧠 Building Chroma index...")
    embedding = SBERTEmbeddings(EMBEDDING_MODEL_NAME)

    # Use from_documents to preserve metadata
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
    )
    db.persist()
    print(f"✅ Index built & saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
