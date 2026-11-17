
import os
import fitz
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "documents",
    "syllabus.pdf"
)


def split_into_chunks(text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings():

    print("\n========== EMBEDDING TEST STARTED ==========")

    print("PDF path:", PDF_PATH)
    print("PDF exists:", os.path.isfile(PDF_PATH))

    if not os.path.isfile(PDF_PATH):
        print("ERROR: syllabus.pdf not found!")
        return None

    try:

        pdf = fitz.open(PDF_PATH)

        all_chunks = []

        print("PDF opened successfully!")
        print("Total pages:", len(pdf))

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            chunks = split_into_chunks(page_text)

            for chunk_number, chunk in enumerate(chunks, start=1):

                all_chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number
                })

        pdf.close()

        print("Total chunks:", len(all_chunks))

        if not all_chunks:
            print("No text chunks found!")
            return None

        print("\nLoading embedding model...")

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded!")

        texts = [item["text"] for item in all_chunks]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print("\n========== RESULTS ==========")

        print("Total chunks:", len(texts))
        print("Embedding shape:", embeddings.shape)

        for index, item in enumerate(all_chunks):

            print("\nPage:", item["page"])
            print("Chunk:", item["chunk"])
            print("Vector dimensions:", len(embeddings[index]))
            print("First 5 values:", embeddings[index][:5])

        print("\nSUCCESS: EMBEDDINGS CREATED!")

        return embeddings

    except Exception as error:

        print("EMBEDDING ERROR:", str(error))
        return None


if __name__ == "__main__":
    create_embeddings()
