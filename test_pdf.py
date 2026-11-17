
import os
import fitz
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PDF LOCATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "documents",
    "syllabus.pdf"
)


# ==========================================
# 2. CHUNKING FUNCTION
# ==========================================

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


# ==========================================
# 3. READ PDF AND CREATE CHUNKS
# ==========================================

def create_embeddings():

    print("\n======================================")
    print("NEUROBOT EMBEDDING TEST STARTED")
    print("======================================")

    print("PDF location:", PDF_PATH)

    if not os.path.isfile(PDF_PATH):

        print("ERROR: PDF file not found.")
        print("Check documents/syllabus.pdf")

        return

    try:

        pdf = fitz.open(PDF_PATH)

        all_chunks = []

        print("PDF opened successfully!")
        print("Total pages:", len(pdf))

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text").strip()

            if not page_text:

                print("No selectable text on page:", page_number)
                continue

            chunks = split_into_chunks(
                page_text,
                chunk_size=500,
                overlap=100
            )

            for chunk_number, chunk in enumerate(chunks, start=1):

                all_chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number
                })

        pdf.close()

        print("\nTotal chunks created:", len(all_chunks))

        if not all_chunks:

            print("ERROR: No text chunks found.")
            return


        # ==================================
        # 4. LOAD EMBEDDING MODEL
        # ==================================

        print("\nLoading embedding model...")

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully!")


        # ==================================
        # 5. CONVERT TEXT INTO EMBEDDINGS
        # ==================================

        texts = [
            item["text"]
            for item in all_chunks
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )


        # ==================================
        # 6. DISPLAY EMBEDDING RESULTS
        # ==================================

        print("\n======================================")
        print("EMBEDDING RESULTS")
        print("======================================")

        print("Total chunks:", len(texts))
        print("Embedding shape:", embeddings.shape)

        for index, item in enumerate(all_chunks):

            print("\nPage:", item["page"])
            print("Chunk:", item["chunk"])

            print(
                "Vector dimensions:",
                len(embeddings[index])
            )

            print(
                "First 5 vector values:",
                embeddings[index][:5]
            )


        # ==================================
        # 7. SUCCESS MESSAGE
        # ==================================

        print("\n======================================")
        print("SUCCESS: EMBEDDINGS CREATED!")
        print("======================================")

        return embeddings

    except Exception as error:

        print("\nEMBEDDING ERROR:", str(error))

        return None


if __name__ == "__main__":

    create_embeddings()
