
import os
import fitz

# Find the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF file location
PDF_PATH = os.path.join(
    BASE_DIR,
    "documents",
    "syllabus.pdf"
)


# STEP 1: Chunking function
def split_into_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward, keeping some text from previous chunk
        start += chunk_size - overlap

    return chunks


# STEP 2: Read PDF and create chunks
def test_pdf():

    print("\n========== PDF TEST STARTED ==========")

    print("Project folder:", BASE_DIR)
    print("PDF path:", PDF_PATH)
    print("PDF exists:", os.path.isfile(PDF_PATH))

    if not os.path.isfile(PDF_PATH):

        print("\nERROR: syllabus.pdf was not found.")

        print("Check your folder structure:")
        print("documents/syllabus.pdf")

        return

    try:

        # Open PDF
        pdf = fitz.open(PDF_PATH)

        print("\nPDF opened successfully!")
        print("Total pages:", len(pdf))

        total_chunks = 0
        total_characters = 0

        # Read each page
        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text").strip()

            print(f"\nReading Page {page_number}...")

            if not page_text:

                print("No selectable text found on this page.")
                print("This page may need OCR.")

                continue

            total_characters += len(page_text)

            # Split this page into smaller chunks
            chunks = split_into_chunks(
                page_text,
                chunk_size=500,
                overlap=100
            )

            # Display each chunk
            for chunk_number, chunk in enumerate(chunks, start=1):

                total_chunks += 1

                print(
                    f"\n========== PAGE {page_number} "
                    f"| CHUNK {chunk_number} =========="
                )

                print(chunk)

        pdf.close()

        # Final summary
        print("\n========== FINAL SUMMARY ==========")

        print("Total PDF pages:", len(pdf))
        print("Total extracted characters:", total_characters)
        print("Total chunks created:", total_chunks)

        if total_chunks > 0:

            print("\nSUCCESS: PDF reading and chunking completed!")

        else:

            print("\nWARNING: No chunks were created.")

    except Exception as error:

        print("\nPDF READING ERROR:", str(error))

    print("\n========== PDF TEST FINISHED ==========")


# STEP 3: Run the program
if __name__ == "__main__":

    test_pdf()
