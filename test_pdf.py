
import fitz

# Open PDF file
pdf = fitz.open("documents/syllabus.pdf")

# Read first page
page = pdf[0]

# Extract text
text = page.get_text("text")

# Display extracted text
print(text)

# Close PDF
pdf.close()
