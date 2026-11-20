
import re
import math
from collections import Counter


class RAGEngine:
    """
    Simple RAG engine for NeuroBot.

    Supports:
    - Document text processing
    - Text chunking
    - Keyword-based retrieval
    - Relevance scoring
    """

    def __init__(self, chunk_size=700, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text):
        """Clean unnecessary spaces and line breaks."""
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_into_chunks(self, text):
        """Split document text into overlapping chunks."""
        text = self.clean_text(text)

        if not text:
            return []

        words = text.split()
        chunks = []

        step = max(1, self.chunk_size - self.overlap)

        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]

            if not chunk_words:
                continue

            chunks.append(" ".join(chunk_words))

            if start + self.chunk_size >= len(words):
                break

        return chunks

    def tokenize(self, text):
        """Convert text into searchable words."""
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def build_index(self, text, filename="uploaded_document"):
        """
        Build a searchable document index.

        Returns an index dictionary used by retrieve().
        """
        chunks = self.split_into_chunks(text)

        indexed_chunks = []

        for chunk_number, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk)

            indexed_chunks.append({
                "chunk_id": chunk_number,
                "text": chunk,
                "tokens": tokens
            })

        return {
            "filename": filename,
            "chunks": indexed_chunks,
            "total_chunks": len(indexed_chunks)
        }

    def calculate_score(self, query_tokens, chunk_tokens):
        """Calculate simple keyword relevance score."""
        if not query_tokens or not chunk_tokens:
            return 0

        query_counts = Counter(query_tokens)
        chunk_counts = Counter(chunk_tokens)

        score = 0

        for word, query_count in query_counts.items():
            if word in chunk_counts:
                score += min(query_count, chunk_counts[word])

        return score / math.sqrt(len(chunk_tokens))

    def retrieve(self, index, query, top_k=4):
        """
        Find the most relevant document chunks for a query.
        """
        if not index or not query:
            return []

        chunks = index.get("chunks", [])

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        results = []

        for chunk in chunks:
            score = self.calculate_score(
                query_tokens,
                chunk.get("tokens", [])
            )

            if score > 0:
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": score
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]


def format_rag_context(hits, filename="uploaded_document"):
    """
    Format retrieved document chunks for the LLM prompt.
    """
    if not hits:
        return ""

    context_parts = []

    for number, hit in enumerate(hits, start=1):
        text = hit.get("text", "")

        if text:
            context_parts.append(
                f"Relevant Information {number}:\n{text}"
            )

    if not context_parts:
        return ""

    return (
        f"DOCUMENT SOURCE: {filename}\n\n"
        "Use the following retrieved information to answer "
        "the user's question.\n"
        "Treat this information as reference material. "
        "If the answer is not available in the document, "
        "clearly say that it is not specified.\n\n"
        + "\n\n".join(context_parts)
    )
