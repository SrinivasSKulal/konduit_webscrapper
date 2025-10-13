# Usage:
#     python rag_fast_v2.py crawl 20 https://example.com
#     python rag_fast_v2.py index
#     python rag_fast_v2.py ask "What is this website about?"

import os
import sys
import time
import json
import requests
import tldextract
import numpy as np
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
CRAWL_FILE = "crawled.json"
INDEX_FILE = "index.faiss"
TEXTS_FILE = "texts.json"
URLS_FILE = "urls.json"


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------
def normalize_url(url):
    """Remove fragments and sort query parameters for deduplication"""
    parsed = urlparse(url)
    # Remove fragment
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",  # No fragment
        )
    )
    return normalized


def is_same_domain(base_url, new_url):
    """Check if two URLs belong to the same domain"""
    base = tldextract.extract(base_url)
    new = tldextract.extract(new_url)
    return base.domain == new.domain and base.suffix == new.suffix


def extract_text(html):
    """Extract clean text from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.extract()
    return " ".join(soup.stripped_strings)


# ----------------------------------------------------------
# 1. Crawl Website
# ----------------------------------------------------------
def crawl(start_url, max_pages, delay):
    """Crawl a website starting from start_url"""
    to_visit = [normalize_url(start_url)]
    visited = set()
    crawled = {}

    print(f"\n[+] Starting crawl from: {start_url}")
    print(f"[+] Max pages: {max_pages}\n")

    while to_visit and len(crawled) < max_pages:
        url = to_visit.pop(0)
        normalized = normalize_url(url)

        if normalized in visited:
            continue
        visited.add(normalized)

        try:
            resp = requests.get(
                url,
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
            )
            if resp.status_code != 200:
                print(f"[!] Skipped (status {resp.status_code}): {url}")
                continue

            text = extract_text(resp.text)
            if len(text) < 100:  # Skip pages with minimal content
                print(f"[!] Skipped (too short): {url}")
                continue

            crawled[normalized] = text
            print(f"[+] Crawled ({len(crawled)}/{max_pages}): {url[:80]}")

            # Find new links
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                full = urljoin(url, link["href"])
                if is_same_domain(start_url, full):
                    normalized_link = normalize_url(full)
                    if normalized_link not in visited:
                        to_visit.append(full)

            time.sleep(delay)

        except requests.RequestException as e:
            print(f"[!] Error crawling {url}: {e}")
        except Exception as e:
            print(f"[!] Unexpected error on {url}: {e}")

    with open(CRAWL_FILE, "w", encoding="utf-8") as f:
        json.dump(crawled, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Crawling complete — {len(crawled)} pages saved to {CRAWL_FILE}")
    return crawled


# ----------------------------------------------------------
# 2. Index and Embed
# ----------------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap

        # If we're at the end and have a small remaining piece, break
        if start >= text_len:
            break

    return chunks


def build_index(crawled_docs):
    """Build FAISS index from crawled documents"""
    print("\n[+] Loading embedding model (MiniLM-L3)...")
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    texts, urls = [], []
    print("\n[+] Chunking documents...")

    for url, doc in crawled_docs.items():
        chunks = chunk_text(doc)
        for chunk in chunks:
            texts.append(chunk)
            urls.append(url)

    print(f"[+] Created {len(texts)} text chunks from {len(crawled_docs)} pages")
    print(f"[+] Encoding {len(texts)} text chunks...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save everything
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2)

    print(f"\n✅ Index built successfully:")
    print(f"   - {len(texts)} chunks indexed")
    print(f"   - {dimension} dimensions")
    print(f"   - Files: {INDEX_FILE}, {TEXTS_FILE}, {URLS_FILE}")

    return index, texts, urls, model


# ----------------------------------------------------------
# 3. Ask Questions
# ----------------------------------------------------------
def load_index_and_model():
    """Load the FAISS index and embedding model"""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"Index file '{INDEX_FILE}' not found. Run 'index' command first."
        )
    if not os.path.exists(TEXTS_FILE):
        raise FileNotFoundError(
            f"Texts file '{TEXTS_FILE}' not found. Run 'index' command first."
        )
    if not os.path.exists(URLS_FILE):
        raise FileNotFoundError(
            f"URLs file '{URLS_FILE}' not found. Run 'index' command first."
        )

    print("\n[+] Loading index and embedding model...")
    index = faiss.read_index(INDEX_FILE)

    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    print(f"✅ Loaded index with {len(texts)} chunks")
    return index, texts, urls, model


def retrieve(query, index, model, texts, urls, top_k=5):
    """Retrieve top_k most relevant chunks for the query"""
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i in range(min(top_k, len(indices[0]))):
        idx = indices[0][i]
        results.append(
            {
                "url": urls[idx],
                "snippet": texts[idx][:500],  # First 500 chars
                "distance": float(distances[0][i]),
            }
        )
    return results


def generate_answer(query, retrieved, generator):
    """Generate an answer using the retrieved context"""
    if not retrieved:
        return "Not enough information found in crawled content."

    # Limit context to avoid exceeding token limits
    # Use top 3 results and truncate snippets to fit within 512 token limit
    context_parts = []
    for r in retrieved[:3]:
        context_parts.append(r["snippet"][:300])  # Shorter snippets

    context = "\n\n".join(context_parts)

    # Simpler, more direct prompt
    prompt = f"""Answer this question based on the context provided.

Context: {context}

Question: {query}

Provide a clear, detailed answer (2-3 sentences):"""

    response = generator(
        prompt,
        max_length=512,  # Total length including prompt
        max_new_tokens=120,
        do_sample=False,
        num_beams=4,
    )
    return response[0]["generated_text"]


def ask(query, index, model, texts, urls, generator):
    """Answer a question using RAG"""
    print(f"\n[?] Question: {query}\n")

    start_time = time.time()

    # Retrieval
    print("[+] Retrieving relevant chunks...")
    retrieved = retrieve(query, index, model, texts, urls, top_k=5)
    retrieval_ms = (time.time() - start_time) * 1000

    # Generation
    print("[+] Generating answer...")
    gen_start = time.time()
    answer = generate_answer(query, retrieved, generator)
    generation_ms = (time.time() - gen_start) * 1000

    total_ms = (time.time() - start_time) * 1000

    return {
        "answer": answer.strip(),
        "sources": [
            {"url": r["url"], "snippet": r["snippet"][:200]} for r in retrieved[:3]
        ],
        "timings": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }


# ----------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nRAG Fast v2 - Usage:")
        print("  python rag_fast_v2.py crawl <max_pages> <start_url> ")
        print("  python rag_fast_v2.py index")
        print("  python rag_fast_v2.py ask <question>")
        print("\nExample:")
        print("  python rag_fast_v2.py crawl https://example.com")
        print("  python rag_fast_v2.py index")
        print('  python rag_fast_v2.py ask "What is this website about?"')
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "crawl":
        if len(sys.argv) < 3:
            print("Error: URL required")
            print("Usage: python rag_fast_v2.py crawl <start_url>")
            sys.exit(1)
        start_url = sys.argv[3]
        max_pages = int(sys.argv[2])

        crawl(start_url, max_pages, 0.5)

    elif command == "index":
        if not os.path.exists(CRAWL_FILE):
            print(f"Error: {CRAWL_FILE} not found. Run 'crawl' command first.")
            sys.exit(1)

        with open(CRAWL_FILE, "r", encoding="utf-8") as f:
            crawled_docs = json.load(f)

        if not crawled_docs:
            print("Error: No documents found in crawled data.")
            sys.exit(1)

        build_index(crawled_docs)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Error: Question required")
            print('Usage: python rag_fast_v2.py ask "your question here"')
            sys.exit(1)

        query = " ".join(sys.argv[2:])

        try:
            # Load index and embedding model
            index, texts, urls, model = load_index_and_model()

            # Load generator only when needed
            print("[+] Loading FLAN-T5-small model for answer generation...")
            generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,  # CPU only
            )
            print("✅ Model loaded\n")

            # Ask question
            result = ask(query, index, model, texts, urls, generator)

            # Print results
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(result["answer"])
            print("\n" + "=" * 60)
            print("SOURCES:")
            print("=" * 60)
            for i, src in enumerate(result["sources"], 1):
                print(f"\n{i}. {src['url']}")
                print(f"   {src['snippet'][:150]}...")

            print("\n" + "=" * 60)
            print("TIMINGS:")
            print("=" * 60)
            print(f"Retrieval: {result['timings']['retrieval_ms']} ms")
            print(f"Generation: {result['timings']['generation_ms']} ms")
            print(f"Total: {result['timings']['total_ms']} ms")
            print("=" * 60 + "\n")

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during question answering: {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: '{command}'")
        print("Valid commands: crawl, index, ask")
        sys.exit(1)
