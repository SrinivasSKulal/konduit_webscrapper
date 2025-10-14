
# Konduit Web Scrapper and summarize

A fast, CPU-only Retrieval-Augmented Generation (RAG) system that crawls websites, builds a searchable knowledge base, and answers questions using the crawled content.

## Features

- 🕷️ **Smart Web Crawler** - Crawls up to 15 pages within a domain with URL deduplication
- 🔍 **Vector Search** - Uses FAISS for fast semantic similarity search
- 🤖 **AI-Powered Q&A** - Generates detailed answers using FLAN-T5
- 💻 **CPU-Only** - No GPU required, runs on any machine
- ⚡ **Optimized** - Models loaded only when needed for efficiency

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu transformers tldextract
```
or
```bash
pip install -r requirements.txt
```


## Usage

The system works in three simple steps:

### 1. Crawl a Website

Crawl a website and extract text content:

```bash
python main.py crawl 20 https://example.com
```

replace 20 with required number
**Options:**
- Automatically deduplicates URLs
- Skips pages with minimal content
- Saves results to `crawled.json`

**Example output:**
```
[+] Starting crawl from: https://example.com
[+] Max pages: 15

[+] Crawled (1/15): https://example.com
[+] Crawled (2/15): https://example.com/about
...
✅ Crawling complete — 12 pages saved to crawled.json
```

### 2. Build the Index

Create a vector index from the crawled content:

```bash
python main.py index
```

**What it does:**
- Splits documents into overlapping chunks (800 chars with 100 char overlap)
- Generates embeddings using `paraphrase-MiniLM-L3-v2`
- Builds a FAISS vector index for fast similarity search
- Saves index files: `index.faiss`, `texts.json`, `urls.json`

**Example output:**
```
[+] Loading embedding model (MiniLM-L3)...
[+] Created 48 text chunks from 12 pages
[+] Encoding 48 text chunks...
✅ Index built successfully:
   - 48 chunks indexed
   - 384 dimensions
```

### 3. Ask Questions

Query your knowledge base:

```bash
python main.py ask "What is this website about?"
```

**Example output:**
```
[?] Question: What is this website about?

[+] Retrieving relevant chunks...
[+] Generating answer...

============================================================
ANSWER:
============================================================
Example.com is an illustrative domain used in documentation
and examples. It serves as a placeholder for demonstrating
web concepts and is maintained by IANA for this purpose.

============================================================
SOURCES:
============================================================
1. https://example.com
   Example Domain This domain is for use in illustrative...

============================================================
TIMINGS:
============================================================
Retrieval: 145.23 ms
Generation: 2341.56 ms
Total: 2486.79 ms
============================================================
```

## How It Works

### Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Crawler   │ ───> │   Indexer    │ ───> │  Q&A Engine │
└─────────────┘      └──────────────┘      └─────────────┘
      │                     │                      │
      ▼                     ▼                      ▼
 crawled.json          index.faiss           Your Answer
                       texts.json
                       urls.json
```

### Components

1. **Web Crawler**
   - Breadth-first crawling within domain
   - URL normalization and deduplication
   - Content extraction from HTML
   - Filters out navigation, scripts, and styles

2. **Vector Indexer**
   - Text chunking with overlap for context preservation
   - Sentence-BERT embeddings (`paraphrase-MiniLM-L3-v2`)
   - FAISS L2 distance index for similarity search
   - ~384-dimensional vector space

3. **Q&A System**
   - Semantic search to find relevant chunks
   - Context assembly from top results
   - Answer generation using FLAN-T5-small
   - Source attribution with timings

## Configuration

You can modify these parameters in the code:

```python
# Crawling
delay = 0.5             # Delay between requests (seconds)

# Chunking
chunk_size = 800        # Characters per chunk
overlap = 100           # Overlap between chunks

# Retrieval
top_k = 5               # Number of chunks to retrieve

# Generation
max_new_tokens = 120    # Maximum tokens in answer
```

## File Structure

```
.
├── main.py             # Main script
├── crawled.json        # Crawled website content
├── index.faiss         # FAISS vector index
├── texts.json          # Text chunks
├── urls.json           # Source URLs for each chunk
└── README.md           # This file
```

## Examples

### Example 1: Documentation Website

```bash
# Crawl Python documentation
python main.py crawl 20 https://docs.python.org/3/

# Build index
python main.py index

# Ask questions
python main.py ask "What are decorators in Python?"
python main.py ask "How do I handle exceptions?"
```

### Example 2: Company Website

```bash
# Crawl company site
python main.py crawl 20 https://yourcompany.com

# Build index
python main.py index

# Ask questions
python main.py ask "What products does this company offer?"
python main.py ask "Where is the company located?"
```

### Example 3: Blog or News Site

```bash
# Crawl blog
python main.py crawl 20 https://techblog.example.com

# Build index
python main.py index

# Ask questions
python main.py ask "What are the latest articles about AI?"
python main.py ask "Tell me about machine learning tutorials"
```

## Troubleshooting

### Issue: "Index file not found"
**Solution:** Run the `index` command before asking questions:
```bash
python main.py index
```

### Issue: "No documents found in crawled data"
**Solution:** Run the `crawl` command first:
```bash
python main.py crawl 20 https://example.com
```

### Issue: "Token indices sequence length is longer than maximum"
**Solution:** This warning is handled automatically. The system now limits context length to avoid this issue.

### Issue: Slow performance
**Solution:**
- First run downloads models (~500MB) - subsequent runs are faster
- Reduce `max_pages` for faster crawling
- Reduce `top_k` for faster retrieval

### Issue: Poor answer quality
**Solution:**
- Crawl more pages for better coverage
- Try rephrasing your question
- Check if the crawled content actually contains relevant information

## Performance

Typical performance on a modern CPU:

| Operation | Time |
|-----------|------|
| Crawl (15 pages) | ~10-20 seconds |
| Index (50 chunks) | ~5-10 seconds |
| Query (retrieval + generation) | ~2-5 seconds |

**First run:** Expect 10 minutes max delay for model downloads (~1-2GB).

## Limitations

- **Domain-specific:** Only crawls within the starting domain
- **No authentication:** Cannot access login-protected content
- **Static content only:** Cannot execute JavaScript or access dynamic content
- **English-optimized:** Best results with English content
- **Context window:** Limited to 512 tokens for generation

## Advanced Usage

### Custom Crawling Limits

Edit the `crawl()` function call:

```python
crawl(start_url, max_pages, delay=1.0)  # 1 second delay
```

### Different Embedding Model

Change the model in `build_index()` and `load_index_and_model()`:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster, less accurate
# or
model = SentenceTransformer("all-mpnet-base-v2")  # Slower, more accurate
```

### Different Generation Model

Change the pipeline in the `ask` command:

```python
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # Larger, better quality
    device=-1
)
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is provided as-is for educational and research purposes.

## Credits

Built with:
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Transformers](https://huggingface.co/transformers/) - Language models
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
