"""
RAG Engine for LSPP Assignment (Levels 1 to 5).
Supports:
- Level 1: Multi-document PDF ingestion, tunable chunking & top-k retrieval, polite refusal on out-of-context queries.
- Level 2: Messy PDF extraction with pdfplumber (table recognition into Markdown, column preservation) vs PyPDF.
- Level 3: Token-by-token streaming generator.
- Level 4: Source citations with document filenames and exact page numbers.
- Level 5: Conversational history & contextual query reformulation for pronoun resolution.
"""

import os
import re
from typing import List, Dict, Any, Generator, Tuple, Optional
import dotenv
import pypdf
import pdfplumber

# Load environment variables
dotenv.load_dotenv()
dotenv.load_dotenv("/home/one-point/apps/conversation_ai/.env")


class DocumentChunk:
    """Represents a text chunk with citation metadata."""
    def __init__(self, text: str, source: str, page: int, chunk_id: int, is_table: bool = False):
        self.text = text
        self.source = source
        self.page = page
        self.chunk_id = chunk_id
        self.is_table = is_table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "is_table": self.is_table,
        }


class PDFExtractor:
    """Extracts text and tables from PDFs using standard (pypdf) or enhanced (pdfplumber) methods."""

    @staticmethod
    def extract_with_pypdf(file_path: str) -> List[DocumentChunk]:
        """Level 1 standard extraction using pypdf."""
        chunks = []
        doc_name = os.path.basename(file_path)
        reader = pypdf.PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(DocumentChunk(
                    text=text.strip(),
                    source=doc_name,
                    page=page_idx + 1,
                    chunk_id=0,
                    is_table=False
                ))
        return chunks

    @staticmethod
    def extract_with_pdfplumber(file_path: str) -> List[DocumentChunk]:
        """Level 2 enhanced extraction using pdfplumber with table awareness and markdown table formatting."""
        chunks = []
        doc_name = os.path.basename(file_path)
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                
                # 1. Extract structured tables
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    # Format as Markdown table
                    header = [str(cell).strip() if cell else "" for cell in table[0]]
                    separator = ["---"] * len(header)
                    rows = []
                    for row in table[1:]:
                        clean_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                        rows.append(f"| {' | '.join(clean_row)} |")
                    
                    md_table = (
                        f"| {' | '.join(header)} |\n"
                        f"| {' | '.join(separator)} |\n"
                        + "\n".join(rows)
                    )
                    table_texts.append(md_table)
                
                # 2. Extract regular text with layout preservation
                regular_text = page.extract_text(layout=True) or ""
                
                # Combine table markdown and regular text
                combined_content = regular_text.strip()
                if table_texts:
                    tables_block = "\n\n### Extracted Tables:\n" + "\n\n".join(table_texts)
                    combined_content = f"{combined_content}\n{tables_block}"

                if combined_content.strip():
                    chunks.append(DocumentChunk(
                        text=combined_content.strip(),
                        source=doc_name,
                        page=page_num,
                        chunk_id=0,
                        is_table=len(tables) > 0
                    ))
        return chunks


class TextSplitter:
    """Chunks documents into overlapping pieces while preserving metadata."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_chunks(self, page_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        result_chunks = []
        global_chunk_id = 0

        for p_chunk in page_chunks:
            text = p_chunk.text
            if len(text) <= self.chunk_size:
                result_chunks.append(DocumentChunk(
                    text=text,
                    source=p_chunk.source,
                    page=p_chunk.page,
                    chunk_id=global_chunk_id,
                    is_table=p_chunk.is_table
                ))
                global_chunk_id += 1
                continue

            # Recursive character splitting logic
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                # Try finding a natural break point (paragraph or newline or sentence end)
                if end < len(text):
                    break_point = max(
                        text.rfind("\n\n", start, end),
                        text.rfind("\n", start, end),
                        text.rfind(". ", start, end)
                    )
                    if break_point != -1 and break_point > start + (self.chunk_size // 2):
                        end = break_point + 1

                sub_text = text[start:end].strip()
                if sub_text:
                    result_chunks.append(DocumentChunk(
                        text=sub_text,
                        source=p_chunk.source,
                        page=p_chunk.page,
                        chunk_id=global_chunk_id,
                        is_table=p_chunk.is_table
                    ))
                    global_chunk_id += 1

                if end >= len(text):
                    break
                start = max(start + 1, end - self.chunk_overlap)

        return result_chunks


class SimpleVectorStore:
    """Lightweight vector store with cosine similarity ranking and embedding caching."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                
                def gemini_embed(texts: List[str]) -> List[List[float]]:
                    vectors = []
                    # Clean and batch texts
                    clean_texts = [t[:2000].replace("\x00", "") for t in texts]
                    for i in range(0, len(clean_texts), 64):
                        batch = clean_texts[i:i+64]
                        resp = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=batch
                        )
                        for emb in resp.embeddings:
                            vectors.append(emb.values)
                    return vectors

                self._embedder = gemini_embed
                return self._embedder
            except Exception as e:
                print(f"Warning: Gemini Embeddings init fallback: {e}")

        # Local fallback embedder using sentence-transformers or fastembed or TF-IDF
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embedder = lambda texts: model.encode(texts).tolist()
            return self._embedder
        except Exception:
            try:
                from fastembed import TextEmbedding
                model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self._embedder = lambda texts: [list(v) for v in model.embed(texts)]
                return self._embedder
            except Exception:
                # Ultimate minimal fallback: character n-gram hashing
                def hash_embed(texts: List[str]) -> List[List[float]]:
                    import math
                    vecs = []
                    for t in texts:
                        vec = [0.0] * 128
                        words = re.findall(r'\w+', t.lower())
                        for w in words:
                            idx = hash(w) % 128
                            vec[idx] += 1.0
                        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
                        vecs.append([x / norm for x in vec])
                    return vecs
                self._embedder = hash_embed
                return self._embedder

    def add_chunks(self, chunks: List[DocumentChunk]):
        embed_fn = self._get_embedder()
        texts = [c.text for c in chunks]
        # Batch embedding
        batch_size = 32
        new_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            new_vecs.extend(embed_fn(batch))
        
        self.chunks.extend(chunks)
        self.embeddings.extend(new_vecs)

    def search(self, query: str, k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        embed_fn = self._get_embedder()
        q_vec = embed_fn([query])[0]

        import math
        def dot_product(v1, v2):
            return sum(a * b for a, b in zip(v1, v2))
        
        def magnitude(v):
            return math.sqrt(sum(a * a for a in v)) or 1.0

        q_mag = magnitude(q_vec)
        scored = []
        for chunk, vec in zip(self.chunks, self.embeddings):
            score = dot_product(q_vec, vec) / (q_mag * magnitude(vec))
            scored.append((chunk, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def clear(self):
        self.chunks = []
        self.embeddings = []


class RAGChatbot:
    """
    Main RAG Chatbot implementing Levels 1 through 5.
    """

    SYSTEM_PROMPT = """You are an accurate, honest AI research assistant for document question answering.
You must answer the user's question using ONLY the provided document excerpts below.

Rules:
1. If the answer is found in the context, synthesize a clear, detailed, and directly cited response.
2. If the context does not contain enough information to answer the question, POLITELY SAY:
   "I don't know based on the provided document."
3. Do NOT invent or extrapolate facts outside the context.
4. When citing tables or experimental data, include the exact numbers and labels from the document.
"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.vector_store = SimpleVectorStore()
        self.loaded_documents: List[str] = []
        self.chat_history: List[Dict[str, str]] = []  # Conversational memory (Level 5)
        self.chunk_size = 1000
        self.chunk_overlap = 150
        self.top_k = 4
        self.parser_mode = "pdfplumber"  # 'pypdf' (Level 1) or 'pdfplumber' (Level 2)

    def load_pdf(self, file_path: str, parser_mode: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 150, clear_existing: bool = True) -> Dict[str, Any]:
        """Loads and indexes a PDF document."""
        if parser_mode:
            self.parser_mode = parser_mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if clear_existing:
            self.vector_store.clear()
            self.loaded_documents = []
            self.chat_history = []

        # Level 1 vs Level 2 Extraction
        if self.parser_mode == "pdfplumber":
            raw_chunks = PDFExtractor.extract_with_pdfplumber(file_path)
        else:
            raw_chunks = PDFExtractor.extract_with_pypdf(file_path)

        splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_chunks = splitter.split_chunks(raw_chunks)
        self.vector_store.add_chunks(split_chunks)

        doc_name = os.path.basename(file_path)
        if doc_name not in self.loaded_documents:
            self.loaded_documents.append(doc_name)

        return {
            "status": "success",
            "document": doc_name,
            "pages": len(raw_chunks),
            "chunks_created": len(split_chunks),
            "parser_used": self.parser_mode,
        }

    def _reformulate_query(self, query: str) -> str:
        """Level 5: Resolves pronouns and conversational context into a standalone retrieval query."""
        if not self.chat_history:
            return query

        history_str = "\n".join([f"{turn['role'].upper()}: {turn['content']}" for turn in self.chat_history[-4:]])
        reformulation_prompt = (
            "Given the following conversation history and a follow-up user question, "
            "rephrase the follow-up question into a clear, standalone search query that resolves all pronouns "
            "(e.g., 'he', 'she', 'it', 'they', 'its formula', 'this model', 'that table') based on context.\n"
            "If the question is already standalone, return it unchanged.\n"
            "Return ONLY the standalone search query without explanations or quotes.\n\n"
            f"Conversation History:\n{history_str}\n\n"
            f"Follow-up Question: {query}\n\n"
            "Standalone Search Query:"
        )

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                from google import genai
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=reformulation_prompt
                )
                reformulated = resp.text.strip()
                if reformulated and len(reformulated) > 3:
                    return reformulated
        except Exception as e:
            print(f"Query reformulation error: {e}")

        return query

    def _call_llm(self, prompt: str) -> str:
        """Helper to invoke LLM synchronously."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            from google import genai
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return resp.text or ""
        return "GEMINI_API_KEY not configured."

    def _call_llm_stream(self, prompt: str) -> Generator[str, None, None]:
        """Level 3: Yields answer tokens as they are generated by the LLM."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            from google import genai
            client = genai.Client(api_key=api_key)
            response_stream = client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        else:
            yield "GEMINI_API_KEY not configured."

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieves top-k relevant document chunks."""
        top_k = k or self.top_k
        # Level 5: Contextual query reformulation
        search_query = self._reformulate_query(query)
        return self.vector_store.search(search_query, k=top_k)

    def _format_context_and_citations(self, results: List[Tuple[DocumentChunk, float]]) -> Tuple[str, List[Dict[str, Any]], str]:
        """Formats context for prompt and constructs citation block (Level 4)."""
        context_parts = []
        citations_data = []
        seen_citations = set()
        citations_markdown = []

        for idx, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"[Source: {chunk.source}, Page: {chunk.page}, Chunk: {chunk.chunk_id}]\n{chunk.text}"
            )
            
            cit_key = (chunk.source, chunk.page)
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations_data.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "snippet": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                    "score": round(score, 4)
                })
                citations_markdown.append(f"- 📄 **`{chunk.source}`**, **Page {chunk.page}**")

        context_str = "\n\n---\n\n".join(context_parts)
        citations_str = "\n\n---\n### 📚 Sources & Citations\n" + "\n".join(citations_markdown) if citations_markdown else ""
        return context_str, citations_data, citations_str

    def ask(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Standard question answering with citations (Level 1, 4, 5)."""
        if not self.loaded_documents:
            return {
                "answer": "No document has been loaded yet. Please upload or select a PDF first.",
                "citations": [],
                "standalone_query": query
            }

        standalone_query = self._reformulate_query(query)
        results = self.vector_store.search(standalone_query, k=k or self.top_k)
        context_str, citations_data, citations_str = self._format_context_and_citations(results)

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"=== DOCUMENT CONTEXT ===\n{context_str}\n\n"
            f"=== USER QUESTION ===\n{query}\n\n"
            "ANSWER:"
        )

        raw_answer = self._call_llm(prompt)
        full_answer = f"{raw_answer.strip()}{citations_str}"

        # Record to conversation history (Level 5)
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": raw_answer.strip()})

        return {
            "answer": full_answer,
            "raw_answer": raw_answer.strip(),
            "citations": citations_data,
            "standalone_query": standalone_query,
            "retrieved_chunks": len(results)
        }

    def ask_stream(self, query: str, k: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Level 3: Streaming generator yielding tokens in real time,
        followed by citation metadata at completion.
        """
        if not self.loaded_documents:
            yield {
                "delta": "No document has been loaded yet. Please upload or select a PDF first.",
                "full_text": "No document has been loaded yet. Please upload or select a PDF first.",
                "citations": [],
                "done": True
            }
            return

        standalone_query = self._reformulate_query(query)
        results = self.vector_store.search(standalone_query, k=k or self.top_k)
        context_str, citations_data, citations_str = self._format_context_and_citations(results)

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"=== DOCUMENT CONTEXT ===\n{context_str}\n\n"
            f"=== USER QUESTION ===\n{query}\n\n"
            "ANSWER:"
        )

        accumulated_text = ""
        for token in self._call_llm_stream(prompt):
            accumulated_text += token
            yield {
                "delta": token,
                "full_text": accumulated_text,
                "citations": [],
                "done": False
            }

        # Append citations to full answer
        full_with_citations = f"{accumulated_text.strip()}{citations_str}"
        
        # Save to chat history (Level 5)
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": accumulated_text.strip()})

        yield {
            "delta": citations_str,
            "full_text": full_with_citations,
            "citations": citations_data,
            "standalone_query": standalone_query,
            "done": True
        }

    def clear_history(self):
        """Clears conversational memory."""
        self.chat_history = []


def compare_parsers_on_page(pdf_path: str, page_number: int = 1) -> Dict[str, str]:
    """
    Level 2 demonstration utility: compares text extraction between pypdf and pdfplumber.
    """
    # 1. pypdf
    reader = pypdf.PdfReader(pdf_path)
    pypdf_text = reader.pages[page_number - 1].extract_text() if page_number <= len(reader.pages) else ""

    # 2. pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if page_number <= len(pdf.pages):
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            table_md = []
            for t in tables:
                if t and len(t) >= 2:
                    header = [str(c).strip() if c else "" for c in t[0]]
                    sep = ["---"] * len(header)
                    rows = [f"| {' | '.join(str(c).strip().replace(chr(10), ' ') if c else '' for c in r)} |" for r in t[1:]]
                    table_md.append(f"| {' | '.join(header)} |\n| {' | '.join(sep)} |\n" + "\n".join(rows))
            
            pdfplumber_text = page.extract_text(layout=True) or ""
            if table_md:
                pdfplumber_text += "\n\n[DETECTED STRUCTURED TABLES]:\n" + "\n\n".join(table_md)
        else:
            pdfplumber_text = ""

    return {
        "pypdf": pypdf_text.strip(),
        "pdfplumber": pdfplumber_text.strip()
    }
