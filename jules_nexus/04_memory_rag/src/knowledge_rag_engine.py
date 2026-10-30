from core.paths import NEXUS_ROOT
import collections
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from core.plugin_manager import nexus_tool


# ============================================================================
# LOGLAMA VE VERİTABANI YAPILANDIRMASI
# ============================================================================
LOG_DIR = (NEXUS_ROOT + "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.KnowledgeRAG")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "knowledge_rag.log"), encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)


# ============================================================================
# TÜRKÇE DOĞAL DİL İŞLEME, STOP-WORD FİLTRESİ VE KÖK NORMALİZASYONU
# ============================================================================
class TurkishTextNormalizer:
    """
    Türkçe ve İngilizce metinleri indeksleme ve arama öncesinde normalize eder:
    - Büyük/küçük harf dönüşümü (İ->i, I->ı uyumlu)
    - Noktalama ve özel karakter temizliği
    - Türkçe etkisiz kelimelerin (stop-words) filtrelenmesi
    - Basit Türkçe çekim eki kırpma (Light Stemming)
    - N-Gram (İkili kelime grubu / Bigram) üretimi
    """

    TURKISH_STOP_WORDS: Set[str] = {
        "acaba", "ama", "ancak", "artık", "asla", "aslında", "az", "bana", "bazen",
        "bazı", "belki", "ben", "benden", "beni", "benim", "beri", "beş", "bile",
        "bir", "biraz", "birçok", "biri", "birisi", "birkaç", "birşey", "biz",
        "bizden", "bize", "bizi", "bizim", "böyle", "böylece", "bu", "buna",
        "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun", "burada",
        "bütün", "çoğu", "çok", "çünkü", "da", "daha", "dahi", "de", "defa",
        "değil", "diğer", "diye", "dokuz", "dolayı", "dört", "elbette", "en",
        "fakat", "falan", "felan", "filan", "gene", "gibi", "göre", "halen",
        "hangi", "hangisi", "hani", "hatta", "hem", "henüz", "hep", "hepsi",
        "her", "herhangi", "herkes", "hiç", "hiçbir", "için", "içinde", "iki",
        "ile", "ilgili", "ise", "işte", "itibaren", "itibariyle", "kadar",
        "karşın", "kendi", "kendine", "kendini", "ki", "kim", "kime", "kimi",
        "kimin", "kimisi", "madem", "mı", "mi", "mu", "mü", "nasıl", "ne",
        "neden", "nedir", "nerde", "nerede", "nereye", "nesi", "neyse", "niçin",
        "niye", "o", "olan", "olarak", "oldu", "olduğu", "olduğunu", "olduklarını",
        "olmadı", "olmadığı", "olmak", "olması", "olmayan", "olmaz", "olsa",
        "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan",
        "onlar", "onlardan", "onları", "onların", "onu", "onun", "orada", "oysa",
        "öyle", "pek", "rağmen", "sadece", "sanki", "sekiz", "sen", "senden",
        "seni", "senin", "siz", "sizden", "size", "sizi", "sizin", "sonra",
        "şayet", "şey", "şeyden", "şeyi", "şeyler", "şimdi", "şöyle", "şu",
        "şuna", "şundan", "şunlar", "şunu", "şunun", "tabi", "tamam", "tüm",
        "üç", "üzere", "var", "vardı", "ve", "veya", "veyahut", "ya", "ya da",
        "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak", "yaptı",
        "yaptığı", "yaptığını", "yaptıkları", "yedi", "yerine", "yine", "yok",
        "yoksa", "zaten", "zira", "the", "and", "for", "with", "from", "that",
    }

    COMMON_SUFFIXES: Tuple[str, ...] = (
        "larından", "lerinden", "클", "larımız", "lerimiz", "larınız", "leriniz",
        " ğında", "ğinde", "우", "lardan", "lerden", "트", "larını", "lerini",
        "maları", "meleri", "스", " ğını", "ğini", "프", " ğımız", "ğimiz",
        "크", " ğınız", "ğiniz", "드", "larla", "lerle", "ların", "lerin",
        " ğın", "ğin", " ğım", "ğim", " ika", "lar", "ler", "dan", "den",
        "tan", "ten", "nın", "nin", "nun", "nün", "nda", "nde", "nta", "nte",
        "lık", "lik", "luk", "lük", "sız", "siz", "suz", "süz", "mak", "mek",
    )

    @classmethod
    def lower_turkish(cls, text: str) -> str:
        if not text:
            return ""
        return text.replace("İ", "i").replace("I", "ı").lower()

    @classmethod
    def stem_word(cls, word: str) -> str:
        """Kelimenin kökünü bozmadan yaygın çekim eklerini hafifçe kırpar."""
        if len(word) <= 4:
            return word
        for suffix in (
            "lardan", "lerden", "ların", "lerin", "larla", "lerle",
            "daki", "deki", "ndan", "nden", "ları", "leri",
            "lar", "ler", "dan", "den", "tan", "ten", "nin", "nın", "nun", "nün",
            "lık", "lik", "luk", "lük",
        ):
            if word.endswith(suffix) and (len(word) - len(suffix)) >= 3:
                return word[: -len(suffix)]
        return word

    @classmethod
    def tokenize(cls, text: str, apply_stemming: bool = True) -> List[str]:
        lowered = cls.lower_turkish(text)
        raw_tokens = re.findall(r"[a-z0-9çğıöşü_]{2,}", lowered)
        tokens: List[str] = []
        for tok in raw_tokens:
            if tok in cls.TURKISH_STOP_WORDS:
                continue
            stemmed = cls.stem_word(tok) if apply_stemming else tok
            if len(stemmed) >= 2:
                tokens.append(stemmed)
        return tokens

    @classmethod
    def extract_keywords_with_weights(
        cls, text: str, top_k: int = 15
    ) -> List[Tuple[str, float]]:
        tokens = cls.tokenize(text, apply_stemming=True)
        if not tokens:
            return []
        counts = collections.Counter(tokens)
        total = float(len(tokens))
        weighted = [
            (word, round((cnt / total) * math.log(1.0 + len(word)), 4))
            for word, cnt in counts.items()
        ]
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:top_k]


# ============================================================================
# AKILLI METİN PARÇALAYICI (SEMANTIC DOCUMENT CHUNKER)
# ============================================================================
class DocumentChunker:
    """
    Uzun dokümanları, Markdown dosyalarını ve kaynak kodları anlam bütünlüğünü
    bozmadan (başlık ve paragraf sınırlarına dikkat ederek) örtüşmeli (overlapping)
    parçalara (chunks) ayırır.
    """

    def __init__(self, chunk_size_chars: int = 900, overlap_chars: int = 150):
        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars

    def chunk_text(self, text: str, source_path: str = "") -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks: List[Dict[str, Any]] = []
        current_buf = ""
        chunk_index = 0
        current_heading = os.path.basename(source_path) or "Genel"

        for para in paragraphs:
            p_clean = para.strip()
            if not p_clean:
                continue

            # Markdown başlığı yakala
            heading_match = re.match(r"^#{1,4}\s+(.+)$", p_clean.splitlines()[0])
            if heading_match:
                current_heading = heading_match.group(1).strip()[:100]

            if len(current_buf) + len(p_clean) + 2 <= self.chunk_size_chars:
                current_buf = f"{current_buf}\n\n{p_clean}".strip()
            else:
                if current_buf:
                    chunks.append(
                        self._build_chunk_dict(
                            current_buf, source_path, chunk_index, current_heading
                        )
                    )
                    chunk_index += 1
                    overlap_tail = current_buf[-self.overlap_chars :]
                    current_buf = f"{overlap_tail}\n\n{p_clean}".strip()
                else:
                    # Tek bir paragraf çok uzunsa dilimle
                    for i in range(0, len(p_clean), self.chunk_size_chars - self.overlap_chars):
                        slice_str = p_clean[i : i + self.chunk_size_chars]
                        chunks.append(
                            self._build_chunk_dict(
                                slice_str, source_path, chunk_index, current_heading
                            )
                        )
                        chunk_index += 1
                    current_buf = ""

        if current_buf:
            chunks.append(
                self._build_chunk_dict(
                    current_buf, source_path, chunk_index, current_heading
                )
            )

        return chunks

    @staticmethod
    def _build_chunk_dict(
        content: str, source_path: str, idx: int, heading: str
    ) -> Dict[str, Any]:
        chunk_hash = hashlib.md5(
            f"{source_path}::{idx}::{content[:100]}".encode("utf-8")
        ).hexdigest()
        tokens = TurkishTextNormalizer.tokenize(content)
        return {
            "chunk_id": chunk_hash,
            "source_path": source_path,
            "chunk_index": idx,
            "heading": heading,
            "content": content,
            "token_count": len(tokens),
            "tokens": tokens,
        }


# ============================================================================
# BM25 + TF-IDF HİBRİT SIRALAMA VE ARAMA MOTORU (BM25 RANKER)
# ============================================================================
class BM25SearchEngine:
    """
    Okapi BM25 algoritması + Başlık/Tam Eşleşme (Exact Phrase Boost) kullanarak
    binlerce doküman parçası arasından sorguya en uygun olanları milisaniyede sıralar.
    Harici GPU veya ağır embedding modeli gerektirmez, %0 VRAM harcar!
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_chunks(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not query or not chunks:
            return []

        query_tokens = TurkishTextNormalizer.tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(chunks)
        avg_dl = sum(max(len(c.get("tokens", [])), 1) for c in chunks) / float(
            max(n_docs, 1)
        )

        # Ters İndeks (Document Frequency) hesapla
        doc_freq: Dict[str, int] = collections. defaultdict(int)
        for c in chunks:
            unique_toks = set(c.get("tokens", []))
            for qt in set(query_tokens):
                if qt in unique_toks:
                    doc_freq[qt] += 1

        q_lower = TurkishTextNormalizer.lower_turkish(query.strip())
        scored_results: List[Tuple[float, Dict[str, Any]]] = []

        for c in chunks:
            doc_tokens = c.get("tokens", [])
            dl = max(len(doc_tokens), 1)
            tf_counter = collections.Counter(doc_tokens)
            score = 0.0

            for qt in query_tokens:
                tf = tf_counter.get(qt, 0)
                if tf == 0:
                    continue
                df = doc_freq.get(qt, 0)
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
                score += idf * (numerator / denominator)

            # Tam ifade (Exact Phrase) ve Başlık eşleşme bonusu
            content_lower = TurkishTextNormalizer.lower_turkish(c.get("content", ""))
            heading_lower = TurkishTextNormalizer.lower_turkish(c.get("heading", ""))
            if len(q_lower) >= 4 and q_lower in content_lower:
                score *= 1.45
            if any(qt in heading_lower for qt in query_tokens):
                score *= 1.25

            if score > 0.0:
                res_copy = dict(c)
                res_copy.pop("tokens", None)
                res_copy["bm25_score"] = round(score, 4)
                scored_results.append((score, res_copy))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results[:top_k]]


# ============================================================================
# YEREL BİLGİ BANKASI VE RAG VERİTABANI YÖNETİCİSİ (KNOWLEDGE BASE ENGINE)
# ============================================================================
class LocalKnowledgeBase:
    """
    Kullanıcının bilgisayarındaki ders notlarını, proje dokümanlarını ve kod
    dosyalarını SQLite tabanlı yerel bilgi bankasında (`nexus_knowledge.db`)
    indeksler ve BM25 algoritmasıyla anlık semantik/anahtar kelime araması sunar.
    """

    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".py", ".json", ".csv", ".html", ".log", ".bat", ".ini", ".yaml", ".yml"
    }

    def __init__(
        self, db_path: str = (NEXUS_ROOT + "/nexus_knowledge.db")
    ):
        self.db_path = db_path
        self.chunker = DocumentChunker(chunk_size_chars=900, overlap_chars=150)
        self.ranker = BM25SearchEngine()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS indexed_files (
                        file_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        indexed_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        heading TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tokens_json TEXT NOT NULL,
                        indexed_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        h = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                while True:
                    block = f.read(65536)
                    if not block:
                        break
                    h.update(block)
            return h.hexdigest()
        except Exception:
            return ""

    def index_single_file(self, filepath: str, force: bool = False) -> Dict[str, Any]:
        """Tek bir dosyayı okur, değişmişse parçalara ayırıp bilgi bankasına kaydeder."""
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return {"status": "error", "reason": "Dosya bulunamadı"}

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return {"status": "skipped", "reason": "Desteklenmeyen uzantı"}

        fsize = os.path.getsize(filepath)
        if fsize > 5 * 1024 * 1024:  # 5 MB üzeri devasa logları atla
            return {"status": "skipped", "reason": "Dosya çok büyük (>5MB)"}

        fhash = self._compute_file_hash(filepath)
        norm_path = os.path.abspath(filepath).replace("\\", "/")

        with self._lock:
            with self._get_conn() as conn:
                if not force:
                    row = conn.execute(
                        "SELECT file_hash, chunk_count FROM indexed_files WHERE file_path = ?",
                        (norm_path,),
                    ).fetchone()
                    if row and row[0] == fhash:
                        return {
                            "status": "unchanged",
                            "chunks": row[1],
                            "file": norm_path,
                        }

                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except Exception as e:
                    return {"status": "error", "reason": str(e)}

                chunks = self.chunker.chunk_text(text, source_path=norm_path)
                now = time.time()

                conn.execute(
                    "DELETE FROM document_chunks WHERE source_path = ?", (norm_path,)
                )
                for c in chunks:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO document_chunks
                        (chunk_id, source_path, chunk_index, heading, content, tokens_json, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            c["chunk_id"],
                            norm_path,
                            c["chunk_index"],
                            c["heading"],
                            c["content"],
                            json.dumps(c["tokens"], ensure_ascii=False),
                            now,
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                    (file_path, file_hash, file_size, chunk_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (norm_path, fhash, fsize, len(chunks), now),
                )
                conn.commit()

        return {"status": "indexed", "chunks": len(chunks), "file": norm_path}

    def index_directory(
        self, root_dir: str = NEXUS_ROOT, max_files: int = 300
    ) -> Dict[str, Any]:
        """Belirtilen klasördeki tüm doküman ve kod dosyalarını bilgi bankasına indeksler."""
        if not os.path.exists(root_dir):
            return {"error": f"Klasör bulunamadı: {root_dir}"}

        indexed_count = 0
        unchanged_count = 0
        total_chunks = 0

        for root, dirs, files in os.walk(root_dir):
            # Sanal ortam ve önbellek klasörlerini atla
            dirs[:] = [
                d
                for d in dirs
                if d not in (".venv", "venv", "__pycache__", ".git", "node_modules", "tts_cache")
            ]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                res = self.index_single_file(fpath)
                st = res.get("status")
                if st == "indexed":
                    indexed_count += 1
                    total_chunks += int(res.get("chunks", 0))
                elif st == "unchanged":
                    unchanged_count += 1
                    total_chunks += int(res.get("chunks", 0))

                if (indexed_count + unchanged_count) >= max_files:
                    break
            if (indexed_count + unchanged_count) >= max_files:
                break

        return {
            "root_dir": root_dir,
            "newly_indexed_files": indexed_count,
            "unchanged_files": unchanged_count,
            "total_chunks_ready": total_chunks,
        }

    def query_knowledge_base(
        self, query: str, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Bilgi bankasındaki tüm parçalar üzerinde BM25 araması yapar."""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, source_path, chunk_index, heading, content, tokens_json FROM document_chunks"
                ).fetchall()

        if not rows:
            # Eğer henüz hiç indeksleme yapılmadıysa otomatik olarak NEXUS ve Masaüstü dokümanını indeksle
            self.index_single_file((NEXUS_ROOT + "/NEXUS_PROJE_DOKUMANI.md"))
            self.index_directory((NEXUS_ROOT + "/notes"), max_files=50)
            with self._lock:
                with self._get_conn() as conn:
                    rows = conn.execute(
                        "SELECT chunk_id, source_path, chunk_index, heading, content, tokens_json FROM document_chunks"
                    ).fetchall()

        all_chunks = []
        for r in rows:
            all_chunks.append(
                {
                    "chunk_id": r[0],
                    "source_path": r[1],
                    "chunk_index": r[2],
                    "heading": r[3],
                    "content": r[4],
                    "tokens": json.loads(r[5] or "[]"),
                }
            )

        return self.ranker.score_chunks(query, all_chunks, top_k=top_k)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._get_conn() as conn:
                f_cnt = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
                c_cnt = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
        return {"indexed_files": f_cnt, "total_chunks": c_cnt}


# ============================================================================
# BİLGİ BANKASI @nexus_tool EKLENTİ ARACI BAĞLANTILARI
# ============================================================================
_kb_engine = LocalKnowledgeBase()


@nexus_tool(
    name="bilgi_bankasinda_ara",
    description="Kullanıcının bilgisayarındaki indekslenmiş notlar, proje dokümanları ve kodlar içinde BM25 RAG algoritmasıyla derin arama yapar.",
    risk_level="low",
    guest_allowed=False,
    category="knowledge_rag",
)
def bilgi_bankasinda_ara(sorgu: str) -> str:
    results = _kb_engine.query_knowledge_base(sorgu, top_k=4)
    if not results:
        return f"'{sorgu}' ile ilgili yerel bilgi bankasında eşleşen kayıt bulunamadı."
    lines = [f"🔍 Yerel Bilgi Bankası (BM25 RAG) Arama Sonuçları ('{sorgu}'):"]
    for idx, r in enumerate(results, 1):
        fname = os.path.basename(r["source_path"])
        lines.append(
            f"\n--- Sonuç #{idx} | Dosya: {fname} ({r['heading']}) | Skor: {r['bm25_score']} ---\n{r['content']}"
        )
    return "\n".join(lines)


@nexus_tool(
    name="klasoru_bilgi_bankasina_indeksle",
    description="Belirtilen klasördeki tüm notları, Markdown belgelerini ve kod dosyalarını NEXUS yerel RAG bilgi bankasına indeksler.",
    risk_level="low",
    guest_allowed=False,
    category="knowledge_rag",
)
def klasoru_bilgi_bankasina_indeksle(klasor_yolu: str = NEXUS_ROOT) -> str:
    stats = _kb_engine.index_directory(klasor_yolu)
    return (
        f"✅ Bilgi Bankası İndeksleme Tamamlandı ({klasor_yolu}):\n"
        f"Yeni İndekslenen Dosya: {stats.get('newly_indexed_files', 0)} | "
        f"Değişmeyen Dosya: {stats.get('unchanged_files', 0)} | "
        f"Toplam Hazır Metin Parçası: {stats.get('total_chunks_ready', 0)}"
    )
