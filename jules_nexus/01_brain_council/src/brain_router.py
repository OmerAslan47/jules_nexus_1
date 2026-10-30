from core.paths import NEXUS_ROOT
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import requests


# ============================================================================
# RETRY MEKANİZMASI (Exponential Backoff)
# ============================================================================
def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_err = None
            for attempt in range(max_retries):
                try:
                    res, src = func(*args, **kwargs)
                    if res is not None:
                        return res, src
                except Exception as e:
                    last_err = e
                    logger.debug(f"{func.__name__} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= backoff_factor
            if last_err:
                logger.warning(f"{func.__name__} failed after {max_retries} attempts.")
            return None, None
        return wrapper
    return decorator


# ============================================================================
# LOGLAMA VE TELEMETRİ YAPILANDIRMASI
# ============================================================================
LOG_DIR = (NEXUS_ROOT + "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.BrainRouter")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "brain_router.log"), encoding="utf-8"
    )
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# ============================================================================
# DEVRE KESİCİ (CIRCUIT BREAKER) VE SAĞLIK İZLEYİCİ SINIFI
# ============================================================================
class EndpointCircuitBreaker:
    """
    Her bir yapay zeka katmanı (Colab, Gemini, Yerel Ollama) için:
    - Ardışık hataları (consecutive failures) sayar,
    - Belirli eşik aşıldığında devreyi AÇAR (Open) ve gereksiz zaman aşımı beklemelerini önler,
    - Soğuma süresi (cooldown_sec) sonunda YARI-AÇIK (Half-Open) moda geçip tekrar dener,
    - Ortalama gecikme (latency ms) ve başarı oranı istatistiklerini tutar.
    """

    STATE_CLOSED = "CLOSED"         # Normal çalışıyor
    STATE_OPEN = "OPEN"             # Hata eşiği aşıldı, geçici devre dışı
    STATE_HALF_OPEN = "HALF_OPEN"   # Soğuma süresi bitti, test ediliyor

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_sec: float = 45.0,
        max_latency_samples: int = 50,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state = self.STATE_CLOSED

        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.last_error_message = ""
        self.latencies_ms: Deque[float] = deque(maxlen=max_latency_samples)
        self._lock = threading.Lock()

    def can_attempt(self) -> bool:
        """Bu uç noktaya istek gönderilip gönderilemeyeceğini kontrol eder."""
        with self._lock:
            if self.state == self.STATE_CLOSED:
                return True
            if self.state == self.STATE_OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.cooldown_sec:
                    self.state = self.STATE_HALF_OPEN
                    logger.info(
                        f"Devre Kesici [{self.name}] HALF_OPEN durumuna geçti (Soğuma süresi doldu)."
                    )
                    return True
                return False
            return True

    def record_success(self, latency_ms: float):
        """Başarılı isteği kaydeder ve devreyi CLOSED durumuna çeker."""
        with self._lock:
            self.total_requests += 1
            self.total_successes += 1
            self.consecutive_failures = 0
            self.last_success_time = time.time()
            self.latencies_ms.append(round(latency_ms, 2))
            if self.state != self.STATE_CLOSED:
                logger.info(
                    f"Devre Kesici [{self.name}] tekrar CLOSED (Normal) durumuna döndü."
                )
            self.state = self.STATE_CLOSED

    def record_failure(self, error_msg: str):
        """Başarısız isteği kaydeder, eşik aşıldıysa devreyi OPEN durumuna alır."""
        with self._lock:
            self.total_requests += 1
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            self.last_error_message = str(error_msg)[:300]

            if self.consecutive_failures >= self.failure_threshold:
                if self.state != self.STATE_OPEN:
                    logger.warning(
                        f"Devre Kesici [{self.name}] OPEN (Devre Dışı) durumuna geçti! "
                        f"Ardışık hata: {self.consecutive_failures} | Son hata: {self.last_error_message}"
                    )
                self.state = self.STATE_OPEN

    def reset(self):
        """Devre kesiciyi manuel olarak sıfırlar (Örn: Yeni Colab URL girildiğinde)."""
        with self._lock:
            self.state = self.STATE_CLOSED
            self.consecutive_failures = 0
            self.last_error_message = ""

    def get_metrics(self) -> Dict[str, Any]:
        """Sağlık ve performans metriklerini sözlük olarak döndürür."""
        with self._lock:
            avg_lat = (
                round(sum(self.latencies_ms) / len(self.latencies_ms), 2)
                if self.latencies_ms
                else 0.0
            )
            success_rate = (
                round((self.total_successes / self.total_requests) * 100, 1)
                if self.total_requests > 0
                else 100.0
            )
            return {
                "name": self.name,
                "state": self.state,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "consecutive_failures": self.consecutive_failures,
                "success_rate_pct": success_rate,
                "avg_latency_ms": avg_lat,
                "last_error": self.last_error_message,
            }


# ============================================================================
# AKILLI YANIT ÖNBELLEĞİ (RESPONSE CACHE)
# ============================================================================
class ResponseCache:
    """
    Aynı veya tekrarlanan sorgular için SQLite (WAL modu) tabanlı önbellekleme sağlar.
    Böylece internet/GPU harcamadan 0.01 sn'de yanıt döner.
    """

    def __init__(
        self,
        db_path: str = (NEXUS_ROOT + "/brain_cache.db"),
        max_entries: int = 500,
        ttl_sec: int = 3600 * 12,
    ):
        self.db_path = db_path
        self.max_entries = max_entries
        self.ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS response_cache (
                            hash_key TEXT PRIMARY KEY,
                            prompt_preview TEXT NOT NULL,
                            response TEXT NOT NULL,
                            original_source TEXT NOT NULL,
                            timestamp REAL NOT NULL,
                            hits INTEGER NOT NULL DEFAULT 1
                        )
                        """
                    )
            finally:
                conn.close()

    @staticmethod
    def _make_key(prompt: str, system_prompt: str) -> str:
        raw = f"{system_prompt.strip()}||{prompt.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, prompt: str, system_prompt: str) -> Optional[Dict[str, str]]:
        key = self._make_key(prompt, system_prompt)
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                with conn:
                    row = conn.execute(
                        "SELECT response, original_source, timestamp, hits FROM response_cache WHERE hash_key = ?",
                        (key,),
                    ).fetchone()
                    
                    if not row:
                        return None
                    
                    response, original_source, timestamp, hits = row
                    
                    if (now - timestamp) > self.ttl_sec:
                        conn.execute("DELETE FROM response_cache WHERE hash_key = ?", (key,))
                        return None
                    
                    conn.execute("UPDATE response_cache SET hits = ? WHERE hash_key = ?", (hits + 1, key))
                    return {
                        "response": response,
                        "source": f"cache ({original_source})",
                    }
            except Exception as e:
                logger.debug(f"SQLite cache okuma hatası: {e}")
                return None
            finally:
                conn.close()

    def put(self, prompt: str, system_prompt: str, response: str, source: str):
        # Araç çağrısı içeren yanıtları önbelleğe alma (her seferinde taze çalışsın)
        if "TOOL_CALL:" in response:
            return
        key = self._make_key(prompt, system_prompt)
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                with conn:
                    # Kapasite kontrolü
                    count = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
                    if count >= self.max_entries:
                        conn.execute(
                            "DELETE FROM response_cache WHERE hash_key IN (SELECT hash_key FROM response_cache ORDER BY timestamp ASC LIMIT 1)"
                        )
                    
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO response_cache
                        (hash_key, prompt_preview, response, original_source, timestamp, hits)
                        VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hits FROM response_cache WHERE hash_key = ?) + 1, 1))
                        """,
                        (key, prompt[:80], response, source, now, key),
                    )
            except Exception as e:
                logger.debug(f"SQLite cache yazma hatası: {e}")
            finally:
                conn.close()

    def clear(self):
        with self._lock:
            conn = self._get_conn()
            try:
                with conn:
                    conn.execute("DELETE FROM response_cache")
            except Exception as e:
                logger.debug(f"SQLite cache temizleme hatası: {e}")
            finally:
                conn.close()


# ============================================================================
# KONUŞMA GEÇMİŞİ VE BAĞLAM PENCERESİ YÖNETİCİSİ (SLIDING WINDOW)
# ============================================================================
class ConversationBuffer:
    """
    NEXUS ile kullanıcı arasındaki çok turlu (multi-turn) konuşma geçmişini tutar.
    Bellek taşmasını önlemek için kayan pencere (sliding window) uygular ve
    eski mesajları özetleyerek bağlam sürekliliği sağlar.
    """

    def __init__(
        self,
        max_turns: int = 12,
        history_path: str = (NEXUS_ROOT + "/conversation_history.json"),
    ):
        self.max_turns = max_turns
        self.history_path = history_path
        self._lock = threading.Lock()
        self.messages: Deque[Dict[str, str]] = deque(maxlen=max_turns * 2)
        self.rolling_summary: str = ""
        self._load()

    def _load(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.rolling_summary = data.get("rolling_summary", "")
                    for msg in data.get("messages", [])[-(self.max_turns * 2) :]:
                        self.messages.append(msg)
            except Exception as e:
                logger.debug(f"Konuşma geçmişi okuma hatası: {e}")

    def save(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rolling_summary": self.rolling_summary,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "messages": list(self.messages),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning(f"Konuşma geçmişi kaydedilemedi: {e}")

    def add_exchange(self, user_text: str, assistant_reply: str):
        with self._lock:
            # Eğer pencere dolduysa en eski kullanıcı mesajını özete ekle
            if len(self.messages) >= self.max_turns * 2:
                oldest_user = self.messages[0].get("content", "")[:100]
                if oldest_user:
                    snippet = f"[{oldest_user}]"
                    if len(self.rolling_summary) > 600:
                        self.rolling_summary = self.rolling_summary[-400:] + " | " + snippet
                    else:
                        self.rolling_summary = (
                            f"{self.rolling_summary} | {snippet}".strip(" |")
                        )

            self.messages.append(
                {
                    "role": "user",
                    "content": user_text,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_reply,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
            self.save()

    def get_formatted_messages(self, limit: int = 8) -> List[Dict[str, str]]:
        """API çağrıları için sadece role ve content içeren son mesajları döndürür."""
        with self._lock:
            recent = list(self.messages)[-(limit * 2) :]
            return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self):
        with self._lock:
            self.messages.clear()
            self.rolling_summary = ""
            self.save()


# ============================================================================
# ANA 3 KATMANLI HİBRİT BEYİN YÖNLENDİRİCİ (BRAIN ROUTER V2)
# ============================================================================
class BrainRouter:
    """
    NEXUS 3 Katmanlı Hibrit Beyin Yönlendirici (Kurumsal Tam Sürüm):
    1. Birincil Katman: Google Colab Pro Büyük Beyin (Cloudflare Tüneli üzerinden Qwen 2.5 32B / 14B)
    2. İkincil Katman (Bulut Yedek): Gemini API (Çoklu model aday havuzu + otomatik geçiş)
    3. Üçüncül Katman (Çevrimdışı Acil Durum): Yerel RTX 3050 Ti üzerinde Ollama (qwen2.5:3b)
       - Yerel Ollama servisi kapalıysa otomatik olarak arka planda başlatma yeteneğine sahiptir.
    """

    def __init__(self, config_path: str = (NEXUS_ROOT + "/config.json")):
        self.config_path = config_path
        self.config = self._load_config()

        # Alt Sistemler
        self.cache = ResponseCache()
        self.conversation = ConversationBuffer()

        # Devre Kesiciler
        self.cb_colab = EndpointCircuitBreaker(
            name="Colab_Tunnel", failure_threshold=3, cooldown_sec=15.0
        )
        self.cb_gemini = EndpointCircuitBreaker(
            name="Gemini_Cloud", failure_threshold=4, cooldown_sec=20.0
        )
        self.cb_local = EndpointCircuitBreaker(
            name="Local_Ollama", failure_threshold=2, cooldown_sec=30.0
        )

        # Telemetri Sayaçları
        self.started_at = time.time()
        self.total_queries = 0
        self.estimated_tokens_used = 0
        self._local_service_checked = False

        # Açılışta Colab tünelini arka planda 1.5 sn zaman aşımıyla test et (Sorguları bekletmemek için)
        threading.Thread(target=self._quick_startup_probe, daemon=True).start()

    def _quick_startup_probe(self):
        colab_url = self.config.get("colab_tunnel_url", "").rstrip("/")
        if not colab_url:
            self.cb_colab.record_failure("Colab URL tanımlı değil")
            return
        try:
            r = requests.get(f"{colab_url}/api/tags", timeout=1.8)
            if r.status_code != 200:
                self.cb_colab.record_failure(f"Colab tüneli kapalı (HTTP {r.status_code})")
        except Exception as e:
            self.cb_colab.record_failure(f"Colab tüneli erişilemez: {e}")

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Config okuma hatası: {e}")
        return {}

    def reload_config(self):
        self.config = self._load_config()

    def update_colab_url(self, new_url: str):
        clean_url = new_url.strip().rstrip("/")
        self.config["colab_tunnel_url"] = clean_url
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.cb_colab.reset()
            logger.info(f"Colab Tünel URL güncellendi ve devre kesici sıfırlandı: {clean_url}")
        except Exception as e:
            logger.error(f"Colab URL kaydedilemedi: {e}")

    def check_colab_models(self) -> List[str]:
        """Colab tünelinde o an yüklü olan modelleri sorgular ve config'i otomatik senkronize eder."""
        colab_url = self.config.get("colab_tunnel_url", "").rstrip("/")
        if not colab_url:
            return []
        try:
            r = requests.get(f"{colab_url}/api/tags", timeout=8)
            if r.status_code == 200:
                models_data = r.json().get("models", [])
                names = [m.get("name", "") for m in models_data if m.get("name")]
                if names and self.config.get("colab_model") not in names:
                    self.config["colab_model"] = names[0]
                    with open(self.config_path, "w", encoding="utf-8") as f:
                        json.dump(self.config, f, indent=2, ensure_ascii=False)
                return names
        except Exception as e:
            logger.debug(f"Colab model listesi alınamadı: {e}")
        return []

    def ensure_local_ollama_running(self) -> bool:
        """Yerel Ollama servisi çalışmıyorsa arka planda otomatik olarak başlatmayı dener."""
        local_url = self.config.get("local_ollama_url", "http://localhost:11434").rstrip("/")
        try:
            r = requests.get(f"{local_url}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except Exception as e:
            logger.debug(f"Yerel Ollama api kontrol hatası: {e}")

        if self._local_service_checked:
            return False

        self._local_service_checked = True
        ollama_exe = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        if os.path.exists(ollama_exe):
            try:
                logger.info("Yerel Ollama servisi kapalı, arka planda başlatılıyor...")
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                subprocess.Popen(
                    [ollama_exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                time.sleep(2.5)
                return True
            except Exception as e:
                logger.warning(f"Yerel Ollama otomatik başlatılamadı: {e}")
        return False

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _try_colab(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.cb_colab.can_attempt():
            return None, None

        colab_url = self.config.get("colab_tunnel_url", "").rstrip("/")
        if not colab_url:
            return None, None

        colab_model = self.config.get("colab_model", "qwen2.5:32b")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{colab_url}/api/chat",
                json={
                    "model": colab_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_ctx": 8192},
                },
                timeout=(2.0, 25.0),
            )
            latency = (time.time() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    self.cb_colab.record_success(latency)
                    return content, f"colab ({colab_model} - {int(latency)}ms)"
            elif resp.status_code == 404:
                # Belki model adı farklıdır (14b vs 32b), otomatik kontrol et
                available = self.check_colab_models()
                if available:
                    alt_model = available[0]
                    resp2 = requests.post(
                        f"{colab_url}/api/chat",
                        json={"model": alt_model, "messages": messages, "stream": False},
                        timeout=90,
                    )
                    if resp2.status_code == 200:
                        content = resp2.json().get("message", {}).get("content", "").strip()
                        if content:
                            self.cb_colab.record_success((time.time() - t0) * 1000.0)
                            return content, f"colab ({alt_model})"
            self.cb_colab.record_failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            self.cb_colab.record_failure(str(e))
            logger.warning(f"Colab katmanı hatası: {e}")

        return None, None

    @retry_with_backoff(max_retries=2, initial_delay=1.5)
    def _try_gemini(
        self, prompt: str, system_prompt: str, history_msgs: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.cb_gemini.can_attempt():
            return None, None

        client = self.get_gemini_client()
        if client is None:
            return None, None

        # Konuşma geçmişini metin bağlamına dönüştür
        history_block = ""
        if history_msgs:
            lines = []
            for m in history_msgs[-6:]:
                role_tr = "Kullanıcı" if m["role"] == "user" else "NEXUS"
                lines.append(f"{role_tr}: {m['content']}")
            history_block = "\n--- Önceki Konuşma Geçmişi ---\n" + "\n".join(lines) + "\n"

        full_contents = f"{system_prompt}\n{history_block}\nKullanıcı: {prompt}"

        t0 = time.time()
        last_err = ""
        for g_model in self.gemini_models():
            try:
                res = client.models.generate_content(
                    model=g_model,
                    contents=full_contents,
                )
                latency = (time.time() - t0) * 1000.0
                if res and res.text:
                    self.cb_gemini.record_success(latency)
                    return res.text.strip(), f"gemini_api ({g_model} - {int(latency)}ms)"
            except Exception as e:
                last_err = str(e)
                continue

        self.cb_gemini.record_failure(last_err)
        return None, None

    # ------------------------------------------------------------------
    # GEMINI İSTEMCİSİ (tek sefer oluşturulur, bağlantı havuzu yeniden kullanılır)
    # ------------------------------------------------------------------
    def _resolve_gemini_key(self) -> str:
        key = os.environ.get("GEMINI_API_KEY") or self.config.get("gemini_api_key")
        if not key:
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as rk:
                    val, _ = winreg.QueryValueEx(rk, "GEMINI_API_KEY")
                    if val:
                        key = str(val).strip()
                        os.environ["GEMINI_API_KEY"] = key
            except Exception as e:
                logger.debug(f"Winreg Gemini key okuma hatası: {e}")
        return (key or "").strip()

    def get_gemini_client(self):
        key = self._resolve_gemini_key()
        if not key:
            return None
        if getattr(self, "_gemini_client", None) is not None and self._gemini_key == key:
            return self._gemini_client
        try:
            from google import genai

            self._gemini_client = genai.Client(api_key=key)
            self._gemini_key = key
            return self._gemini_client
        except Exception as e:
            self.cb_gemini.record_failure(f"GenAI SDK init hatası: {e}")
            return None

    def gemini_models(self) -> List[str]:
        models = self.config.get("gemini_models") or [
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
        ]
        return [m for m in models if m]

    # ------------------------------------------------------------------
    # COLAB ÇOKLU MODEL ROL EŞLEŞTİRMESİ
    # ------------------------------------------------------------------
    ROLE_FAMILY_HINTS = {
        "reasoner": ("deepseek", "qwq", "phi4-reasoning"),
        "auditor": ("qwen2.5-coder", "qwen3-coder", "qwen", "codellama"),
        "challenger": ("llama", "mistral", "phi"),
        "synthesizer": ("gemma", "qwen2.5", "llama"),
        "fast_reflex": ("qwen2.5:7b", "qwen2.5:3b", "gemma2:2b", "llama3.2", "qwen"),
    }

    def get_colab_role_models(self, force: bool = False) -> Dict[str, str]:
        """
        Colab'da GERÇEKTEN kurulu olan modelleri öğrenir ve konsey rollerine eşler.
        Önce gateway'in /nexus_info uç noktasını, yoksa /api/tags listesini kullanır.
        Sonuç 5 dakika önbelleğe alınır. Kurulu olmayan modele asla istek atılmaz.
        """
        now = time.time()
        cache = getattr(self, "_role_cache", None)
        if cache and not force and now - cache[0] < 300:
            return cache[1]

        colab_url = self.config.get("colab_tunnel_url", "").rstrip("/")
        roles: Dict[str, str] = {}
        if colab_url:
            try:
                r = requests.get(f"{colab_url}/nexus_info", timeout=6)
                if r.status_code == 200:
                    roles = dict(r.json().get("roles", {}))
            except Exception as e:
                logger.debug(f"Colab rol okuma hatası: {e}")
            if not roles:
                names = self.check_colab_models()
                for role, hints in self.ROLE_FAMILY_HINTS.items():
                    for h in hints:
                        hit = next((n for n in names if h in n.lower()), None)
                        if hit:
                            roles[role] = hit
                            break
                if names:
                    for role in self.ROLE_FAMILY_HINTS:
                        roles.setdefault(role, names[0])

        manual = self.config.get("colab_role_models") or {}
        roles.update({k: v for k, v in manual.items() if v})
        self._role_cache = (now, roles)
        return roles

    # ------------------------------------------------------------------
    # AKIŞLI (STREAMING) ÜRETİM — Uzun sesli yanıtlar için
    # ------------------------------------------------------------------
    @staticmethod
    def strip_think(text: str) -> str:
        import re as _re

        text = _re.sub(r"<think>[\s\S]*?</think>", "", text or "")
        if "<think>" in text:
            text = text.split("<think>")[0]
        return text.strip()

    def _ollama_stream(self, base_url: str, model: str, messages, max_tokens: int, timeout_read: float):
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.5, "num_ctx": 8192, "num_predict": max_tokens},
            },
            stream=True,
            timeout=(4.0, timeout_read),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
        in_think = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            piece = data.get("message", {}).get("content", "")
            if piece:
                # DeepSeek-R1 <think> bloklarını sese yansıtma
                out = ""
                buf = piece
                while buf:
                    if in_think:
                        end = buf.find("</think>")
                        if end < 0:
                            buf = ""
                        else:
                            buf = buf[end + 8 :]
                            in_think = False
                    else:
                        st = buf.find("<think>")
                        if st < 0:
                            out += buf
                            buf = ""
                        else:
                            out += buf[:st]
                            buf = buf[st + 7 :]
                            in_think = True
                if out:
                    yield out
            if data.get("done"):
                break

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096,
        role: str = "synthesizer",
        airgap: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Yanıtı parça parça üretir (generator). Sırasıyla Colab -> Gemini -> Yerel Ollama dener.
        İlk parça geldikten sonra kaynak değiştirilmez. Son eleman olarak ('__SOURCE__', kaynak) döner.
        """
        self.reload_config()
        history_msgs = self.conversation.get_formatted_messages(limit=4)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": prompt})
        cancel_check = cancel_check or (lambda: False)

        # 1) COLAB
        colab_url = self.config.get("colab_tunnel_url", "").rstrip("/")
        if not airgap and colab_url and self.cb_colab.can_attempt():
            model = self.get_colab_role_models().get(role) or self.config.get("colab_model")
            got_any = False
            try:
                t0 = time.time()
                for piece in self._ollama_stream(colab_url, model, messages, max_tokens, 120.0):
                    if cancel_check():
                        return
                    got_any = True
                    yield piece
                if got_any:
                    self.cb_colab.record_success((time.time() - t0) * 1000.0)
                    yield ("__SOURCE__", f"colab_stream({model})")
                    return
            except Exception as e:
                self.cb_colab.record_failure(str(e))
                logger.warning(f"Colab akış hatası: {e}")
                if got_any:
                    yield ("__SOURCE__", f"colab_stream({model}, yarım)")
                    return

        # 2) GEMINI
        client = None if airgap else self.get_gemini_client()
        if client is not None and self.cb_gemini.can_attempt():
            history_block = "\n".join(
                f"{'Kullanıcı' if m['role'] == 'user' else 'NEXUS'}: {m['content']}"
                for m in history_msgs[-6:]
            )
            contents = f"{system_prompt}\n\n--- Önceki Konuşma ---\n{history_block}\n\nKullanıcı: {prompt}"
            for g_model in self.gemini_models():
                got_any = False
                try:
                    from google.genai import types

                    t0 = time.time()
                    stream = client.models.generate_content_stream(
                        model=g_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            max_output_tokens=max_tokens, temperature=0.6
                        ),
                    )
                    for chunk in stream:
                        if cancel_check():
                            return
                        txt = getattr(chunk, "text", None)
                        if txt:
                            got_any = True
                            yield txt
                    if got_any:
                        self.cb_gemini.record_success((time.time() - t0) * 1000.0)
                        yield ("__SOURCE__", f"gemini_stream({g_model})")
                        return
                except Exception as e:
                    logger.warning(f"Gemini akış hatası ({g_model}): {e}")
                    if got_any:
                        yield ("__SOURCE__", f"gemini_stream({g_model}, yarım)")
                        return
                    continue
            self.cb_gemini.record_failure("stream başarısız")

        # 3) YEREL OLLAMA
        if self.cb_local.can_attempt() and self.ensure_local_ollama_running():
            local_url = self.config.get("local_ollama_url", "http://localhost:11434").rstrip("/")
            local_model = self.config.get("local_fallback_model", "qwen2.5:3b")
            got_any = False
            try:
                for piece in self._ollama_stream(local_url, local_model, messages, max_tokens, 120.0):
                    if cancel_check():
                        return
                    got_any = True
                    yield piece
                if got_any:
                    yield ("__SOURCE__", f"local_stream({local_model})")
                    return
            except Exception as e:
                self.cb_local.record_failure(str(e))

        yield ("__SOURCE__", "error")

    @retry_with_backoff(max_retries=2, initial_delay=1.0)
    def _try_local_ollama(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.cb_local.can_attempt():
            return None, None

        self.ensure_local_ollama_running()
        local_url = self.config.get("local_ollama_url", "http://localhost:11434").rstrip("/")
        local_model = self.config.get("local_fallback_model", "qwen2.5:3b")

        t0 = time.time()
        try:
            resp = requests.post(
                f"{local_url}/api/chat",
                json={"model": local_model, "messages": messages, "stream": False},
                timeout=35,
            )
            latency = (time.time() - t0) * 1000.0
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "").strip()
                if content:
                    self.cb_local.record_success(latency)
                    return content, f"local_ollama ({local_model} - {int(latency)}ms)"
            self.cb_local.record_failure(f"HTTP {resp.status_code}")
        except Exception as e:
            self.cb_local.record_failure(str(e))
        return None, None

    def ask(
        self,
        prompt: str,
        system_prompt: str = "Sen NEXUS adında Türkçe konuşan gelişmiş bir kişisel yapay zeka asistanısın.",
        context_messages: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = False,
        record_history: bool = True,
    ) -> Dict[str, str]:
        """
        Ana sorgu metodu.
        Sırasıyla: Önbellek -> Colab Pro (32B/14B) -> Gemini API -> Yerel Ollama (3B) katmanlarını dener.
        """
        self.reload_config()
        self.total_queries += 1

        if use_cache:
            cached = self.cache.get(prompt, system_prompt)
            if cached:
                return cached

        history_msgs = (
            context_messages
            if context_messages is not None
            else self.conversation.get_formatted_messages(limit=5)
        )

        messages = [{"role": "system", "content": system_prompt}]
        if self.conversation.rolling_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Önceki konuşmaların özeti: {self.conversation.rolling_summary}",
                }
            )
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": prompt})

        # 1. KATMAN: Google Colab Pro Tüneli
        reply, source = self._try_colab(messages)

        # 2. KATMAN: Gemini Bulut Yedek
        if not reply:
            reply, source = self._try_gemini(prompt, system_prompt, history_msgs)

        # 3. KATMAN: Yerel RTX 3050 Ti Ollama
        if not reply:
            reply, source = self._try_local_ollama(messages)

        if reply and source:
            self.estimated_tokens_used += (len(prompt) + len(reply)) // 4
            if use_cache:
                self.cache.put(prompt, system_prompt, reply, source)
            if record_history and "TOOL_CALL:" not in reply:
                self.conversation.add_exchange(prompt, reply)
            return {"response": reply, "source": source}

        err_msg = (
            "Şu anda ne Google Colab ne Gemini ne de yerel Ollama motoruna ulaşılamadı. "
            "Lütfen internet bağlantınızı veya Colab tünel adresinizi kontrol edin."
        )
        return {"response": err_msg, "source": "error"}

    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Tüm beyin katmanlarının sağlık, devre kesici ve performans raporunu döndürür."""
        uptime_min = round((time.time() - self.started_at) / 60.0, 1)
        return {
            "uptime_minutes": uptime_min,
            "total_queries": self.total_queries,
            "estimated_tokens": self.estimated_tokens_used,
            "conversation_turns": len(self.conversation.messages) // 2,
            "endpoints": {
                "colab": self.cb_colab.get_metrics(),
                "gemini": self.cb_gemini.get_metrics(),
                "local_ollama": self.cb_local.get_metrics(),
            },
        }
