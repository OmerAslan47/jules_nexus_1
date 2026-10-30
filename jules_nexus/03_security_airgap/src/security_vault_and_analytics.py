from core.paths import NEXUS_ROOT
import base64
import csv
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from core.plugin_manager import nexus_tool


# ============================================================================
# LOGLAMA YAPILANDIRMASI
# ============================================================================
LOG_DIR = (NEXUS_ROOT + "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.VaultAndAnalytics")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "vault_analytics.log"), encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)


# ============================================================================
# PBKDF2-HMAC-SHA256 ŞİFRELİ KİŞİSEL KASA (ENCRYPTED PERSONAL VAULT)
# ============================================================================
class EncryptedPersonalVault:
    """
    Kullanıcının özel notlarını, proje anahtarlarını veya gizli bilgilerini
    standart kütüphane tabanlı PBKDF2-HMAC-SHA256 (100.000 iterasyon) akış şifrelemesi
    ve HMAC doğrulaması ile `nexus_vault.json` içinde şifreli olarak saklar.
    Misafir modunda kesinlikle erişilemez!
    """

    def __init__(self, vault_path: str = (NEXUS_ROOT + "/nexus_vault.json")):
        self.vault_path = vault_path
        self._lock = threading.Lock()
        self._ensure_vault()

    def _ensure_vault(self):
        if not os.path.exists(self.vault_path):
            with open(self.vault_path, "w", encoding="utf-8") as f:
                json.dump({"version": "2.0", "entries": {}}, f, indent=2)

    @staticmethod
    def _derive_keystream(master_key: str, salt: bytes, length: int) -> bytes:
        """PBKDF2-HMAC-SHA256 ile istenen uzunlukta kriptografik anahtar akışı üretir."""
        dk = hashlib.pbkdf2_hmac(
            "sha256", master_key.encode("utf-8"), salt, 100000, dklen=max(length, 64)
        )
        return dk[:length]

    @classmethod
    def encrypt_secret(cls, plaintext: str, master_key: str) -> Dict[str, str]:
        raw_bytes = plaintext.encode("utf-8")
        salt = secrets.token_bytes(16)
        keystream = cls._derive_keystream(master_key, salt, len(raw_bytes))
        cipher_bytes = bytes(b ^ k for b, k in zip(raw_bytes, keystream))
        mac_key = hashlib.pbkdf2_hmac("sha256", master_key.encode("utf-8"), salt, 1000)
        tag = hmac.new(mac_key, cipher_bytes, hashlib.sha256).hexdigest()
        return {
            "salt_b64": base64.b64encode(salt).decode("ascii"),
            "cipher_b64": base64.b64encode(cipher_bytes).decode("ascii"),
            "hmac_sha256": tag,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def decrypt_secret(cls, record: Dict[str, str], master_key: str) -> Optional[str]:
        try:
            salt = base64.b64decode(record["salt_b64"])
            cipher_bytes = base64.b64decode(record["cipher_b64"])
            expected_tag = record["hmac_sha256"]

            mac_key = hashlib.pbkdf2_hmac(
                "sha256", master_key.encode("utf-8"), salt, 1000
            )
            computed_tag = hmac.new(mac_key, cipher_bytes, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected_tag, computed_tag):
                return None

            keystream = cls._derive_keystream(master_key, salt, len(cipher_bytes))
            plain_bytes = bytes(c ^ k for c, k in zip(cipher_bytes, keystream))
            return plain_bytes.decode("utf-8")
        except Exception:
            return None

    def store_entry(self, label: str, secret_text: str, master_key: str = "nexus_owner_default"):
        with self._lock:
            self._ensure_vault()
            with open(self.vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            enc = self.encrypt_secret(secret_text, master_key)
            data.setdefault("entries", {})[label.strip().lower()] = enc
            with open(self.vault_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def retrieve_entry(self, label: str, master_key: str = "nexus_owner_default") -> Optional[str]:
        with self._lock:
            self._ensure_vault()
            with open(self.vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = data.get("entries", {}).get(label.strip().lower())
            if not rec:
                return None
            return self.decrypt_secret(rec, master_key)

    def list_labels(self) -> List[str]:
        with self._lock:
            self._ensure_vault()
            with open(self.vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return sorted(data.get("entries", {}).keys())


# ============================================================================
# VERİ BİLİMİ VE TABLOSAL (CSV / JSON) İSTATİSTİK ANALİZ MOTORU
# ============================================================================
class TabularDataAnalyzer:
    """
    CSV veya JSON veri dosyalarını okuyup:
    - Satır ve sütun sayılarını,
    - Sayısal sütunlar için Ortalama (Mean), Medyan, Standart Sapma, Min ve Max değerlerini,
    - Eksik (Null/Boş) veri oranlarını otomatik hesaplar.
    """

    @staticmethod
    def analyze_csv(filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            return {"error": f"CSV dosyası bulunamadı: {filepath}"}

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                headers = reader.fieldnames or []
        except Exception as e:
            return {"error": f"CSV okuma hatası: {e}"}

        if not rows or not headers:
            return {"file": filepath, "row_count": 0, "columns": []}

        col_stats: Dict[str, Dict[str, Any]] = {}
        for col in headers:
            numeric_vals: List[float] = []
            missing_cnt = 0
            unique_vals = set()

            for r in rows:
                val = (r.get(col) or "").strip()
                if not val:
                    missing_cnt += 1
                    continue
                unique_vals.add(val)
                try:
                    numeric_vals.append(float(val.replace(",", ".")))
                except ValueError as e:
                    logger.debug(f"Sayisal degere cevirme hatasi: {e}")

            if len(numeric_vals) >= max(len(rows) * 0.5, 1):
                sorted_n = sorted(numeric_vals)
                n = len(sorted_n)
                mean_v = sum(sorted_n) / float(n)
                median_v = sorted_n[n // 2]
                variance = sum((x - mean_v) ** 2 for x in sorted_n) / float(max(n - 1, 1))
                col_stats[col] = {
                    "type": "numeric",
                    "count": n,
                    "missing": missing_cnt,
                    "min": round(sorted_n[0], 3),
                    "max": round(sorted_n[-1], 3),
                    "mean": round(mean_v, 3),
                    "median": round(median_v, 3),
                    "std_dev": round(math.sqrt(variance), 3),
                }
            else:
                col_stats[col] = {
                    "type": "categorical",
                    "unique_values": len(unique_vals),
                    "missing": missing_cnt,
                }

        return {
            "file": os.path.basename(filepath),
            "row_count": len(rows),
            "column_count": len(headers),
            "column_statistics": col_stats,
        }


# ============================================================================
# GİT DEPO VE SÜRÜM KONTROL DENETÇİSİ (GIT REPOSITORY INSPECTOR)
# ============================================================================
class GitRepositoryInspector:
    """
    Belirtilen proje klasöründeki Git durumunu (aktif dal, değiştirilen dosyalar,
    son 5 commit geçmişi) güvenli şekilde analiz eder.
    """

    @staticmethod
    def inspect_repo(repo_dir: str = NEXUS_ROOT) -> str:
        if not os.path.exists(repo_dir):
            return f"Klasör bulunamadı: {repo_dir}"

        try:
            status_out = subprocess.check_output(
                ["git", "-C", repo_dir, "status", "--short", "--branch"],
                stderr=subprocess.STDOUT,
                timeout=5,
            ).decode("utf-8", errors="replace")
            log_out = subprocess.check_output(
                ["git", "-C", repo_dir, "log", "-n", "5", "--oneline"],
                stderr=subprocess.STDOUT,
                timeout=5,
            ).decode("utf-8", errors="replace")
            return (
                f"🌿 Git Depo Raporu ({repo_dir}):\n"
                f"--- Durum ---\n{status_out.strip()}\n"
                f"--- Son 5 Commit ---\n{log_out.strip()}"
            )
        except Exception:
            return f"'{repo_dir}' klasöründe aktif bir Git deposu (.git) bulunamadı."


# ============================================================================
# KASA VE VERİ ANALİZİ @nexus_tool EKLENTİ ARACI BAĞLANTILARI
# ============================================================================
_vault = EncryptedPersonalVault()


@nexus_tool(
    name="sifreli_kasaya_kaydet",
    description="Özel bir notu, şifreyi veya anahtarı PBKDF2-HMAC-SHA256 ile şifreleyerek NEXUS güvenli kasasına kaydeder.",
    risk_level="low",
    guest_allowed=False,
    category="security_vault",
)
def sifreli_kasaya_kaydet(etiket: str, gizli_icerik: str) -> str:
    _vault.store_entry(etiket, gizli_icerik)
    return f"🔐 '{etiket}' kaydı PBKDF2-SHA256 ile şifrelenerek güvenli kasaya kaydedildi."


@nexus_tool(
    name="sifreli_kasadan_oku",
    description="NEXUS güvenli kasasında kayıtlı şifreli bir notu veya anahtarı çözüp getirir (Misafir modunda kilitlidir).",
    risk_level="low",
    guest_allowed=False,
    category="security_vault",
)
def sifreli_kasadan_oku(etiket: str = "") -> str:
    if not etiket.strip():
        labels = _vault.list_labels()
        return f"🔐 Şifreli Kasadaki Kayıt Etiketleri ({len(labels)} adet): {labels}"
    val = _vault.retrieve_entry(etiket)
    if val is None:
        return f"Kasada '{etiket}' adında bir kayıt bulunamadı."
    return f"🔓 [{etiket}]: {val}"


@nexus_tool(
    name="csv_veri_dosyasi_analiz_et",
    description="Belirtilen CSV veri dosyasının satır/sütun sayılarını, ortalama, medyan, standart sapma ve eksik veri istatistiklerini hesaplar.",
    risk_level="low",
    guest_allowed=True,
    category="data_science",
)
def csv_veri_dosyasi_analiz_et(csv_yolu: str) -> str:
    rep = TabularDataAnalyzer.analyze_csv(csv_yolu)
    if "error" in rep:
        return rep["error"]
    return json.dumps(rep, indent=2, ensure_ascii=False)
