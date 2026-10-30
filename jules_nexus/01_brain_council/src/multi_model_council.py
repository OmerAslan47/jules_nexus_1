from core.paths import NEXUS_ROOT
import concurrent.futures
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from core.privacy_airgap_firewall import PrivacyAirGapFirewall


LOG_DIR = (NEXUS_ROOT + "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.MultiModelCouncil")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "multi_model_council.log"), encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)


class MultiModelCouncil:
    """
    NEXUS v4.0 - Çoklu YZ Aileleri Katmanlı Bilişsel Doğrulama Konseyi
    (Multi-Family High-Parameter Mixture-of-Agents & Air-Gap Privacy Protected):
    Farklı yapay zeka ailelerinin (DeepSeek + Qwen + Meta Llama + Google Gemma) güçlü yönlerini
    katmanlı bir mimaride birleştirir:
      - AİLE 1 / KATMAN 1 (DeepSeek-R1 32B/14B): İlk İlkeler (First Principles), Derin Mantık ve Matematiksel Çıkarım
      - AİLE 2 / KATMAN 2 (Qwen 2.5 Coder 32B/14B): Algoritma, Mühendislik, Doğruluk ve Hata Denetimi
      - AİLE 3 / KATMAN 3 (Llama 3.3 70B / Llama 3.1 / Gemma 3): Karşıt Görüş (Devil's Advocate) ve Çelişki Tespiti
      - AİLE 4 / KATMAN 4 (Qwen 2.5 32B / Gemma 3 Baş Sentezleyici): Tüm katmanların bulgularını birleştirip
        hataları elenmiş, son derece zeki ve akıcı Türkçe konuşma yanıtını sentezler.
    """

    COLAB_MULTI_FAMILY_MODELS = {
        "reasoner": "deepseek-r1:14b",        # Aile 1: DeepSeek (Mantık & Matematik)
        "auditor": "qwen2.5-coder:14b",       # Aile 2: Alibaba Qwen-Coder (Doğruluk & Kod Denetimi)
        "challenger": "llama3.1:8b",          # Aile 3: Meta Llama (Eleştirel Kontrol & Mantık Testi)
        "synthesizer": "gemma2:9b",           # Aile 4: Google Gemma / Qwen (Akıcı Türkçe Sentez)
        "fast_reflex": "qwen2.5:7b",          # Kısa sorular için <0.4s hızlı yanıt
    }

    GEMINI_SPECIALIST_MODELS = {
        "reasoner": "gemini-3.5-flash-lite",
        "auditor": "gemini-flash-lite-latest",
        "challenger": "gemini-3.5-flash-lite",
        "synthesizer": "gemini-flash-lite-latest",
        "fast_reflex": "gemini-flash-lite-latest",
    }

    def __init__(self, brain_router, firewall: Optional[PrivacyAirGapFirewall] = None):
        self.brain = brain_router
        self.firewall = firewall or PrivacyAirGapFirewall()
        self.council_invocations = 0
        self.fast_reflex_invocations = 0
        self.last_drive_sync_time = 0.0
        self._gemini_client = None

    def _get_gemini_client(self):
        if self._gemini_client is None:
            gemini_key = self._resolve_gemini_key()
            if gemini_key:
                try:
                    from google import genai

                    self._gemini_client = genai.Client(api_key=gemini_key)
                except Exception as e:
                    logger.debug(f"Gemini client oluşturulurken hata: {e}")
        return self._gemini_client

    def _resolve_gemini_key(self) -> str:
        k = os.environ.get("GEMINI_API_KEY") or self.brain.config.get("gemini_api_key")
        if k:
            return str(k).strip()
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as rk:
                val, _ = winreg.QueryValueEx(rk, "GEMINI_API_KEY")
                if val:
                    os.environ["GEMINI_API_KEY"] = str(val).strip()
                    return str(val).strip()
        except Exception as e:
            logger.debug(f"Winreg Gemini key okuma hatası: {e}")
        return ""

    def _call_specific_specialist(
        self,
        role_type: str,
        system_instruction: str,
        user_prompt: str,
        timeout_sec: float = 35.0,
    ) -> Tuple[str, str]:
        """
        Kapalı Kutu Mahremiyet Duvarı'ndan geçirerek ilgili YZ ailesi uzmanını çağırır.
        Eğer strict_airgap_mode açıksa dış ağa hiç çıkmadan doğrudan yerel RTX 3050 Ti Ollama'yı kullanır.
        """
        safe_sys = self.firewall.sanitize_outbound_payload(system_instruction)
        safe_usr = self.firewall.sanitize_outbound_payload(user_prompt)

        messages = [
            {"role": "system", "content": safe_sys},
            {"role": "user", "content": safe_usr},
        ]

        # TAM KAPALI KUTU MODU AKTİFSE SADECE YEREL OLLAMA ÇALIŞIR
        if self.firewall.strict_airgap_mode:
            reply, src = self.brain._try_local_ollama(messages)
            return (reply or ""), (src or "local_airgap_ollama")

        # 1. Colab Çoklu YZ Ailesi Cluster Aktifse
        if self.brain.cb_colab.can_attempt():
            colab_url = self.brain.config.get("colab_tunnel_url", "").rstrip("/")
            if colab_url:
                colab_roles = self.brain.get_colab_role_models()
                target_model = colab_roles.get(role_type) or self.COLAB_MULTI_FAMILY_MODELS.get(
                    role_type, self.brain.config.get("colab_model", "qwen2.5:7b")
                )
                try:
                    resp = requests.post(
                        f"{colab_url}/api/chat",
                        json={
                            "model": target_model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": 0.25, "num_ctx": 8192},
                        },
                        timeout=(2.0, timeout_sec),
                    )
                    if resp.status_code == 200:
                        txt = resp.json().get("message", {}).get("content", "").strip()
                        if txt:
                            return txt, f"colab_family({target_model})"
                    else:
                        self.brain.cb_colab.record_failure(f"HTTP {resp.status_code}")
                except Exception as e:
                    self.brain.cb_colab.record_failure(str(e))

        # 2. Hızlı Bulut Uzman Havuzu
        client = self._get_gemini_client()
        if client and self.brain.cb_gemini.can_attempt():
            try:
                preferred = self.GEMINI_SPECIALIST_MODELS.get(
                    role_type, "gemini-flash-lite-latest"
                )
                candidates = [
                    preferred,
                    "gemini-flash-lite-latest",
                    "gemini-3.5-flash-lite",
                ]
                seen = set()
                ordered = [m for m in candidates if not (m in seen or seen.add(m))]

                full_input = f"{safe_sys}\n\n{safe_usr}"
                for g_model in ordered:
                    try:
                        res = client.models.generate_content(
                            model=g_model,
                            contents=full_input,
                        )
                        if res and res.text:
                            return res.text.strip(), f"cloud_specialist({g_model})"
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Cloud specialist hatası: {e}")

        # 3. Yerel RTX 3050 Ti Ollama Yedeği
        reply, src = self.brain._try_local_ollama(messages)
        if reply:
            return reply, src or "local_ollama"

        return "", "none"

    def run_ultra_fast_short_query(
        self,
        user_query: str,
        owner_dossier: str,
        personal_graph_context: str,
        conversation_history_text: str,
    ) -> Dict[str, Any]:
        """
        Kısa ve net sorular (<= 9 kelime) için Maksimum Hızlı (<0.5 sn) Tek Adımlı Keskin Yanıt Motoru.
        """
        t0 = time.time()
        self.fast_reflex_invocations += 1

        fast_sys = (
            "Sen NEXUS'un J.A.R.V.I.S. Hızlı Refleks ve Keskin Zeka Motorusun.\n"
            "KURAL: Kullanıcının sorusuna sıfır dolgu cümlesiyle, 1 ila 3 akıcı ve doğal Türkçe cümlede, "
            "%100 doğru, net ve zeki yanıt ver. Asla 'Sahip' gibi yapay kelimeler kullanma.\n"
            f"{owner_dossier}\n"
            f"Bağlam: {personal_graph_context}\n"
            f"Son Konuşma: {conversation_history_text}"
        )
        reply, src = self._call_specific_specialist(
            "fast_reflex", fast_sys, f"Kullanıcı: {user_query}", timeout_sec=6.0
        )
        ms = int((time.time() - t0) * 1000)
        return {
            "response": reply or "Seni dinliyorum patron.",
            "source": f"UltraFast_Reflex ({src} | {ms}ms)",
            "deliberation_ms": ms,
        }

    def run_fast_verified_pipeline(
        self,
        user_query: str,
        personal_graph_context: str,
        conversation_history_text: str,
        active_window_info: str,
        owner_dossier: str = "",
    ) -> Dict[str, Any]:
        """
        2-Katmanlı Doğrulama Konseyi (Farklı YZ Aileleri: Mantık Üretici -> Doğruluk & Hata Denetçisi):
        - Katman 1 (DeepSeek / Reasoner): İlk kapsamlı analitik çözümü üretir.
        - Katman 2 (Qwen-Coder / Auditor): Doğruluğu, mantığı ve akıcılığı denetleyip kusursuzlaştırır.
        """
        t0 = time.time()
        self.council_invocations += 1

        if not self.firewall.strict_airgap_mode and self.brain.cb_colab.can_attempt():
            colab_url = self.brain.config.get("colab_tunnel_url", "").rstrip("/")
            if colab_url:
                try:
                    r = requests.post(
                        f"{colab_url}/nexus_council",
                        json={
                            "query": self.firewall.sanitize_outbound_payload(user_query),
                            "context": self.firewall.sanitize_outbound_payload(personal_graph_context),
                            "history": self.firewall.sanitize_outbound_payload(conversation_history_text),
                            "mode": "layered_verify",
                        },
                        timeout=(2.0, 20.0),
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("response"):
                            ms = int((time.time() - t0) * 1000)
                            return {
                                "response": data["response"],
                                "source": f"Colab_MultiFamily_Council ({ms}ms)",
                                "deliberation_ms": ms,
                            }
                except Exception as e:
                    logger.debug(f"Konsey deliberasyon isteği hatası (Colab): {e}")

        reasoner_sys = (
            "Sen NEXUS Katmanlı Konseyi'nin 1. Ailesi (DeepSeek Mantık ve Çözüm Uzmanı) olarak görev yapıyorsun.\n"
            "GÖREVİN:\n"
            "1. Mikrofondan gelen cümlede ufak harf kayması varsa konuşma geçmişinden gerçek niyeti anla.\n"
            "2. Soruya en doğru, mantıklı ve teknik olarak sağlam çözümü üret.\n"
            f"{owner_dossier}\n"
            f"Bilgi Grafiği:\n{personal_graph_context}\n"
            f"Son Konuşma Geçmişi:\n{conversation_history_text}"
        )
        draft_reply, src1 = self._call_specific_specialist(
            "reasoner", reasoner_sys, f"Kullanıcı Sorusu: {user_query}", timeout_sec=25.0
        )

        auditor_sys = (
            "Sen NEXUS Katmanlı Konseyi'nin 2. Ailesi (Doğruluk, Mantık, Hata Denetçisi ve Baş Sentezleyici) olarak görev yapıyorsun.\n"
            "DENETİM KURALLARIN:\n"
            "1. 1. Katmanın taslağındaki mantık hatalarını, uydurma bilgileri veya sorudan kopuklukları tamamen düzelt.\n"
            "2. 'Sahip', 'holografik formum' gibi yapay rol yapma ifadelerini temizle; J.A.R.V.I.S. gibi keskin zekalı, samimi ve akıcı bir Türkçe kullan.\n"
            "3. Yanıt sesli okunacağı için 2-4 akıcı cümlede doğrudan özü ver.\n"
            "SADECE mükemmelleştirilmiş nihai Türkçe yanıtı döndür."
        )
        audit_input = (
            f"SON KONUŞMA GEÇMİŞİ:\n{conversation_history_text}\n\n"
            f"KULLANICI SORUSU:\n{user_query}\n\n"
            f"1. KATMAN TASLAĞI:\n{draft_reply}\n\n"
            "Doğruluğu ve mantığı denetlenmiş nihai cevap:"
        )
        verified_reply, src2 = self._call_specific_specialist(
            "auditor", auditor_sys, audit_input, timeout_sec=20.0
        )

        final_out = verified_reply or draft_reply or "Seni dinliyorum patron."
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "response": final_out,
            "source": f"Layered_Council ({src1} -> {src2} | {elapsed_ms}ms)",
            "deliberation_ms": elapsed_ms,
        }

    def run_council_deliberation(
        self,
        user_query: str,
        personal_graph_context: str,
        active_window_info: str,
        conversation_history_text: str = "",
        owner_dossier: str = "",
    ) -> Dict[str, Any]:
        """
        Sistem-2 Derin Düşünme: 3 Farklı YZ Ailesi (DeepSeek Mantık + Qwen Mühendislik + Llama Eleştirmen)
        paralel çalışır, ardından 4. Katman (Gemma/Qwen Baş Hakem) nihai sentezi yapar!
        """
        t0 = time.time()
        self.council_invocations += 1

        proposer_sys = (
            "Sen Konseyin 1. YZ Ailesi (DeepSeek-R1 Baş Analitik ve Mantık Çözümleyicisi) uzmanısın.\n"
            "Problemi İlk İlkelerden (First Principles) başlayarak çöz ve matematiksel/mantıksal adımları kur.\n"
            f"{owner_dossier}\n{personal_graph_context}"
        )
        auditor_sys = (
            "Sen Konseyin 2. YZ Ailesi (Qwen-Coder Algoritma, Mimari ve Doğruluk Denetçisi) uzmanısın.\n"
            "Problemdeki teknik darboğazları, güvenlik kriterlerini ve en verimli mühendislik çözümünü çıkar."
        )
        challenger_sys = (
            "Sen Konseyin 3. YZ Ailesi (Llama-3.3 Şeytanın Avukatı & Hata Yakalayıcı) uzmanısın.\n"
            "Standart cevaplarda yapılan yaygın hataları, yanılgıları ve gözden kaçan uç durumları tespit et."
        )

        draft_1, draft_2, draft_3 = "", "", ""
        sources_used: List[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f1 = executor.submit(
                self._call_specific_specialist, "reasoner", proposer_sys, user_query, 12.0
            )
            f2 = executor.submit(
                self._call_specific_specialist, "auditor", auditor_sys, user_query, 12.0
            )
            f3 = executor.submit(
                self._call_specific_specialist, "challenger", challenger_sys, user_query, 12.0
            )

            for fut, idx in ((f1, 1), (f2, 2), (f3, 3)):
                try:
                    txt, src = fut.result(timeout=13.0)
                    if src:
                        sources_used.append(src)
                    if idx == 1:
                        draft_1 = txt
                    elif idx == 2:
                        draft_2 = txt
                    else:
                        draft_3 = txt
                except Exception as e:
                    logger.debug(f"Uzman çağrı hatası (idx {idx}): {e}")

        arbiter_sys = (
            "Sen NEXUS Çoklu YZ Ailesi Konseyi'nin Baş Hakemi ve Sentez Motorusun.\n"
            "Aşağıda 3 farklı yapay zeka ailesinin (DeepSeek Mantık, Qwen Mühendislik, Llama Eleştirmen) analizleri yer alıyor.\n"
            "Bu üç uzmanın bulgularını birleştir, tüm hataları ele ve kullanıcıya J.A.R.V.I.S. seviyesinde "
            "son derece zeki, akıcı, net ve uygulanabilir nihai Türkçe yanıtı ver."
        )
        synthesis_prompt = (
            f"KULLANICI SORUSU:\n{user_query}\n\n"
            f"--- AİLE 1 (DEEPSEEK MANTIK UZMANI) ---\n{draft_1}\n\n"
            f"--- AİLE 2 (QWEN MÜHENDİSLİK & DOĞRULUK DENETÇİSİ) ---\n{draft_2}\n\n"
            f"--- AİLE 3 (LLAMA HATA & ÇELİŞKİ DENETÇİSİ) ---\n{draft_3}\n\n"
            "Nihai Sentez Yanıtı:"
        )

        final_answer, src_arb = self._call_specific_specialist(
            "synthesizer", arbiter_sys, synthesis_prompt, timeout_sec=12.0
        )
        if src_arb:
            sources_used.append(src_arb)

        if not final_answer:
            final_answer = draft_1 or draft_2 or draft_3 or "Analiz tamamlandı patron."

        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "response": final_answer,
            "source": f"MultiFamily_4Layer_Council ({' + '.join(sources_used[:4])} | {elapsed_ms}ms)",
            "deliberation_ms": elapsed_ms,
        }

    def sync_knowledge_to_colab_drive(self, graph_snapshot: Dict[str, Any]) -> bool:
        if self.firewall.strict_airgap_mode:
            return False
        colab_url = self.brain.config.get("colab_tunnel_url", "").rstrip("/")
        if not colab_url or not self.brain.cb_colab.can_attempt():
            return False
        try:
            r = requests.post(
                f"{colab_url}/nexus_sync_drive",
                json=graph_snapshot,
                timeout=4.0,
            )
            if r.status_code == 200:
                self.last_drive_sync_time = time.time()
                return True
        except Exception as e:
            logger.debug(f"Colab drive sync hatası: {e}")
        return False
