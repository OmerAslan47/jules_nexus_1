from core.paths import NEXUS_ROOT
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.self_evolving_memory_graph import SelfEvolvingMemoryGraph
from core.multi_model_council import MultiModelCouncil
from core.jarvis_computation_engine import JarvisComputationEngine
from core.privacy_airgap_firewall import PrivacyAirGapFirewall
from core.semantic_response_cache import SemanticResponseCache


LOG_DIR = (NEXUS_ROOT + "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.CognitiveEngine")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "cognitive_reasoning.log"), encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)


class CognitiveReasoningEngine:
    """
    NEXUS v4.0 - 4 Vitesli J.A.R.V.I.S. & Çoklu YZ Ailesi Bilişsel Düşünme Motoru:
    - VİTES 0 (Anlık Refleks & J.A.R.V.I.S. Matematik/Mühendislik Hesaplayıcı - <10ms):
      * Selamlaşma, sahip tanıma ve matematik/VRAM/yüzde hesaplamalarında sıfır gecikmeli yanıt.
    - VİTES 1 (Kısa Sorular İçin Ultra-Hızlı Keskin Yanıt - <0.5 sn):
      * 9 kelimenin altındaki kısa ve pratik sorularda tek adımlı maksimum hızda yanıt.
    - VİTES 2 (Katmanlı Doğrulama Konseyi - 2 Farklı YZ Ailesi):
      * 1. Aile cevabı üretir, 2. Aile doğruluğunu, mantığını ve akıcılığını denetler.
    - VİTES 3 (Derin Konsey - 4 Farklı YZ Ailesi Paralel + Baş Hakem):
      * DeepSeek + Qwen-Coder + Llama-3 paralel çalışır, Baş Hakem sentezler.
    """

    DEEP_THINK_TRIGGERS = (
        "derin düşün",
        "uzun düşün",
        "detaylı analiz",
        "derinlemesine analiz",
        "konseyi topla",
        "mimari kur",
        "mimarisini tasarla",
        "en ince ayrıntısına",
        "kapsamlı analiz",
        "adım adım çöz",
        "karşılaştırmalı analiz",
        "nedenleriyle açıkla",
    )

    COMPLEX_DOMAIN_KEYWORDS = (
        "algoritma",
        "mimari",
        "optimizasyon",
        "veritabanı",
        "dağıtık sistem",
        "kriptografi",
        "yapay zeka",
        "sinir ağı",
        "performans darboğazı",
        "güvenlik açığı",
        "refactoring",
        "tasarım deseni",
        "diferansiyel",
        "olasılık",
        "strateji",
    )

    INSTANT_CONVERSATIONAL_REFLEXES = {
        "nasılsın": "Gayet iyiyim patron, tüm sistemlerim, görüntü tanıma çekirdeğim ve çoklu model konseyim tam kapasite çalışıyor. Sen nasılsın, bugün ne üzerinde çalışıyoruz?",
        "nasilsin": "Gayet iyiyim patron, tüm sistemlerim, görüntü tanıma çekirdeğim ve çoklu model konseyim tam kapasite çalışıyor. Sen nasılsın, bugün ne üzerinde çalışıyoruz?",
        "naber": "Her şey yolunda patron, seni dinliyorum. Bugün hangi projeye veya analize odaklanalım?",
        "ne haber": "Her şey yolunda patron, seni dinliyorum. Bugün hangi projeye veya analize odaklanalım?",
        "merhaba": "Merhaba patron! Seni dinliyorum, bugün nasıl yardımcı olabilirim?",
        "selam": "Selam patron! Tüm çekirdeklerim hazır, seni dinliyorum.",
        "beni duyuyor musun": "Seni çok net duyuyorum patron. Mikrofon algılamam ve yerel nöral ses motorum tam senkronize çalışıyor.",
        "sesim geliyor mu": "Evet patron, sesin gayet net geliyor. Seni dinliyorum.",
        "orda mısın": "Buradayım patron, her zaman nöbetteyim. Ne yapmamı istersin?",
        "orada mısın": "Buradayım patron, her zaman nöbetteyim. Ne yapmamı istersin?",
        "beni tanıyor musun": "Seni çok iyi tanıyorum patron. RTX 3050 Ti donanımın, derin uzay temalı neon buz mavisi ve platin beyazı 3B Dünya vizyonun, çoklu model konseylerin ve tüm mühendislik tercihlerin bilişsel hafızamda kayıtlı.",
        "benim hakkımda ne biliyorsun": "RTX 3050 Ti ekran kartına, kapalı kutu güvenli bilişsel mimariye, çoklu yapay zeka konseylerine ve uzay temalı neon buz mavisi ile platin hatlara sahip 3B arayüz vizyonuna sahip olduğunu biliyorum patron.",
    }

    def __init__(
        self,
        brain_router,
        memory_graph: Optional[SelfEvolvingMemoryGraph] = None,
        ui_state_callback: Optional[Callable[[str], None]] = None,
        firewall: Optional[PrivacyAirGapFirewall] = None,
        jarvis_engine: Optional[JarvisComputationEngine] = None,
    ):
        self.brain = brain_router
        self.graph = memory_graph or SelfEvolvingMemoryGraph()
        self.firewall = firewall or PrivacyAirGapFirewall()
        self.jarvis = jarvis_engine or JarvisComputationEngine()
        self.semantic_cache = SemanticResponseCache()
        self.council = MultiModelCouncil(brain_router, firewall=self.firewall)
        self.ui_state_callback = ui_state_callback

        self.system0_instant_calls = 0
        self.system1_fast_calls = 0
        self.system2_verified_calls = 0
        self.system3_deep_calls = 0

    @classmethod
    def analyze_query_complexity(cls, user_text: str) -> Tuple[str, float]:
        t = user_text.strip().lower()
        score = 1.0

        if any(trig in t for trig in cls.DEEP_THINK_TRIGGERS):
            return "SYSTEM_3_DEEP", 9.5

        word_count = len(t.split())
        if word_count > 28:
            score += 3.5
        elif word_count > 12:
            score += 2.0
        elif word_count <= 8:
            score += 0.2

        if t.count("?") >= 2:
            score += 1.5

        domain_hits = sum(1 for kw in cls.COMPLEX_DOMAIN_KEYWORDS if kw in t)
        score += domain_hits * 2.2

        if any(
            w in t
            for w in (
                "nasıl çalışır",
                "farkı nedir",
                "avantaj ve dezavantaj",
                "hangisi daha iyi",
                "mimari",
                "karşılaştır",
            )
        ):
            score += 2.2

        score = min(10.0, round(score, 1))
        if score >= 6.0:
            return "SYSTEM_3_DEEP", score
        if word_count <= 9 and score < 3.2:
            return "SYSTEM_1_ULTRA_FAST", score
        return "SYSTEM_2_VERIFIED", score

    def _format_recent_history(self, limit: int = 4) -> str:
        msgs = self.brain.conversation.get_formatted_messages(limit=limit)
        if not msgs:
            return "Henüz önceki mesaj yok."
        lines = []
        for m in msgs:
            r = "Kullanıcı" if m["role"] == "user" else "NEXUS"
            lines.append(f"{r}: {m['content']}")
        return "\n".join(lines)

    def think_and_respond(
        self,
        user_text: str,
        base_system_prompt: str,
        active_window_info: str = "",
        has_selected_tools: bool = False,
        record_history: bool = True,
    ) -> Dict[str, Any]:
        """
        4 Vitesli Bilişsel Akış:
        - Vites 0: Anlık Refleks & J.A.R.V.I.S. Matematik/VRAM Hesaplayıcı (<10ms)
        - Vites 1: Kısa Sorular İçin Ultra-Hızlı Yanıt (<0.5s)
        - Vites 2: Katmanlı Doğrulama Konseyi (Mantık -> Doğruluk/Hata Denetimi)
        - Vites 3: 4-Katmanlı Çoklu YZ Ailesi Derin Düşünme Konseyi
        """
        self.jarvis.observe_user_interaction(user_text)

        # 0-A: J.A.R.V.I.S. Anlık Matematik / Mühendislik / VRAM Hesaplaması (<5ms)
        calc_reply = self.jarvis.try_instant_jarvis_calculation(user_text)
        if calc_reply:
            self.system0_instant_calls += 1
            if record_history:
                self.brain.conversation.add_exchange(user_text, calc_reply)
                self.graph.record_episode(
                    user_text, calc_reply, reasoning_depth="jarvis_instant_math", utility_score=2.0
                )
            return {
                "response": calc_reply,
                "source": "JARVIS_Computation_Engine (<5ms)",
                "cognitive_gear": "SYSTEM_0_JARVIS_CALC",
                "complexity_score": 1.0,
            }

        # 0-B: Anlık Sohbet Refleksi (<10ms)
        normalized_q = re.sub(r"[^\w\sçğıöşü]", "", user_text.strip().lower()).strip()
        if normalized_q.startswith("nexus "):
            normalized_q = normalized_q[6:].strip()

        if normalized_q in self.INSTANT_CONVERSATIONAL_REFLEXES:
            self.system0_instant_calls += 1
            instant_reply = self.INSTANT_CONVERSATIONAL_REFLEXES[normalized_q]
            if record_history:
                self.brain.conversation.add_exchange(user_text, instant_reply)
                self.graph.record_episode(
                    user_text,
                    instant_reply,
                    reasoning_depth="system0_instant_reflex",
                    utility_score=1.5,
                )
            return {
                "response": instant_reply,
                "source": "Sistem-0 Anlık Refleks (<10ms)",
                "cognitive_gear": "SYSTEM_1_REFLEX",
                "complexity_score": 0.5,
            }

        # 0-C: Yerel Semantik Vektör Önbelleği (<3ms - Çevrimdışı ve Tekrarlı Sorular İçin)
        if not has_selected_tools:
            sem_hit = self.semantic_cache.lookup(user_text)
            if sem_hit:
                cached_reply, sim_score, orig_src = sem_hit
                self.system0_instant_calls += 1
                if record_history:
                    self.brain.conversation.add_exchange(user_text, cached_reply)
                return {
                    "response": cached_reply,
                    "source": f"Semantic_Vector_Cache (sim={sim_score} | <3ms | {orig_src})",
                    "cognitive_gear": "SYSTEM_0_SEMANTIC_CACHE",
                    "complexity_score": 1.0,
                }

        mode, complexity = self.analyze_query_complexity(user_text)
        graph_context = self.graph.recall_associative_context(user_text, max_items=7)
        owner_dossier = self.jarvis.get_owner_dossier_for_prompt()
        history_text = self._format_recent_history(limit=4)

        # Araç seçimi varsa hızlı araç çağrısı kontrolü
        if has_selected_tools:
            self.system1_fast_calls += 1
            enriched_system_prompt = (
                f"{base_system_prompt}\n"
                f"--- SAHİP PROFİLİ ---\n{owner_dossier}\n"
                f"--- KULLANICI BİLGİ GRAFİĞİ ---\n{graph_context}\n"
            )
            res = self.brain.ask(
                prompt=user_text,
                system_prompt=enriched_system_prompt,
                use_cache=False,
                record_history=record_history,
            )
            reply = res.get("response", "")
            if "TOOL_CALL:" in reply:
                res["cognitive_gear"] = "SYSTEM_1_TOOL"
                res["complexity_score"] = complexity
                return res
            if mode == "SYSTEM_1_ULTRA_FAST" and reply:
                res["cognitive_gear"] = "SYSTEM_1_REFLEX"
                res["complexity_score"] = complexity
                return res

        # VİTES 1: KISA SORULAR İÇİN MAKSİMUM HIZDA (<0.5 SN) YANIT
        if mode == "SYSTEM_1_ULTRA_FAST":
            self.system1_fast_calls += 1
            fast_res = self.council.run_ultra_fast_short_query(
                user_query=user_text,
                owner_dossier=owner_dossier,
                personal_graph_context=graph_context,
                conversation_history_text=history_text,
            )
            reply = fast_res["response"]
            if record_history:
                self.brain.conversation.add_exchange(user_text, reply)
                self.graph.record_episode(
                    user_text, reply, reasoning_depth="system1_ultra_fast", utility_score=1.5
                )
                self.semantic_cache.store(user_text, reply, fast_res["source"])
            return {
                "response": reply,
                "source": fast_res["source"],
                "cognitive_gear": "SYSTEM_1_REFLEX",
                "complexity_score": complexity,
            }

        # VİTES 3: 4-KATMANLI ÇOKLU YZ AİLESİ DERİN KONSEY
        if mode == "SYSTEM_3_DEEP":
            self.system3_deep_calls += 1
            if self.ui_state_callback:
                try:
                    self.ui_state_callback("DEEP_THINKING")
                except Exception as e:
                    logger.debug(f"UI durum bildirimi hatası (DEEP_THINKING): {e}")

            council_res = self.council.run_council_deliberation(
                user_query=user_text,
                personal_graph_context=graph_context,
                active_window_info=active_window_info,
                conversation_history_text=history_text,
                owner_dossier=owner_dossier,
            )
            reply = council_res["response"]
            if record_history:
                self.brain.conversation.add_exchange(user_text, reply)
                self.graph.record_episode(
                    user_text,
                    reply,
                    reasoning_depth="system3_multifamily_council",
                    utility_score=2.5,
                )
                self.semantic_cache.store(user_text, reply, council_res["source"])
                snap = self.graph.export_graph_snapshot()
                self.council.sync_knowledge_to_colab_drive(snap)

            return {
                "response": reply,
                "source": council_res["source"],
                "cognitive_gear": "SYSTEM_2_DEEP",
                "complexity_score": complexity,
            }

        # VİTES 2: 2-KATMANLI DOĞRULUK, MANTIK VE HATA DENETİM KONSEYİ
        self.system2_verified_calls += 1
        verified_res = self.council.run_fast_verified_pipeline(
            user_query=user_text,
            personal_graph_context=graph_context,
            conversation_history_text=history_text,
            active_window_info=active_window_info,
            owner_dossier=owner_dossier,
        )
        reply = verified_res["response"]
        if record_history:
            self.brain.conversation.add_exchange(user_text, reply)
            self.graph.record_episode(
                user_text,
                reply,
                reasoning_depth="system2_layered_verified",
                utility_score=1.9,
            )
            self.semantic_cache.store(user_text, reply, verified_res["source"])
        return {
            "response": reply,
            "source": verified_res["source"],
            "cognitive_gear": "SYSTEM_1_VERIFIED",
            "complexity_score": complexity,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        snap = self.graph.export_graph_snapshot()
        return {
            "system0_instant_calls": self.system0_instant_calls,
            "system1_fast_calls": self.system1_fast_calls,
            "system2_verified_calls": self.system2_verified_calls,
            "system3_deep_council_calls": self.system3_deep_calls,
            "council_invocations": self.council.council_invocations,
            "semantic_cache": self.semantic_cache.get_stats(),
            "knowledge_graph_nodes": snap.get("total_nodes", 0),
            "knowledge_graph_edges": snap.get("total_edges", 0),
            "episodic_memories": snap.get("total_episodes", 0),
            "firewall": self.firewall.get_security_report(),
        }
