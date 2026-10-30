from core.paths import NEXUS_ROOT
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np


# ============================================================================
# LOGLAMA VE GÜVENLİK KLASÖRLERİ
# ============================================================================
LOG_DIR = (NEXUS_ROOT + "/logs")
SECURITY_SNAP_DIR = (NEXUS_ROOT + "/security_snapshots")
FACE_GALLERY_DIR = (NEXUS_ROOT + "/face_gallery")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SECURITY_SNAP_DIR, exist_ok=True)
os.makedirs(FACE_GALLERY_DIR, exist_ok=True)

logger = logging.getLogger("NEXUS.VisionAuth")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "vision_auth.log"), encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)


# ============================================================================
# ORTAM AYDINLATMA VE CANLILIK (ANTI-SPOOFING / LIVENESS) ANALİZCİSİ
# ============================================================================
class SceneAndLivenessAnalyzer:
    """
    Kamera karesinin ortam kalitesini ve canlılık (gerçek yüz vs telefon ekranı/fotoğraf)
    durumunu analiz eder:
    - Ortam parlaklık seviyesi (Gece/Karanlık/Normal/Parlak)
    - Laplacian netlik/bulanıklık skoru (Blur Detection)
    - Mikro-hareket canlılık testi (Ardışık 2 kare arasındaki doğal piksel değişimi)
    - Omuz Sörfü (Shoulder Surfing) Tespiti: Ekrana aynı anda 2 veya daha fazla kişi
      bakıyorsa gizlilik uyarısı üretir!
    """

    @staticmethod
    def evaluate_scene_lighting(gray: np.ndarray) -> Dict[str, Any]:
        mean_lum = float(np.mean(gray))
        std_contrast = float(np.std(gray))
        if mean_lum < 35.0:
            condition = "cok_karanlik"
        elif mean_lum < 80.0:
            condition = "los_isik"
        elif mean_lum > 215.0:
            condition = "asiri_parlak"
        else:
            condition = "ideal"

        return {
            "mean_luminance": round(mean_lum, 1),
            "contrast_std": round(std_contrast, 1),
            "lighting_condition": condition,
        }

    @staticmethod
    def compute_sharpness_score(gray_face: np.ndarray) -> float:
        """Yüz bölgesinin Laplacian varyansını (keskinlik/netlik) hesaplar."""
        if hasattr(cv2, "Laplacian"):
            try:
                lap = cv2.Laplacian(gray_face, cv2.CV_64F)
                return round(float(lap.var()), 2)
            except Exception as e:
                logger.debug(f"Laplacian var hesaplama hatasi: {e}")
        gx = np.diff(gray_face.astype(np.float32), axis=1)
        return round(float(np.var(gx)), 2)

    @staticmethod
    def verify_micro_motion_liveness(
        frame1_gray: np.ndarray, frame2_gray: np.ndarray
    ) -> Tuple[bool, float]:
        """
        İki ardışık kare arasındaki mikro-hareketi ölçer.
        Tamamen donuk bir fotoğraf (0.0 değişim) veya aşırı ekran titremesi varsa tespit eder.
        """
        if frame1_gray.shape != frame2_gray.shape:
            return True, 5.0
        diff = np.abs(frame1_gray.astype(np.float32) - frame2_gray.astype(np.float32))
        mean_diff = float(np.mean(diff))
        # Gerçek bir insan kameraya bakarken 0.3 ile 35.0 arasında mikro hareket üretir
        is_live = 0.25 <= mean_diff <= 45.0
        return is_live, round(mean_diff, 2)


# ============================================================================
# FAZ 3 KURUMSAL GÖRSEL BİYOMETRİ VE VARLIK TAKİP MOTORU (VISION AUTH V2)
# ============================================================================
class VisionAuth:
    """
    NEXUS Görsel Biyometri, Çoklu Açı Yüz Galerisi, Omuz Sörfü Koruması,
    Misafir Tespiti ve Otomatik Uyku Motoru (Kurumsal Tam Sürüm):
    - Kamera SÜREKLİ AÇIK KALMAZ; sadece kontrol anında milisaniyelik kare çekip kapanır.
    - Çoklu Açı Yüz Galerisi (Multi-Angle Face Gallery): Farklı ışık ve açılardan alınan sahip
      yüz vektörleriyle karşılaştırma yaparak hatalı misafir moduna geçmeyi önler.
    - Omuz Sörfü Algılama (Shoulder Surfing Alert): Kamerada birden fazla yüz görünürse
      arkadan ekrana bakan yabancı olabileceği için uyarı üretir.
    - Yabancı / Misafir Kaydı (Intruder Snapshot): Tanınmayan bir kişi bilgisayar başına
      oturduğunda security_snapshots/ klasörüne tarihli fotoğraf kaydeder.
    - Otomatik Uyku Modu (Sleep Mode): Masada kimse olmadığında sistemi düşük güç moduna alır.
    """

    def __init__(
        self,
        owner_face_path: str = (NEXUS_ROOT + "/owner_face.jpg"),
        owner_face_vec_path: str = (NEXUS_ROOT + "/owner_face_vec.npy"),
        check_interval_sec: int = 300,
        similarity_threshold: float = 0.78,
    ):
        self.owner_face_path = owner_face_path
        self.owner_face_vec_path = owner_face_vec_path
        self.check_interval_sec = check_interval_sec
        self.similarity_threshold = similarity_threshold

        self.scene_analyzer = SceneAndLivenessAnalyzer()
        self.last_check_time = 0.0
        self.user_present = True
        self.sleep_mode = False
        self.is_guest = False
        self.shoulder_surfing_detected = False
        self.detected_face_count = 0
        self.last_similarity_score = 1.0
        self.last_lighting_info: Dict[str, Any] = {}
        self.consecutive_absent_checks = 0

        self._lock = threading.Lock()
        self.presence_history: Deque[Dict[str, Any]] = deque(maxlen=120)
        self.intruder_alerts: List[Dict[str, Any]] = []

        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        data_obj = getattr(cv2, "data", None)
        if cascade_cls and data_obj:
            self._face_cascade = cascade_cls(
                data_obj.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            self._face_cascade = None

    # ========================================================================
    # GÖRÜNTÜ ÖN-İŞLEME VE 512-BOYUTLU ÖZNİTELİK VEKTÖRÜ ÇIKARMA
    # ========================================================================
    @staticmethod
    def _preprocess_illumination(gray_img: np.ndarray) -> np.ndarray:
        """Gece/gündüz farklı ışık koşullarını dengelemek için CLAHE veya Histogram Eşitleme uygular."""
        if hasattr(cv2, "createCLAHE"):
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(gray_img)
            except Exception as e:
                logger.debug(f"CLAHE uygulama hatasi: {e}")
        if hasattr(cv2, "equalizeHist"):
            try:
                return cv2.equalizeHist(gray_img)
            except Exception as e:
                logger.debug(f"equalizeHist uygulama hatasi: {e}")
        return gray_img

    @classmethod
    def _extract_face_embedding(cls, gray_face: np.ndarray) -> np.ndarray:
        """
        Yüz bölgesinden aydınlatmaya dayanıklı 512-boyutlu birleşik öznitelik vektörü çıkarır:
        - 256 boyut: 16x16 normalize uzamsal yoğunluk haritası
        - 256 boyut: Yatay ve dikey gradyan (kenar/kontur) dağılımı
        """
        resized = cv2.resize(gray_face, (64, 64))
        norm_light = cls._preprocess_illumination(resized).astype(np.float32)

        # 1. Uzamsal 16x16 öznitelik (256-dim)
        spatial = cv2.resize(norm_light, (16, 16)).flatten()

        # 2. Gradyan / Kontur özniteliği (256-dim)
        gx = np.diff(norm_light, axis=1, append=norm_light[:, -1:])
        gy = np.diff(norm_light, axis=0, append=norm_light[-1:, :])
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_feat = cv2.resize(grad_mag, (16, 16)).flatten()

        combined = np.concatenate([spatial, grad_feat]).astype(np.float32)
        combined -= np.mean(combined)
        norm = np.linalg.norm(combined)
        return combined / (norm + 1e-8)

    # ========================================================================
    # KAMERA YAKALAMA VE ÇOKLU YÜZ TESPİTİ
    # ========================================================================
    def capture_single_frame(
        self, save_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Kamerayı açar, birkaç ısınma karesinden sonra net bir kare çeker ve kamerayı hemen kapatır."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        frame = None
        for _ in range(3):
            ret, temp = cap.read()
            if ret and temp is not None:
                frame = temp
        cap.release()

        if frame is not None and save_path:
            try:
                cv2.imwrite(save_path, frame)
            except Exception as e:
                logger.warning(f"Kare kaydedilemedi: {e}")
        return frame

    def capture_frame_pair_for_liveness(
        self, delay_sec: float = 0.18
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Canlılık (Liveness) kontrolü için 180ms arayla iki hızlı kare yakalar ve kamerayı kapatır."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None, None
        ret1, f1 = cap.read()
        time.sleep(delay_sec)
        ret2, f2 = cap.read()
        cap.release()
        return (f1 if ret1 else None), (f2 if ret2 else None)

    def _detect_all_faces(
        self, gray: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Karedeki tüm yüzleri alan büyüklüğüne göre (en yakından uzağa) sıralı döndürür."""
        if self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48)
            )
            if len(faces) == 0:
                return []
            sorted_faces = sorted(
                faces, key=lambda rect: rect[2] * rect[3], reverse=True
            )
            return [
                (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
                for f in sorted_faces
            ]

        if gray.std() > 18.0:
            h, w = gray.shape[:2]
            return [(w // 4, h // 4, w // 2, h // 2)]
        return []

    # ========================================================================
    # SAHİP YÜZ KAYDI VE ÇOKLU AÇI GALERİSİ
    # ========================================================================
    def register_owner_face(self, sample_name: str = "primary") -> bool:
        """Sahibin yüz fotoğrafını ve 512-boyutlu referans vektörünü kaydeder."""
        frame = self.capture_single_frame(self.owner_face_path)
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_all_faces(gray)
        if faces:
            x, y, w, h = faces[0]
            face_crop = gray[y : y + h, x : x + w]
        else:
            face_crop = gray

        vec = self._extract_face_embedding(face_crop)
        np.save(self.owner_face_vec_path, vec)

        gallery_vec_path = os.path.join(
            FACE_GALLERY_DIR, f"owner_{sample_name}_{int(time.time())}.npy"
        )
        np.save(gallery_vec_path, vec)
        logger.info(f"Sahip yüz biyometrisi kaydedildi: {gallery_vec_path}")
        return True

    def _load_all_owner_vectors(self) -> List[np.ndarray]:
        """Ana yüz vektörünü ve face_gallery/ içindeki tüm sahip vektörlerini yükler."""
        vectors: List[np.ndarray] = []
        if os.path.exists(self.owner_face_vec_path):
            try:
                vectors.append(np.load(self.owner_face_vec_path))
            except Exception as e:
                logger.debug(f"Sahip face vec okuma hatasi: {e}")

        for fname in os.listdir(FACE_GALLERY_DIR):
            if fname.endswith(".npy"):
                try:
                    vectors.append(np.load(os.path.join(FACE_GALLERY_DIR, fname)))
                except Exception as e:
                    logger.debug(f"Galeri face vec okuma hatasi: {e}")
        return vectors

    def _record_intruder_snapshot(
        self, frame: np.ndarray, score: float, reason: str = "unknown_face"
    ):
        """Tanınmayan bir yüz veya omuz sörfü algılandığında güvenlik klasörüne görüntü kaydeder."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(SECURITY_SNAP_DIR, f"{reason}_{stamp}.jpg")
        try:
            cv2.imwrite(snap_path, frame)
            alert = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "similarity_score": round(score, 3),
                "snapshot_path": snap_path,
            }
            with self._lock:
                self.intruder_alerts.append(alert)
                self.intruder_alerts = self.intruder_alerts[-40:]
            logger.warning(f"Güvenlik anlık görüntüsü ({reason}) kaydedildi: {snap_path}")
        except Exception as e:
            logger.error(f"Güvenlik fotoğrafı kaydedilemedi: {e}")

    # ========================================================================
    # 5 DAKİKALIK VARLIK, KİMLİK VE OMUZ SÖRFÜ KONTROLÜ
    # ========================================================================
    def check_presence_and_identity(self, force: bool = False) -> Dict[str, Any]:
        """
        5 dakikada bir veya zorunlu çağrıda tek kare çekip:
        1. Masada birisi var mı? (user_present)
        2. Masadaki kişi sahip mi yoksa misafir mi? (is_guest)
        3. Arkadan ekrana bakan ikinci bir kişi var mı? (shoulder_surfing)
        4. Sistem uyku modunda mı? (sleep_mode)
        bilgilerini döndürür.
        """
        now = time.time()
        if not force and (now - self.last_check_time) < self.check_interval_sec:
            return {
                "user_present": self.user_present,
                "is_guest": self.is_guest,
                "sleep_mode": self.sleep_mode,
                "shoulder_surfing": self.shoulder_surfing_detected,
                "face_count": self.detected_face_count,
                "similarity_score": self.last_similarity_score,
            }

        self.last_check_time = now
        frame = self.capture_single_frame()
        if frame is None:
            return {
                "user_present": self.user_present,
                "is_guest": self.is_guest,
                "sleep_mode": self.sleep_mode,
                "shoulder_surfing": False,
                "face_count": 0,
                "similarity_score": self.last_similarity_score,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_lighting_info = self.scene_analyzer.evaluate_scene_lighting(gray)
        faces = self._detect_all_faces(gray)
        self.detected_face_count = len(faces)

        if len(faces) == 0:
            self.consecutive_absent_checks += 1
            self.user_present = False
            self.sleep_mode = True
            self.shoulder_surfing_detected = False
            self._log_presence_event(False, self.is_guest, 0.0, 0)
            return {
                "user_present": False,
                "is_guest": self.is_guest,
                "sleep_mode": True,
                "shoulder_surfing": False,
                "face_count": 0,
                "similarity_score": 0.0,
            }

        self.consecutive_absent_checks = 0
        self.user_present = True
        self.sleep_mode = False
        self.shoulder_surfing_detected = len(faces) > 1

        if self.shoulder_surfing_detected:
            self._record_intruder_snapshot(
                frame, self.last_similarity_score, reason="shoulder_surfing"
            )

        owner_vectors = self._load_all_owner_vectors()
        if owner_vectors:
            try:
                x, y, w, h = faces[0]
                face_crop = gray[y : y + h, x : x + w]
                cur_vec = self._extract_face_embedding(face_crop)
                valid_scores = [
                    float(np.dot(ov, cur_vec))
                    for ov in owner_vectors
                    if ov.shape == cur_vec.shape
                ]
                best_sim = max(valid_scores) if valid_scores else 1.0
                self.last_similarity_score = round(best_sim, 3)
                self.is_guest = best_sim < self.similarity_threshold
                if self.is_guest:
                    self._record_intruder_snapshot(frame, best_sim, reason="guest_detected")
            except Exception as e:
                logger.warning(f"Yüz karşılaştırma uyarısı: {e}")
                self.is_guest = False
        else:
            self.is_guest = False
            self.last_similarity_score = 1.0

        self._log_presence_event(
            True, self.is_guest, self.last_similarity_score, len(faces)
        )
        return {
            "user_present": self.user_present,
            "is_guest": self.is_guest,
            "sleep_mode": self.sleep_mode,
            "shoulder_surfing": self.shoulder_surfing_detected,
            "face_count": len(faces),
            "similarity_score": self.last_similarity_score,
        }

    def _log_presence_event(
        self, present: bool, is_guest: bool, score: float, face_count: int
    ):
        with self._lock:
            self.presence_history.append(
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "present": present,
                    "is_guest": is_guest,
                    "face_count": face_count,
                    "score": score,
                }
            )

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "user_present": self.user_present,
                "sleep_mode": self.sleep_mode,
                "is_guest": self.is_guest,
                "shoulder_surfing_detected": self.shoulder_surfing_detected,
                "detected_face_count": self.detected_face_count,
                "last_similarity_score": self.last_similarity_score,
                "lighting": self.last_lighting_info,
                "enrolled_gallery_samples": len(os.listdir(FACE_GALLERY_DIR)),
                "recent_intruder_alerts": list(self.intruder_alerts)[-5:],
                "recent_presence_checks": list(self.presence_history)[-10:],
            }
