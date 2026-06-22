#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Ana Cikarim Hatti (Inference Pipeline)

4 Kademeli Kaskat Mikro-Model mimarisini orkestre eder:
  Modul 1: Kuresel tespit (arac, renk, nesneler, ROI cikarim)
  Modul 2: Kabin ici analiz (sofor eylemleri, yolcular)
  Modul 3: Plaka OCR (perspektif duzeltme, regex dogrulama)
  Modul 4: Yorunge takibi ve slalom tespiti

Kullanim (main.py tarafindan cagirilir):
    from src.pipeline import InferencePipeline
    pipeline = InferencePipeline(models_dir="/app/models/")
    output_data = pipeline.run(video_path="/app/data/input/video.mp4")
"""

import os
import logging
from typing import Dict, Any, Optional

import cv2
import numpy as np

from src.module1_global.detector import GlobalDetector
from src.module2_cabin.analyzer import CabinAnalyzer
from src.module3_ocr.plate_reader import PlateReader
from src.module4_trajectory.tracker import TrajectoryTracker
from src.postprocessor import PostProcessor

logger = logging.getLogger("pipeline")


# ==============================================================================
# Varsayilan Yapilandirma
# ==============================================================================
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_FRAME_SKIP = 3          # Her N karede bir isle (5-10 FPS hedefi)
DEFAULT_IMG_SIZE = 640


class InferencePipeline:
    """
    4 Kademeli Kaskat Cikarim Hatti.

    Video akisini okuyup, 4 modulu sirali olarak calistirarak
    yarisma standartlarina uygun JSON ciktisi uretir.

    Kosullu Dallanma (Conditional Execution):
      - On cam tespit edilmediginde -> Modul 2 CALISMAZ
      - Plaka tespit edilmediginde  -> Modul 3 CALISMAZ
      Bu sayede gereksiz hesaplama onlenir.

    Attributes:
        module1: Kuresel tespit modulu (GlobalDetector).
        module2: Kabin analiz modulu (CabinAnalyzer).
        module3: Plaka OCR modulu (PlateReader).
        module4: Yorunge takip modulu (TrajectoryTracker).
        postprocessor: Son islem motoru (PostProcessor).
    """

    def __init__(
        self,
        models_dir: str = "/app/models/",
        device: str = "auto",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        img_size: int = DEFAULT_IMG_SIZE,
        use_tracker: bool = True,
        shared_model: bool = True,
    ):
        """
        Args:
            models_dir: Model agirliklari dizini.
            device: Cihaz secimi ("auto", "cuda", "cpu").
            conf_threshold: Minimum guven esigi.
            iou_threshold: NMS IOU esigi.
            frame_skip: Kare atlama miktari.
            img_size: YOLO girdi boyutu.
            use_tracker: ByteTrack takip algoritmasi kullan.
            shared_model: True ise tek model tum moduller tarafindan paylasılır.
                          False ise her modul kendi modelini yukler.
        """
        self.models_dir = models_dir
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.use_tracker = use_tracker
        self.shared_model = shared_model

        # Modulleri baslat
        self.module1: Optional[GlobalDetector] = None
        self.module2: Optional[CabinAnalyzer] = None
        self.module3: Optional[PlateReader] = None
        self.module4: Optional[TrajectoryTracker] = None
        self.postprocessor = PostProcessor()

    # ==================================================================
    # Model Yukleme
    # ==================================================================
    def _find_model_file(self, prefix: str = "") -> Optional[str]:
        """
        models/ dizininde uygun model dosyasini arar.

        Arama onceligi: .engine > .onnx > .pt

        Args:
            prefix: Dosya adi oneki (orn: "global_", "cabin_").

        Returns:
            Model dosyasi yolu veya None.
        """
        if not os.path.exists(self.models_dir):
            logger.warning(f"Model dizini bulunamadi: {self.models_dir}")
            return None

        # Oncelik sirasiyla uzantilari tara
        for ext in [".engine", ".onnx", ".pt"]:
            for f in sorted(os.listdir(self.models_dir)):
                if f.startswith(prefix) and f.endswith(ext):
                    return os.path.join(self.models_dir, f)

        # Prefix olmadan herhangi bir model dosyasi ara
        for ext in [".engine", ".onnx", ".pt"]:
            for f in sorted(os.listdir(self.models_dir)):
                if f.endswith(ext):
                    return os.path.join(self.models_dir, f)

        return None

    def load_models(self):
        """
        Tum modullerin modellerini yukler.

        shared_model=True ise tek model dosyasi tum moduller tarafindan
        kullanilir (gelistirme asamasi). False ise her modul kendi model
        dosyasini yukler (uretim asamasi).
        """
        logger.info("=" * 60)
        logger.info("Modeller yukleniyor...")

        if self.shared_model:
            # --- Tek Paylasimli Model ---
            model_path = self._find_model_file()
            if model_path is None:
                raise FileNotFoundError(
                    f"Model dosyasi bulunamadi: {self.models_dir}"
                )

            logger.info(f"Paylasimli model: {model_path}")

            # Modul 1 modeli yukler, Modul 2 paylasir
            self.module1 = GlobalDetector(
                model_path=model_path,
                device=self.device,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                img_size=self.img_size,
            )

            # Modul 2 paylasimli modda calisir (YOLO sonuclari dogrudan aktarilir)
            self.module2 = CabinAnalyzer(
                device=self.device,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                img_size=self.img_size,
            )

        else:
            # --- Bagimsiz Modeller ---
            # Modul 1: YOLOv12s
            m1_path = self._find_model_file("global_")
            if m1_path:
                self.module1 = GlobalDetector(
                    model_path=m1_path,
                    device=self.device,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    img_size=self.img_size,
                )

            # Modul 2: YOLOv12n
            m2_path = self._find_model_file("cabin_")
            if m2_path:
                self.module2 = CabinAnalyzer(
                    model_path=m2_path,
                    device=self.device,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    img_size=self.img_size,
                )

        # Modul 3: Plaka OCR (model bagimsiz baslar)
        self.module3 = PlateReader()

        # Modul 4: Yorunge takip (model gerektirmez, CPU tabanli)
        self.module4 = TrajectoryTracker()

        logger.info("Tum moduller basariyla baslatildi.")
        logger.info("=" * 60)

    # ==================================================================
    # Ana Cikarim Dongusu
    # ==================================================================
    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Video uzerinde 4 kademeli kaskat cikarim yapar.

        Islem Adimlari:
          1. Video dosyasini ac ve meta bilgilerini al
          2. Modelleri yukle
          3. Her N karede:
             a. Modul 1: Kuresel tespit (arac, ROI)
             b. Modul 2: Kabin analizi (kosullu - on cam varsa)
             c. Modul 3: Plaka OCR (kosullu - plaka ROI varsa)
             d. Modul 4: Yorunge guncelleme
          4. Son islem: Zamansal NMS, confidence toplama, JSON uretimi

        Args:
            video_path: Girdi video dosyasi yolu.

        Returns:
            Yarisma formatinda JSON-uyumlu sozluk.
        """
        # --- Video dosyasini kontrol et ---
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video bulunamadi: {video_path}")

        video_id = os.path.basename(video_path)
        logger.info(f"[Pipeline] {video_path} analiz ediliyor...")

        # --- Modelleri yukle ---
        self.load_models()

        # --- Video'yu ac ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30.0
            logger.warning(f"FPS okunamadi, varsayilan {fps} kullanilacak.")

        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"  Video: {width}x{height} @ {fps} FPS")
        logger.info(f"  Toplam kare: {total_frames} ({duration:.1f} sn)")
        logger.info(f"  Kare atlama: her {self.frame_skip}. kare isleniyor")

        # --- Cikarim Dongusu ---
        frame_count = 0
        processed_count = 0
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Kare atlama
            if frame_count % self.frame_skip != 0:
                continue

            frame_time = frame_count / fps

            try:
                # ================================================
                # MODUL 1: Kuresel Tespit
                # ================================================
                global_result = self.module1.detect(
                    frame,
                    frame_time=frame_time,
                    use_tracker=self.use_tracker,
                )

                # Nesne tespitlerini (teknocan, bilgisayar) topla
                all_detections.extend(global_result.object_detections)

                # ================================================
                # MODUL 2: Kabin Ici Analiz (KOSULLU)
                # ================================================
                if self.shared_model:
                    # Paylasimli model: YOLO sonuclari dogrudan aktarilir
                    # (Modul 1'in YOLO sonuclari kullanilir)
                    cabin_dets = self.module2.analyze_with_shared_results(
                        yolo_result=self.module1.model.predictor.results[0]
                        if hasattr(self.module1.model, "predictor")
                        and self.module1.model.predictor
                        and self.module1.model.predictor.results
                        else None,
                        frame_time=frame_time,
                    )
                    all_detections.extend(cabin_dets)

                elif global_result.windshield_roi:
                    # Bagimsiz model: On cam kirpintisi ile analiz
                    wx1, wy1, wx2, wy2 = global_result.windshield_roi
                    windshield_crop = frame[wy1:wy2, wx1:wx2]

                    if windshield_crop.size > 0:
                        cabin_dets = self.module2.analyze(
                            windshield_crop,
                            frame_time=frame_time,
                            full_frame=frame,
                        )
                        all_detections.extend(cabin_dets)

                # ================================================
                # MODUL 3: Plaka OCR (KOSULLU)
                # ================================================
                if global_result.plate_roi and self.module3:
                    px1, py1, px2, py2 = global_result.plate_roi
                    plate_crop = frame[py1:py2, px1:px2]

                    if plate_crop.size > 0:
                        plate_text = self.module3.read(plate_crop)
                        if plate_text:
                            self.module1.add_plate_detection(
                                plate_text, global_result.plate_conf
                            )

                # ================================================
                # MODUL 4: Yorunge Takibi
                # ================================================
                if self.module4:
                    self.module4.update(
                        vehicle_bbox=global_result.vehicle_bbox,
                        frame_time=frame_time,
                    )

                processed_count += 1

                # Ilerleme raporu (her 100 karede bir)
                if processed_count % 100 == 0:
                    progress = (
                        (frame_count / total_frames * 100)
                        if total_frames > 0
                        else 0
                    )
                    logger.info(
                        f"  Ilerleme: {progress:.1f}% "
                        f"({frame_count}/{total_frames})"
                    )

            except Exception as e:
                logger.warning(f"  Kare {frame_count} hatasi: {e}")
                continue

        cap.release()

        logger.info(
            f"  Cikarim tamamlandi: {processed_count}/{total_frames} kare"
        )

        # ================================================
        # SON ISLEM (Post-Processing)
        # ================================================

        # Slalom olaylarini ekle
        if self.module4:
            slalom_events = self.module4.get_slalom_events()
            all_detections.extend(slalom_events)

        # Zamansal 1D NMS uygula
        filtered_detections = self.postprocessor.apply_temporal_nms(
            all_detections
        )

        # Arac bilgisini topla
        vehicle_info = self.postprocessor.aggregate_vehicle_info(
            type_votes=self.module1.get_type_votes(),
            color_votes=self.module1.get_color_votes(),
            plate_detections=self.module1.get_plate_detections(),
        )

        # Konsolide JSON cikti olustur
        output = self.postprocessor.build_output(
            video_id=video_id,
            vehicle_info=vehicle_info,
            detections=filtered_detections,
        )

        # Istatistikleri logla
        self._log_summary(vehicle_info, filtered_detections)

        return output

    # ==================================================================
    # Ozet Loglama
    # ==================================================================
    def _log_summary(
        self,
        vehicle_info: Dict[str, Any],
        detections: list,
    ):
        """Cikarim sonuclarinin ozetini loglar."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("CIKARIM OZETI")
        logger.info("=" * 60)
        logger.info(f"  Arac tipi    : {vehicle_info.get('tip', '-')}")
        logger.info(f"  Plaka        : {vehicle_info.get('plaka', '-')}")
        logger.info(f"  Renk         : {vehicle_info.get('renk', '-')}")
        logger.info(
            f"  Confidence   : {vehicle_info.get('confidence_score', 0):.4f}"
        )
        logger.info(f"  Tespit sayisi: {len(detections)}")

        # Kategori bazli ozet
        cats = {}
        for d in detections:
            cat = d.get("kategori", "diger")
            cats[cat] = cats.get(cat, 0) + 1

        for cat, count in cats.items():
            logger.info(f"    {cat}: {count}")

        # Modul 4 istatistikleri
        if self.module4:
            stats = self.module4.get_trajectory_stats()
            logger.info(
                f"  Yorunge: {stats['toplam_nokta']} nokta, "
                f"yanal_std={stats['yanal_std']:.1f}, "
                f"slalom={stats['slalom_sayisi']}"
            )

        logger.info("=" * 60)
