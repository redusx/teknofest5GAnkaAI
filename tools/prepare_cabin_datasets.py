#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Modül 2 (Kabin İçi Analiz) Veri Seti Uyumlaştırma Betigi

Bu betik ham veri setlerini yarışma kılavuzuna uygun formata dönüştürür:
  1. gelismisSurucuİzlemeSD       -> YOLO Detection (esneme, kemer, telefon)
  2. dmd                          -> YOLO Detection (sigara, telefon, kemer)
  3. Driver Behavior Image Dataset -> YOLO Detection (telefon, arkaya_bakma) - tam kare bbox

KURALLAR (FTR Kılavuz):
  - Tüm etiketler ASCII-safe ve küçük harfli
  - Hedef etiketler: arkaya_bakma(7), esneme(8), sigara_icme(9), su_icme(10),
                     telefonla_konusma(11), etrafa_bakinma(13), emniyet_kemeri_ihlali(14)
  - Orijinal veri setlerine DOKUNULMAZ (sadece kopyalama)
"""

import os
import sys
import shutil
import random
import logging
from pathlib import Path

# ==============================================================================
# Logger
# ==============================================================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cabin_harmonizer")

# ==============================================================================
# Sabitler
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "merged_cabin_monitoring"

RANDOM_SEED = 42

# --- gelismisSurucuİzlemeSD eşleme ---
# yawn(0)->esneme(8), seatbelt(2)->emniyet_kemeri_ihlali(14), mobile(3)->telefonla_konusma(11)
GELISMIS_MAP = {
    0: 8,
    2: 14,
    3: 11,
}

# --- dmd eşleme ---
# Cigarette(2)->sigara_icme(9), Phone(3)->telefonla_konusma(11), Seatbelt(4)->emniyet_kemeri_ihlali(14)
DMD_MAP = {
    2: 9,
    3: 11,
    4: 14,
}

# --- Driver Behavior Image Dataset eşleme ---
DRIVER_BEHAVIOR_MAP = {
    "talking_phone": 11,
    "texting_phone": 11,
    "turning": 7,
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path):
    """Dizin yoksa oluştur."""
    path.mkdir(parents=True, exist_ok=True)


def is_image(filename: str) -> bool:
    """Dosya bir görüntü mü?"""
    return Path(filename).suffix.lower() in IMG_EXTENSIONS


def find_image_for_label(lbl_path: Path, img_dir: Path):
    """Etiket dosyasına karsılık gelen görüntü dosyasını bulur."""
    stem = lbl_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ==============================================================================
# ADIM 1: gelismisSurucuİzlemeSD -> YOLO Detection
# ==============================================================================
def convert_gelismis_surucu():
    """
    gelismisSurucuİzlemeSD veri setindeki etiketleri standart class ID'lerine
    çevirir. Train setinin %10'u valid olarak ayrılır.
    """
    logger.info("=" * 60)
    logger.info("ADIM 1: gelismisSurucuİzlemeSD -> YOLO Detection")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "gelismisSurucuİzlemeSD" / "Yolo Annotated Dataset"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadı: {src_dir}")
        return

    random.seed(RANDOM_SEED)
    total_copied = 0

    # Train işle ve %90 train / %10 valid olarak böl
    train_lbl_dir = src_dir / "Train" / "labels"
    train_img_dir = src_dir / "Train" / "images"

    if train_lbl_dir.exists() and train_img_dir.exists():
        lbl_files = sorted(list(train_lbl_dir.glob("*.txt")))
        lbl_files = [f for f in lbl_files if f.name.lower() != "classes.txt"]

        random.shuffle(lbl_files)
        n_val = int(len(lbl_files) * 0.1)
        val_files = set(lbl_files[:n_val])

        for lbl_file in lbl_files:
            dst_split = "valid" if lbl_file in val_files else "train"
            img_out = OUTPUT_DIR / dst_split / "images"
            lbl_out = OUTPUT_DIR / dst_split / "labels"
            ensure_dir(img_out)
            ensure_dir(lbl_out)

            converted_lines = []
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        old_id = int(parts[0])
                        if old_id in GELISMIS_MAP:
                            new_id = GELISMIS_MAP[old_id]
                            converted_lines.append(f"{new_id} {' '.join(parts[1:])}")
                    except ValueError:
                        continue

            if not converted_lines:
                continue

            img_file = find_image_for_label(lbl_file, train_img_dir)
            if not img_file:
                continue

            new_name = f"gel_{img_file.name}"
            new_stem = f"gel_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)
            (lbl_out / f"{new_stem}.txt").write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            total_copied += 1

        logger.info(f"  Train/Valid bölündü: {total_copied - n_val} train, {n_val} valid")

    # Test işle
    test_lbl_dir = src_dir / "Test" / "labels"
    test_img_dir = src_dir / "Test" / "images"
    test_count = 0

    if test_lbl_dir.exists() and test_img_dir.exists():
        img_out = OUTPUT_DIR / "test" / "images"
        lbl_out = OUTPUT_DIR / "test" / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for lbl_file in sorted(test_lbl_dir.glob("*.txt")):
            if lbl_file.name.lower() == "classes.txt":
                continue

            converted_lines = []
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        old_id = int(parts[0])
                        if old_id in GELISMIS_MAP:
                            new_id = GELISMIS_MAP[old_id]
                            converted_lines.append(f"{new_id} {' '.join(parts[1:])}")
                    except ValueError:
                        continue

            if not converted_lines:
                continue

            img_file = find_image_for_label(lbl_file, test_img_dir)
            if not img_file:
                continue

            new_name = f"gel_{img_file.name}"
            new_stem = f"gel_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)
            (lbl_out / f"{new_stem}.txt").write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            test_count += 1

        logger.info(f"  Test: {test_count} görsel kopyalandı")
        total_copied += test_count

    logger.info(f"  gelismisSurucuİzlemeSD TOPLAM: {total_copied} görsel alindi")
    logger.info("")


# ==============================================================================
# ADIM 2: dmd -> YOLO Detection
# ==============================================================================
def convert_dmd():
    """
    dmd veri setindeki etiketleri standart class ID'lerine çevirir.
    Mevcut train, valid, test ayrımlarını korur.
    """
    logger.info("=" * 60)
    logger.info("ADIM 2: dmd -> YOLO Detection")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "dmd"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadı: {src_dir}")
        return

    split_map = {"train": "train", "valid": "valid", "test": "test"}
    total_copied = 0

    for src_split, dst_split in split_map.items():
        lbl_dir = src_dir / src_split / "labels"
        img_dir = src_dir / src_split / "images"

        if not lbl_dir.exists() or not img_dir.exists():
            continue

        img_out = OUTPUT_DIR / dst_split / "images"
        lbl_out = OUTPUT_DIR / dst_split / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        split_count = 0
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            converted_lines = []
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        old_id = int(parts[0])
                        if old_id in DMD_MAP:
                            new_id = DMD_MAP[old_id]
                            converted_lines.append(f"{new_id} {' '.join(parts[1:])}")
                    except ValueError:
                        continue

            if not converted_lines:
                continue

            img_file = find_image_for_label(lbl_file, img_dir)
            if not img_file:
                continue

            new_name = f"dmd_{img_file.name}"
            new_stem = f"dmd_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)
            (lbl_out / f"{new_stem}.txt").write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            split_count += 1

        logger.info(f"  {src_split} -> {dst_split}: {split_count} görsel dönüştürüldü")
        total_copied += split_count

    logger.info(f"  dmd TOPLAM: {total_copied} görsel alindi")
    logger.info("")


# ==============================================================================
# ADIM 3: Driver Behavior Image Dataset -> Tam Kare YOLO Detection
# ==============================================================================
def convert_driver_behavior():
    """
    Klasor bazlı sınıflandırma veri setini tam kare (0.5 0.5 1.0 1.0)
    bounding box detection formatına dönüştürür.
    %80 train / %10 valid / %10 test olarak böler.
    """
    logger.info("=" * 60)
    logger.info("ADIM 3: Driver Behavior Image Dataset -> YOLO Detection")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "Driver Behavior Image Dataset" / "Revitsone-5classes"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadı: {src_dir}")
        return

    pairs = []
    for folder_name, class_id in DRIVER_BEHAVIOR_MAP.items():
        folder_path = src_dir / folder_name
        if not folder_path.exists():
            logger.warning(f"Sınıf klasörü bulunamadı: {folder_path}")
            continue

        for img_file in sorted(folder_path.iterdir()):
            if img_file.is_file() and is_image(img_file.name):
                pairs.append((img_file, class_id, folder_name))

    logger.info(f"  Toplanan toplam görsel: {len(pairs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test":  pairs[n_train + n_valid:],
    }

    total_copied = 0
    for split_name, split_pairs in splits.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for img_file, class_id, folder_name in split_pairs:
            new_name = f"drv_{folder_name}_{img_file.name}"
            new_stem = Path(new_name).stem

            shutil.copy2(img_file, img_out / new_name)
            label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
            (lbl_out / f"{new_stem}.txt").write_text(label_content, encoding="utf-8")

        logger.info(f"  {split_name}: {len(split_pairs)} görsel kopyalandı")
        total_copied += len(split_pairs)

    logger.info(f"  Driver Behavior TOPLAM: {total_copied} görsel alindi")
    logger.info("")


# ==============================================================================
# ADIM 3.1: Teknocan Veri Setleri -> YOLO Detection
# ==============================================================================
def convert_teknocan_objects():
    """Teknocan etiketlerini (Class 0) standart Class ID 15'e dönüştürür."""
    logger.info("=" * 60)
    logger.info("ADIM 3.1: Teknocan Veri Setleri -> Class ID 15")
    logger.info("=" * 60)

    dirs_to_check = [
        DATASETS_DIR / "sadece_teknocan_dataset-20260627T100238Z-3-001" / "sadece_teknocan_dataset",
        DATASETS_DIR / "teknocan-dataset" / "teknocan-dataset"
    ]

    pairs = []
    for d in dirs_to_check:
        img_dir = d / "images" / "train"
        lbl_dir = d / "labels" / "train"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for lbl_file in sorted(lbl_dir.iterdir()):
            if lbl_file.suffix != ".txt":
                continue
            stem = lbl_file.stem
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    pairs.append((candidate, lbl_file))
                    break

    if not pairs:
        return

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test":  pairs[n_train + n_valid:],
    }

    for split_name, split_pairs in splits.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for img_file, lbl_file in split_pairs:
            new_name = f"tkn_{img_file.name}"
            new_stem = f"tkn_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)

            new_lines = []
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    new_lines.append(f"15 {' '.join(parts[1:])}")

            (lbl_out / f"{new_stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    logger.info(f"  Teknocan uyumlaştırıldı: {len(pairs)} görsel.")
    logger.info("")


# ==============================================================================
# ADIM 3.2: Bilgisayar Veri Seti -> YOLO Detection
# ==============================================================================
def convert_bilgisayar_objects():
    """Bilgisayar etiketlerini (Class 0) standart Class ID 16'ya dönüştürür."""
    logger.info("=" * 60)
    logger.info("ADIM 3.2: Bilgisayar Veri Seti -> Class ID 16")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "bilgisayar-dataset" / "bilgisayar-dataset"
    if not src_dir.exists():
        return

    split_map = {"train": "train", "valid": "valid", "test": "test"}

    for src_split, dst_split in split_map.items():
        img_dir = src_dir / src_split / "images"
        lbl_dir = src_dir / src_split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_out = OUTPUT_DIR / dst_split / "images"
        lbl_out = OUTPUT_DIR / dst_split / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        count = 0
        for lbl_file in sorted(lbl_dir.iterdir()):
            if lbl_file.suffix != ".txt":
                continue
            stem = lbl_file.stem
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            new_name = f"blg_{img_file.name}"
            new_stem = f"blg_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)

            new_lines = []
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    new_lines.append(f"16 {' '.join(parts[1:])}")

            (lbl_out / f"{new_stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            count += 1

        logger.info(f"  Bilgisayar {dst_split}: {count} görsel.")

    logger.info("")


# ==============================================================================
# ADIM 4: data.yaml Oluştur
# ==============================================================================
def create_data_yaml():
    """YOLO eğitim YAML dosyasını oluşturur."""
    logger.info("=" * 60)
    logger.info("ADIM 4: data.yaml Oluşturuluyor")
    logger.info("=" * 60)

    abs_path = OUTPUT_DIR.as_posix()
    yaml_content = f"""# ==============================================================================
# TEKNOFEST 2026 - ANKAAI
# Birleşik Kabin İçi Analiz Veri Seti (merged_cabin_monitoring)
# ==============================================================================

path: {abs_path}
train: train/images
val: valid/images
test: test/images

nc: 21

names:
  0: sedan
  1: suv
  2: hatchback
  3: pickup
  4: minibus
  5: panelvan
  6: kamyon
  7: arkaya_bakma
  8: esneme
  9: sigara_icme
  10: su_icme
  11: telefonla_konusma
  12: slalom
  13: etrafa_bakinma
  14: emniyet_kemeri_ihlali
  15: teknocan
  16: bilgisayar
  17: arka_koltuk_1
  18: arka_koltuk_2
  19: on_koltuk
  20: plaka
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info(f"  data.yaml kaydedildi: {yaml_path}")
    logger.info("")


def print_summary():
    """Özet rapor yazdır."""
    logger.info("=" * 60)
    logger.info("SONUÇ İSTATİSTİKLERİ")
    logger.info("=" * 60)

    class_names = {
        7: "arkaya_bakma", 8: "esneme", 9: "sigara_icme",
        10: "su_icme", 11: "telefonla_konusma", 13: "etrafa_bakinma",
        14: "emniyet_kemeri_ihlali", 15: "teknocan", 16: "bilgisayar",
    }

    for split in ["train", "valid", "test"]:
        img_dir = OUTPUT_DIR / split / "images"
        lbl_dir = OUTPUT_DIR / split / "labels"
        if not img_dir.exists():
            continue

        img_cnt = sum(1 for f in img_dir.iterdir() if f.is_file())
        counts = {}
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != ".txt":
                continue
            for line in lbl_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        cid = int(line.split()[0])
                        counts[cid] = counts.get(cid, 0) + 1
                    except ValueError:
                        pass

        logger.info(f"  {split}: {img_cnt} görsel")
        for cid in sorted(counts.keys()):
            name = class_names.get(cid, f"class_{cid}")
            logger.info(f"    Class {cid} ({name}): {counts[cid]}")

    logger.info("=" * 60)
    logger.info("Veri seti hazırlama başarıyla tamamlandı!")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("TEKNOFEST 2026 - ANKAAI")
    logger.info("Modül 2 Kabin Veri Seti Birleştirici Başlatılıyor...")
    logger.info("=" * 60)
    logger.info("")

    if OUTPUT_DIR.exists():
        logger.info(f"Eski çıktı dizini temizleniyor: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    ensure_dir(OUTPUT_DIR)

    convert_gelismis_surucu()
    convert_dmd()
    convert_driver_behavior()
    convert_teknocan_objects()
    convert_bilgisayar_objects()
    create_data_yaml()
    print_summary()


if __name__ == "__main__":
    main()
