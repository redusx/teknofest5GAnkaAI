#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Veri Seti Uyumlastirma Betigi (Dataset Harmonization Script)

Bu betik 4 ham veri setini yarisma kilavuzuna uygun formata donusturur:
  1. Cars_Body_Type       -> YOLO Detection (merged_detection)
  2. vehicles.v2          -> YOLO Detection (merged_detection) - secici filtreleme
  3. plateRecognition     -> YOLO Detection (merged_detection) - ID donusumu + split
  4. colorRecognition     -> YOLO Classification (color_classification) - Turkce isimler

KURALLAR (FTR Kilavuz):
  - Tum etiketler ASCII-safe ve kucuk harfli
  - Arac tipleri: sedan, suv, hatchback, pickup, minibus, panelvan, kamyon
  - Renkler: beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi
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
logger = logging.getLogger("dataset_harmonizer")

# ==============================================================================
# Sabitler
# ==============================================================================
DATASETS_DIR = Path(r"c:\Users\red\Desktop\RED\5GTeknofest\ankaai\datasets")

# Cikti dizinleri
MERGED_DIR = DATASETS_DIR / "merged_detection"
COLOR_DIR = DATASETS_DIR / "color_classification"

# Rastgele seed (tekrarlanabilirlik)
RANDOM_SEED = 42

# --- Cars_Body_Type sinif esleme ---
CBT_CLASS_MAP = {
    "Sedan":       0,   # sedan
    "SUV":         1,   # suv
    "Hatchback":   2,   # hatchback
    "Pick-Up":     3,   # pickup
    "VAN":         5,   # panelvan
    "Coupe":       0,   # -> sedan (kilavuzda coupe yok)
    "Convertible": 0,   # -> sedan (kilavuzda convertible yok)
}

# --- vehicles.v2 sinif esleme (sadece bu ID'ler alinacak) ---
VEH_CLASS_MAP = {
    0: 4,   # big bus    -> minibus
    1: 6,   # big truck  -> kamyon
    6: 4,   # small bus  -> minibus
    7: 6,   # small truck -> kamyon
}

# --- plateRecognition sinif esleme ---
PLATE_CLASS_MAP = {
    0: 20,  # plaka -> Class ID 20
}

# --- colorRecognition renk esleme (Ingilizce -> Turkce ASCII-safe) ---
COLOR_MAP = {
    "black":  "siyah",
    "white":  "beyaz",
    "grey":   "gri",
    "silver": "gri",        # gri ile birlestir
    "red":    "kirmizi",
    "blue":   "mavi",
    "yellow": "sari",
    "gold":   "sari",       # sari ile birlestir
    "green":  "yesil",
    "orange": "turuncu",
    "brown":  "kahverengi",
    "tan":    "kahverengi",  # kahverengi ile birlestir
    "beige":  "kahverengi",  # kahverengi ile birlestir
}
# pink ve purple CIKARILACAK (kilavuzda yok)
COLOR_EXCLUDE = {"pink", "purple"}

# Goruntu uzantilari
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path):
    """Dizin yoksa olustur."""
    path.mkdir(parents=True, exist_ok=True)


def is_image(filename: str) -> bool:
    """Dosya bir goruntu mu?"""
    return Path(filename).suffix.lower() in IMG_EXTENSIONS


# ==============================================================================
# ADIM 1: Cars_Body_Type -> YOLO Detection
# ==============================================================================
def convert_cars_body_type():
    """
    Klasor bazli siniflandirma verisini YOLO detection formatina cevirir.
    Her goruntunun tamami arac oldugu icin tam kare bbox atanir.
    """
    logger.info("=" * 60)
    logger.info("ADIM 1: Cars_Body_Type -> YOLO Detection")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "Cars_Body_Type"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadi: {src_dir}")
        return

    split_map = {"train": "train", "valid": "valid", "test": "test"}
    total_copied = 0

    for src_split, dst_split in split_map.items():
        split_dir = src_dir / src_split
        if not split_dir.exists():
            logger.warning(f"Split bulunamadi: {split_dir}")
            continue

        img_out = MERGED_DIR / dst_split / "images"
        lbl_out = MERGED_DIR / dst_split / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            if class_name not in CBT_CLASS_MAP:
                logger.warning(f"  Bilinmeyen sinif atlandi: {class_name}")
                continue

            class_id = CBT_CLASS_MAP[class_name]
            count = 0

            for img_file in sorted(class_folder.iterdir()):
                if not img_file.is_file() or not is_image(img_file.name):
                    continue

                # Benzersiz isim: cbt_{sinif}_{orijinal}
                new_name = f"cbt_{class_name.lower()}_{img_file.name}"
                new_stem = Path(new_name).stem

                # Goruntu kopyala
                shutil.copy2(img_file, img_out / new_name)

                # YOLO label olustur (tam kare bbox)
                label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                (lbl_out / f"{new_stem}.txt").write_text(label_content)

                count += 1

            logger.info(
                f"  {src_split}/{class_name} -> Class {class_id}: "
                f"{count} goruntu kopyalandi"
            )
            total_copied += count

    logger.info(f"  Cars_Body_Type TOPLAM: {total_copied} goruntu")
    logger.info("")


# ==============================================================================
# ADIM 2: vehicles.v2 - Secici Filtreleme
# ==============================================================================
def convert_vehicles_v2():
    """
    vehicles.v2 veri setinden sadece minibus ve kamyon siniflarini alir.
    Diger siniflar atilir.
    """
    logger.info("=" * 60)
    logger.info("ADIM 2: vehicles.v2 -> Secici Filtreleme")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "vehicles.v2-release.yolov12"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadi: {src_dir}")
        return

    split_map = {"train": "train", "valid": "valid", "test": "test"}
    total_copied = 0
    total_skipped = 0

    for src_split, dst_split in split_map.items():
        img_src = src_dir / src_split / "images"
        lbl_src = src_dir / src_split / "labels"

        if not img_src.exists() or not lbl_src.exists():
            logger.warning(f"Split bulunamadi: {src_split}")
            continue

        img_out = MERGED_DIR / dst_split / "images"
        lbl_out = MERGED_DIR / dst_split / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        split_copied = 0
        split_skipped = 0

        for lbl_file in sorted(lbl_src.iterdir()):
            if lbl_file.suffix != ".txt":
                continue

            # Label dosyasini oku ve filtrele
            new_lines = []
            for line in lbl_file.read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                old_class_id = int(parts[0])
                if old_class_id in VEH_CLASS_MAP:
                    new_class_id = VEH_CLASS_MAP[old_class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}"
                    new_lines.append(new_line)

            # Hic gecerli satir yoksa bu dosyayi atla
            if not new_lines:
                split_skipped += 1
                continue

            # Goruntu dosyasini bul
            stem = lbl_file.stem
            img_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_src / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                split_skipped += 1
                continue

            # Benzersiz isim
            new_name = f"veh_{img_file.name}"
            new_stem = f"veh_{stem}"

            # Kopyala
            shutil.copy2(img_file, img_out / new_name)
            (lbl_out / f"{new_stem}.txt").write_text(
                "\n".join(new_lines) + "\n"
            )

            split_copied += 1

        logger.info(
            f"  {src_split}: {split_copied} goruntu alindi, "
            f"{split_skipped} goruntu atildi (gecerli sinif yok)"
        )
        total_copied += split_copied
        total_skipped += split_skipped

    logger.info(
        f"  vehicles.v2 TOPLAM: {total_copied} alindi, "
        f"{total_skipped} atildi"
    )
    logger.info("")


# ==============================================================================
# ADIM 3: plateRecognition -> Class ID Donusumu + Split
# ==============================================================================
def convert_plate_recognition():
    """
    plateRecognition verisinde Class ID 0 -> 20 donusumu yapar
    ve %80/%10/%10 oraninda train/valid/test olarak boler.
    """
    logger.info("=" * 60)
    logger.info("ADIM 3: plateRecognition -> ID Donusumu + Split")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "plateRecognition"
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "label"

    if not img_dir.exists() or not lbl_dir.exists():
        logger.error(f"Kaynak dizin bulunamadi: {src_dir}")
        return

    # Tum goruntu-etiket ciftlerini topla
    pairs = []
    for lbl_file in sorted(lbl_dir.iterdir()):
        if lbl_file.suffix != ".txt":
            continue

        stem = lbl_file.stem
        img_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is not None:
            pairs.append((img_file, lbl_file))

    logger.info(f"  Toplam cift: {len(pairs)}")

    # Rastgele karistir ve bol
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
        img_out = MERGED_DIR / split_name / "images"
        lbl_out = MERGED_DIR / split_name / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for img_file, lbl_file in split_pairs:
            # Benzersiz isim
            new_name = f"plt_{img_file.name}"
            new_stem = f"plt_{lbl_file.stem}"

            # Goruntu kopyala
            shutil.copy2(img_file, img_out / new_name)

            # Label donustur: Class 0 -> Class 20
            new_lines = []
            for line in lbl_file.read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    new_id = PLATE_CLASS_MAP.get(old_id, old_id)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")

            (lbl_out / f"{new_stem}.txt").write_text(
                "\n".join(new_lines) + "\n"
            )

        logger.info(f"  {split_name}: {len(split_pairs)} goruntu")

    logger.info("")


# ==============================================================================
# ADIM 4: colorRecognition -> YOLO Classification (Turkce isimler)
# ==============================================================================
def convert_color_recognition():
    """
    colorRecognition siniflandirma verisini Turkce ASCII-safe
    klasor isimleriyle yeni dizine kopyalar.
    pink ve purple cikarilir.
    silver->gri, tan/beige->kahverengi, gold->sari birlestirilir.
    """
    logger.info("=" * 60)
    logger.info("ADIM 4: colorRecognition -> Turkce Siniflandirma")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "colorRecognition"
    if not src_dir.exists():
        logger.error(f"Kaynak dizin bulunamadi: {src_dir}")
        return

    split_map = {"train": "train", "val": "val", "test": "test"}
    total_copied = 0
    total_excluded = 0

    for src_split, dst_split in split_map.items():
        split_dir = src_dir / src_split
        if not split_dir.exists():
            logger.warning(f"Split bulunamadi: {split_dir}")
            continue

        for color_folder in sorted(split_dir.iterdir()):
            if not color_folder.is_dir():
                continue

            eng_name = color_folder.name.lower()

            # Cikarilacak renkler
            if eng_name in COLOR_EXCLUDE:
                excluded = sum(
                    1 for f in color_folder.iterdir()
                    if f.is_file() and is_image(f.name)
                )
                total_excluded += excluded
                logger.info(
                    f"  {src_split}/{eng_name}: {excluded} goruntu "
                    f"CIKARILDI (kilavuzda yok)"
                )
                continue

            # Turkce isim esleme
            if eng_name not in COLOR_MAP:
                logger.warning(f"  Bilinmeyen renk atlandi: {eng_name}")
                continue

            tr_name = COLOR_MAP[eng_name]
            out_dir = COLOR_DIR / dst_split / tr_name
            ensure_dir(out_dir)

            count = 0
            for img_file in sorted(color_folder.iterdir()):
                if not img_file.is_file() or not is_image(img_file.name):
                    continue

                # Dosya isim cakismasini onle (kaynak renk oneki)
                new_name = f"{eng_name}_{img_file.name}"
                shutil.copy2(img_file, out_dir / new_name)
                count += 1

            logger.info(
                f"  {src_split}/{eng_name} -> {tr_name}: {count} goruntu"
            )
            total_copied += count

    logger.info(
        f"  colorRecognition TOPLAM: {total_copied} kopyalandi, "
        f"{total_excluded} cikarildi"
    )
    logger.info("")


# ==============================================================================
# ADIM 4.1: Teknocan Veri Setleri -> YOLO Detection
# ==============================================================================
def convert_teknocan_objects():
    """
    sadece_teknocan_dataset ve teknocan-dataset icerisindeki Class 0
    etiketlerini projenin standart teknocan ID'sine (15) donusturur
    ve 80/10/10 oraninda train/valid/test olarak boler.
    """
    logger.info("=" * 60)
    logger.info("ADIM 4.1: Teknocan Veri Setleri -> Class ID 15")
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
            logger.warning(f"Dizin bulunamadi veya eksik: {d}")
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

    logger.info(f"  Toplam teknocan cifti: {len(pairs)}")
    if not pairs:
        logger.info("")
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
        img_out = MERGED_DIR / split_name / "images"
        lbl_out = MERGED_DIR / split_name / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for img_file, lbl_file in split_pairs:
            new_name = f"tkn_{img_file.name}"
            new_stem = f"tkn_{lbl_file.stem}"

            shutil.copy2(img_file, img_out / new_name)

            new_lines = []
            for line in lbl_file.read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    new_lines.append(f"15 {' '.join(parts[1:])}")

            (lbl_out / f"{new_stem}.txt").write_text(
                "\n".join(new_lines) + "\n"
            )

        logger.info(f"  {split_name}: {len(split_pairs)} goruntu")

    logger.info("")


# ==============================================================================
# ADIM 4.2: Bilgisayar Veri Seti -> YOLO Detection
# ==============================================================================
def convert_bilgisayar_objects():
    """
    bilgisayar-dataset icerisindeki Class 0 etiketlerini
    projenin standart bilgisayar ID'sine (16) donusturur.
    """
    logger.info("=" * 60)
    logger.info("ADIM 4.2: Bilgisayar Veri Seti -> Class ID 16")
    logger.info("=" * 60)

    src_dir = DATASETS_DIR / "bilgisayar-dataset" / "bilgisayar-dataset"
    if not src_dir.exists():
        logger.warning(f"Dizin bulunamadi: {src_dir}")
        return

    split_map = {"train": "train", "valid": "valid", "test": "test"}

    for src_split, dst_split in split_map.items():
        img_dir = src_dir / src_split / "images"
        lbl_dir = src_dir / src_split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_out = MERGED_DIR / dst_split / "images"
        lbl_out = MERGED_DIR / dst_split / "labels"
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
            for line in lbl_file.read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    new_lines.append(f"16 {' '.join(parts[1:])}")

            (lbl_out / f"{new_stem}.txt").write_text(
                "\n".join(new_lines) + "\n"
            )
            count += 1

        logger.info(f"  {dst_split}: {count} goruntu")

    logger.info("")


# ==============================================================================
# ADIM 5: data.yaml Olustur
# ==============================================================================
def create_data_yaml():
    """
    merged_detection icin YOLO data.yaml olusturur.
    """
    logger.info("=" * 60)
    logger.info("ADIM 5: data.yaml Olusturuluyor")
    logger.info("=" * 60)

    yaml_content = f"""# ==============================================================================
# TEKNOFEST 2026 - ANKAAI
# Birlesik Detection Veri Seti (merged_detection)
# ==============================================================================
# Otomatik olusturuldu: tools/prepare_datasets.py
#
# Kaynaklar:
#   - Cars_Body_Type (cbt_*) - arac tipi tespiti
#   - vehicles.v2 (veh_*)    - minibus ve kamyon
#   - plateRecognition (plt_*) - plaka tespiti
#   - teknocan (tkn_*)       - teknocan nesne tespiti
#   - bilgisayar (blg_*)     - bilgisayar nesne tespiti
# ==============================================================================

path: {MERGED_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

# En yuksek Class ID (20) + 1 = 21
# Bos siniflar (7-14, 17-19) YOLO tarafindan otomatik yonetilir
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

    yaml_path = MERGED_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info(f"  data.yaml olusturuldu: {yaml_path}")
    logger.info("")


# ==============================================================================
# ADIM 6: Istatistik Raporu
# ==============================================================================
def print_statistics():
    """
    Cikti dizinlerinin istatistiklerini yazdirir.
    """
    logger.info("=" * 60)
    logger.info("SONUC ISTATISTIKLERI")
    logger.info("=" * 60)

    # --- merged_detection ---
    logger.info("\n--- merged_detection ---")
    for split in ["train", "valid", "test"]:
        img_dir = MERGED_DIR / split / "images"
        lbl_dir = MERGED_DIR / split / "labels"

        if not img_dir.exists():
            continue

        img_count = sum(1 for f in img_dir.iterdir() if f.is_file())
        lbl_count = sum(
            1 for f in lbl_dir.iterdir() if f.is_file()
        ) if lbl_dir.exists() else 0

        # Sinif dagilimi
        class_counts = {}
        if lbl_dir.exists():
            for lbl_file in lbl_dir.iterdir():
                if lbl_file.suffix != ".txt":
                    continue
                for line in lbl_file.read_text().strip().split("\n"):
                    line = line.strip()
                    if line:
                        cid = line.split()[0]
                        class_counts[cid] = class_counts.get(cid, 0) + 1

        logger.info(f"  {split}: {img_count} goruntu, {lbl_count} etiket")
        for cid in sorted(class_counts.keys(), key=lambda x: int(x)):
            # Sinif ismi
            names = {
                "0": "sedan", "1": "suv", "2": "hatchback",
                "3": "pickup", "4": "minibus", "5": "panelvan",
                "6": "kamyon", "15": "teknocan", "16": "bilgisayar", "20": "plaka",
            }
            name = names.get(cid, f"class_{cid}")
            logger.info(f"    Class {cid} ({name}): {class_counts[cid]}")

    # --- color_classification ---
    logger.info("\n--- color_classification ---")
    for split in ["train", "val", "test"]:
        split_dir = COLOR_DIR / split
        if not split_dir.exists():
            continue

        logger.info(f"  {split}:")
        for color_dir in sorted(split_dir.iterdir()):
            if not color_dir.is_dir():
                continue
            count = sum(
                1 for f in color_dir.iterdir()
                if f.is_file() and is_image(f.name)
            )
            logger.info(f"    {color_dir.name}: {count}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("TAMAMLANDI!")
    logger.info("=" * 60)


# ==============================================================================
# ANA FONKSIYON
# ==============================================================================
def main():
    logger.info("=" * 60)
    logger.info("TEKNOFEST 2026 - ANKAAI")
    logger.info("Veri Seti Uyumlastirma Baslatiliyor...")
    logger.info("=" * 60)
    logger.info("")

    # Temiz baslangic: eski cikti dizinlerini temizle
    for out_dir in [MERGED_DIR, COLOR_DIR]:
        if out_dir.exists():
            logger.info(f"Eski cikti dizini temizleniyor: {out_dir}")
            shutil.rmtree(out_dir)

    # Adimlari sirasi ile calistir
    convert_cars_body_type()      # Adim 1
    convert_vehicles_v2()          # Adim 2
    convert_plate_recognition()    # Adim 3
    convert_color_recognition()    # Adim 4
    convert_teknocan_objects()     # Adim 4.1
    convert_bilgisayar_objects()   # Adim 4.2
    create_data_yaml()             # Adim 5
    print_statistics()             # Adim 6


if __name__ == "__main__":
    main()

