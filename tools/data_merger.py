#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Veri Birlestirme ve Etiket Temizleme Betigi (Data Merger & Label Cleaner)

Bu betik, farkli kaynaklardan (Kaggle, Roboflow vb.) gelen YOLO formatindaki
veri setlerini tek bir standart formata donusturur.

Ozellikler:
  - Farkli veri setlerindeki Class ID'lerini standart ID'lere donusturur
  - Turkce karakterleri ASCII-safe karsiliklarina cevirir
  - Dosya isim cakismasini benzersiz on-ek (prefix) ile engeller
  - Gecersiz etiketleri filtreler ve log tutar

Kullanim:
  python tools/data_merger.py \
      --sources data/raw/kaggle_set:kaggle_driver_distraction \
                data/raw/roboflow_set:roboflow_vehicle_detection \
      --output data/merged \
      --config configs/class_config.yaml
"""

import os
import sys
import glob
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ==============================================================================
# Logger Konfigurasyonu
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_merger")

# ==============================================================================
# Turkce Karakter -> ASCII Donusum Tablosu
# ==============================================================================
TR_CHAR_MAP: Dict[str, str] = {
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U",
}


def sanitize_label(text: str) -> str:
    """
    Etiketi ASCII-safe ve kucuk harfli formata donusturur.

    Kurallar (FTR Model Kilavuz §1, §5):
      - Tum harfler kucuk olacak
      - Turkce karakterler ASCII karsiliklarina donusturulecek
      - Bosluklar alt cizgi (_) ile degistirilecek

    Args:
        text: Donusturulecek etiket metni.

    Returns:
        ASCII-safe, kucuk harfli etiket.
    """
    for tr_char, ascii_char in TR_CHAR_MAP.items():
        text = text.replace(tr_char, ascii_char)
    text = text.lower().strip()
    text = text.replace(" ", "_")
    return text


def generate_prefix(source_name: str, file_path: str) -> str:
    """
    Dosya isim cakismasini engellemek icin benzersiz bir on-ek (prefix) uretir.

    Format: {kaynak_adi}_{hash_kisa}
    Hash, kaynak adi + dosya yolunun MD5 ozetinin ilk 6 karakteridir.

    Args:
        source_name: Veri seti kaynaginin adi.
        file_path: Orijinal dosya yolu.

    Returns:
        Benzersiz on-ek metni.
    """
    hash_input = f"{source_name}_{file_path}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    return f"{source_name}_{short_hash}"


def load_config(config_path: str) -> dict:
    """
    YAML konfigürasyon dosyasini yukler.

    Args:
        config_path: class_config.yaml dosyasinin yolu.

    Returns:
        Konfigürasyon sozlugu.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        yaml.YAMLError: YAML parse hatasi.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyasi bulunamadi: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Konfigürasyon yuklendi: {config_path}")
    logger.info(f"  Standart sinif sayisi: {len(config.get('standard_classes', {}))}")
    logger.info(f"  Kaynak esleme sayisi: {len(config.get('source_mappings', {}))}")
    return config


def remap_label_line(
    line: str,
    class_mapping: Dict[int, int],
    standard_classes: Dict[int, str],
) -> Optional[str]:
    """
    Tek bir YOLO etiket satirindaki Class ID'yi yeni standart ID'ye donusturur.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>

    Args:
        line: YOLO formatlı etiket satiri.
        class_mapping: Eski ID -> Yeni ID esleme sozlugu.
        standard_classes: Standart sinif tanimlari.

    Returns:
        Donusturulmus etiket satiri veya None (gecersiz etiketler icin).
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        old_class_id = int(parts[0])
    except ValueError:
        logger.warning(f"  Gecersiz class ID: '{parts[0]}' -> satir atlanacak")
        return None

    # Esleme tablosunda bu eski ID var mi?
    if old_class_id not in class_mapping:
        logger.debug(f"  Esleme bulunamadi: eski_id={old_class_id} -> satir atlanacak")
        return None

    new_class_id = class_mapping[old_class_id]

    # Yeni ID standart siniflar icinde mi?
    if new_class_id not in standard_classes:
        logger.warning(
            f"  Yeni ID standart siniflar icinde degil: {new_class_id} -> satir atlanacak"
        )
        return None

    # Bbox koordinatlarini dogrula (0-1 arasi)
    try:
        coords = [float(p) for p in parts[1:5]]
        for coord in coords:
            if not (0.0 <= coord <= 1.0):
                logger.warning(
                    f"  Gecersiz bbox koordinati: {coord} -> satir atlanacak"
                )
                return None
    except ValueError:
        logger.warning(f"  Bbox parse hatasi -> satir atlanacak")
        return None

    # Yeni satiri olustur
    parts[0] = str(new_class_id)
    return " ".join(parts)


def process_label_file(
    label_path: Path,
    class_mapping: Dict[int, int],
    standard_classes: Dict[int, str],
) -> Tuple[List[str], int, int]:
    """
    Bir etiket dosyasindaki tum satirlari isler.

    Args:
        label_path: .txt etiket dosyasinin yolu.
        class_mapping: Eski ID -> Yeni ID esleme sozlugu.
        standard_classes: Standart sinif tanimlari.

    Returns:
        (donusturulmus_satirlar, basarili_sayisi, atlanan_sayisi)
    """
    converted_lines = []
    skipped = 0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            new_line = remap_label_line(line, class_mapping, standard_classes)
            if new_line is not None:
                converted_lines.append(new_line)
            else:
                skipped += 1

    return converted_lines, len(converted_lines), skipped


def find_matching_image(label_path: Path, images_dir: Path) -> Optional[Path]:
    """
    Bir etiket dosyasina karsilik gelen goruntu dosyasini bulur.

    Args:
        label_path: .txt etiket dosyasi.
        images_dir: Goruntulerin bulundugu dizin.

    Returns:
        Eslesen goruntu dosyasinin yolu veya None.
    """
    stem = label_path.stem
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for ext in image_extensions:
        image_path = images_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path

    return None


def merge_dataset(
    source_dir: str,
    source_name: str,
    output_dir: str,
    config: dict,
) -> Dict[str, int]:
    """
    Tek bir veri seti kaynagini isler ve cikti dizinine yazar.

    Args:
        source_dir: Kaynak veri setinin dizini (images/ ve labels/ alt dizinleri icermeli).
        source_name: Kaynak adi (config'deki source_mappings anahtari).
        output_dir: Birlestirilmis ciktinin yazilacagi dizin.
        config: Yuklenmmis YAML konfigurasyonu.

    Returns:
        Istatistik sozlugu: {"processed", "skipped", "no_image", "total_labels"}
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Cikti dizinlerini olustur
    out_images = output_path / "images"
    out_labels = output_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Kaynak esleme tablosunu al
    source_mappings = config.get("source_mappings", {})
    if source_name not in source_mappings:
        logger.error(f"Kaynak esleme bulunamadi: '{source_name}'")
        logger.error(f"Mevcut kaynaklar: {list(source_mappings.keys())}")
        return {"processed": 0, "skipped": 0, "no_image": 0, "total_labels": 0}

    class_mapping = source_mappings[source_name]
    # YAML'dan int'e cevirme (YAML bazen string key yapar)
    class_mapping = {int(k): int(v) for k, v in class_mapping.items()}

    standard_classes = config.get("standard_classes", {})
    standard_classes = {int(k): v for k, v in standard_classes.items()}

    # Kaynak dizinde images/ ve labels/ bul
    src_labels_dir = source_path / "labels"
    src_images_dir = source_path / "images"

    if not src_labels_dir.exists():
        # labels/ yoksa, train/labels, valid/labels dene
        for sub in ["train", "valid", "test"]:
            candidate = source_path / sub / "labels"
            if candidate.exists():
                src_labels_dir = candidate
                src_images_dir = source_path / sub / "images"
                break

    if not src_labels_dir.exists():
        logger.error(f"Labels dizini bulunamadi: {source_path}")
        return {"processed": 0, "skipped": 0, "no_image": 0, "total_labels": 0}

    label_files = list(src_labels_dir.glob("*.txt"))
    logger.info(f"Kaynak: {source_name} | Dizin: {source_dir}")
    logger.info(f"  Bulunan etiket dosyasi: {len(label_files)}")

    stats = {"processed": 0, "skipped": 0, "no_image": 0, "total_labels": 0}

    for label_file in label_files:
        # Benzersiz prefix olustur
        prefix = generate_prefix(source_name, str(label_file))
        new_name = f"{prefix}_{label_file.stem}"

        # Etiket dosyasini isle
        converted_lines, success_count, skip_count = process_label_file(
            label_file, class_mapping, standard_classes
        )
        stats["total_labels"] += success_count + skip_count
        stats["skipped"] += skip_count

        if not converted_lines:
            stats["skipped"] += 1
            continue

        # Eslesen goruntu dosyasini bul
        image_file = find_matching_image(label_file, src_images_dir)
        if image_file is None:
            logger.warning(f"  Goruntu bulunamadi: {label_file.stem} -> atlanacak")
            stats["no_image"] += 1
            continue

        # Donusturulmus etiketi yaz
        out_label_path = out_labels / f"{new_name}.txt"
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(converted_lines) + "\n")

        # Goruntu dosyasini kopyala
        out_image_path = out_images / f"{new_name}{image_file.suffix}"
        shutil.copy2(image_file, out_image_path)

        stats["processed"] += 1

    logger.info(f"  Islenen: {stats['processed']} | Atlanan: {stats['skipped']} | "
                f"Goruntu yok: {stats['no_image']}")

    return stats


def generate_dataset_yaml(output_dir: str, config: dict) -> str:
    """
    Birlestirilmis veri seti icin YOLO egitim YAML dosyasini olusturur.

    Args:
        output_dir: Birlestirilmis veri setinin dizini.
        config: YAML konfigurasyonu.

    Returns:
        Olusturulan YAML dosyasinin yolu.
    """
    output_path = Path(output_dir)
    standard_classes = config.get("standard_classes", {})
    standard_classes = {int(k): v for k, v in standard_classes.items()}

    dataset_yaml = {
        "path": str(output_path.resolve()),
        "train": "images",
        "val": "images",
        "names": standard_classes,
    }

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Dataset YAML olusturuldu: {yaml_path}")
    return str(yaml_path)


def main():
    """Ana giris noktasi."""
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 - Veri Birlestirme ve Etiket Temizleme Betigi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek Kullanim:
  python tools/data_merger.py \\
      --sources data/raw/kaggle_set:kaggle_driver_distraction \\
                data/raw/roboflow_set:roboflow_vehicle_detection \\
      --output data/merged \\
      --config configs/class_config.yaml
        """,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Kaynak dizin:kaynak_adi ciftleri (Orn: data/raw/kaggle:kaggle_driver_distraction)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Birlestirilmis cikti dizini (Orn: data/merged)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/class_config.yaml",
        help="Sinif esleme konfigürasyon dosyasi (varsayilan: configs/class_config.yaml)",
    )
    parser.add_argument(
        "--generate-yaml",
        action="store_true",
        default=True,
        help="YOLO egitim YAML dosyasini olustur (varsayilan: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detayli log ciktisi",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Konfigurasyonu yukle
    try:
        config = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Konfigürasyon yuklenemedi: {e}")
        sys.exit(1)

    # Kaynaklari isle
    total_stats = {"processed": 0, "skipped": 0, "no_image": 0, "total_labels": 0}

    logger.info("=" * 60)
    logger.info("VERI BIRLESTIRME ISLEMI BASLATILDI")
    logger.info("=" * 60)

    for source_spec in args.sources:
        if ":" not in source_spec:
            logger.error(
                f"Gecersiz kaynak formati: '{source_spec}'. "
                f"Format: dizin_yolu:kaynak_adi"
            )
            continue

        source_dir, source_name = source_spec.rsplit(":", 1)

        try:
            stats = merge_dataset(source_dir, source_name, args.output, config)
            for key in total_stats:
                total_stats[key] += stats[key]
        except Exception as e:
            logger.error(f"Kaynak islenirken hata: {source_name} -> {e}")
            continue

    # Ozet
    logger.info("=" * 60)
    logger.info("BIRLESTIRME TAMAMLANDI")
    logger.info(f"  Toplam islenen dosya : {total_stats['processed']}")
    logger.info(f"  Toplam atlanan etiket: {total_stats['skipped']}")
    logger.info(f"  Goruntu bulunamayan  : {total_stats['no_image']}")
    logger.info(f"  Toplam etiket satiri : {total_stats['total_labels']}")
    logger.info("=" * 60)

    # YOLO dataset.yaml olustur
    if args.generate_yaml:
        generate_dataset_yaml(args.output, config)

    logger.info("Islem basariyla tamamlandi.")


if __name__ == "__main__":
    main()
