# VERİ VE ETİKET SÖZLÜĞÜ

Aşağıdaki JSON şeması ve etiket isimleri DİREKT OLARAK kullanılmalıdır. Tüm etiketler Türkçe karakter İÇERMEYEN (ASCII-safe) ve küçük harfli standart metinler olmak ZORUNDADIR (Örn: `kırmızı` YASAK, `kirmizi` DOĞRU).

## 1. Araç Bilgisi (`arac_bilgisi` nesnesi)
Video boyunca tek bir yapı olarak verilir.
- **`tip`**: hatchback, pickup, sedan, suv, minibus, panelvan, kamyon
- **`renk`**: beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi
- **`plaka`**: Türkiye plaka standart formatı, regex ile düzeltilmiş, birleşik (Örn: 34ABC123). Çözülemezse "tespit edilemedi".
- **`confidence_score`**: 0.0 - 1.0 arası float (Tüm araç tip, plaka ve renk tahmininin genel güven skoru).

## 2. Yol Güvenliği Tespitleri (`tespitler` dizisi)
Tespit edilen her anomali/nesne/yolcu için ayrı ayrı şu formatta bir obje oluşturulur: `zaman_saniye` (float), `kategori`, `etiket`, `confidence_score` (float).

**Kategori 1: sofor_eylemi**
- `arkaya_bakma`
- `esneme`
- `sigara_icme`
- `su_icme`
- `telefonla_konusma`
- `slalom`
- `etrafa_bakinma`
- `emniyet_kemeri_ihlali`

**Kategori 2: nesneler**
- `teknocan`
- `bilgisayar`

**Kategori 3: yolcular**
- `arka_koltuk_1`
- `arka_koltuk_2`
- `on_koltuk`

## 3. Örnek JSON Çıktı Çatısı
Çıktı `results.json` dosyası kesinlikle aşağıdaki hiyerarşik yapıyı (Anahtarları) kullanmalıdır:

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.94
  },
  "tespitler": [
    {
      "zaman_saniye": 14.5,
      "kategori": "sofor_eylemi",
      "etiket": "telefonla_konusma",
      "confidence_score": 0.89
    }
  ]
}