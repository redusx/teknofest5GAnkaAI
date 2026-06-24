# ==============================================================================
# TEKNOFEST 2026 - ANKAAI
# Veri Seti Uyumlastirma Betigi (PowerShell)
# ==============================================================================

$ErrorActionPreference = "Stop"

$datasetsDir = "c:\Users\red\Desktop\RED\5GTeknofest\ankaai\datasets"
$mergedDir   = "$datasetsDir\merged_detection"
$colorDir    = "$datasetsDir\color_classification"

# --- Temiz baslangic ---
foreach ($d in @($mergedDir, $colorDir)) {
    if (Test-Path $d) {
        Write-Host "[INFO] Eski cikti temizleniyor: $d"
        Remove-Item -Recurse -Force $d
    }
}

# ==================================================================
# ADIM 1: Cars_Body_Type -> YOLO Detection
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "ADIM 1: Cars_Body_Type -> YOLO Detection"
Write-Host ("=" * 60)

$cbtMap = @{
    "Sedan"       = 0   # sedan
    "SUV"         = 1   # suv
    "Hatchback"   = 2   # hatchback
    "Pick-Up"     = 3   # pickup
    "VAN"         = 5   # panelvan
    "Coupe"       = 0   # sedan (kilavuzda coupe yok)
    "Convertible" = 0   # sedan (kilavuzda convertible yok)
}

$cbtTotal = 0
foreach ($split in @("train","valid","test")) {
    $srcSplit = "$datasetsDir\Cars_Body_Type\$split"
    if (-not (Test-Path $srcSplit)) { continue }

    $imgOut = "$mergedDir\$split\images"
    $lblOut = "$mergedDir\$split\labels"
    New-Item -ItemType Directory -Path $imgOut -Force | Out-Null
    New-Item -ItemType Directory -Path $lblOut -Force | Out-Null

    foreach ($className in $cbtMap.Keys | Sort-Object) {
        $classDir = "$srcSplit\$className"
        if (-not (Test-Path $classDir)) { continue }

        $classId = $cbtMap[$className]
        $count = 0

        Get-ChildItem $classDir -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp)$' } | ForEach-Object {
            $newName = "cbt_$($className.ToLower())_$($_.Name)"
            $newStem = [System.IO.Path]::GetFileNameWithoutExtension($newName)

            Copy-Item $_.FullName -Destination "$imgOut\$newName"
            Set-Content -Path "$lblOut\$newStem.txt" -Value "$classId 0.5 0.5 1.0 1.0" -NoNewline

            $count++
        }

        Write-Host "  $split/$className -> Class $classId : $count goruntu"
        $cbtTotal += $count
    }
}
Write-Host "  Cars_Body_Type TOPLAM: $cbtTotal goruntu"

# ==================================================================
# ADIM 2: vehicles.v2 — Secici Filtreleme
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "ADIM 2: vehicles.v2 -> Secici Filtreleme"
Write-Host ("=" * 60)

# Kaynak ID -> Hedef ID (sadece bunlar alinacak)
$vehMap = @{ "0" = 4; "1" = 6; "6" = 4; "7" = 6 }

$vehCopied = 0; $vehSkipped = 0
foreach ($split in @("train","valid","test")) {
    $imgSrc = "$datasetsDir\vehicles.v2-release.yolov12\$split\images"
    $lblSrc = "$datasetsDir\vehicles.v2-release.yolov12\$split\labels"
    if (-not (Test-Path $imgSrc) -or -not (Test-Path $lblSrc)) { continue }

    $imgOut = "$mergedDir\$split\images"
    $lblOut = "$mergedDir\$split\labels"
    New-Item -ItemType Directory -Path $imgOut -Force | Out-Null
    New-Item -ItemType Directory -Path $lblOut -Force | Out-Null

    $splitCopied = 0; $splitSkipped = 0

    Get-ChildItem $lblSrc -Filter "*.txt" | ForEach-Object {
        $lblFile = $_
        $stem = $lblFile.BaseName
        $lines = Get-Content $lblFile.FullName

        $newLines = @()
        foreach ($line in $lines) {
            $line = $line.Trim()
            if ($line -eq "") { continue }
            $parts = $line -split '\s+'
            if ($parts.Count -lt 5) { continue }
            $oldId = $parts[0]
            if ($vehMap.ContainsKey($oldId)) {
                $newId = $vehMap[$oldId]
                $rest = ($parts[1..($parts.Count-1)]) -join ' '
                $newLines += "$newId $rest"
            }
        }

        if ($newLines.Count -eq 0) {
            $splitSkipped++
            return  # ForEach-Object icinde continue gibi calisir
        }

        # Goruntu bul
        $imgFile = $null
        foreach ($ext in @(".jpg",".jpeg",".png")) {
            $candidate = "$imgSrc\$stem$ext"
            if (Test-Path $candidate) { $imgFile = $candidate; break }
        }
        if ($null -eq $imgFile) { $splitSkipped++; return }

        $imgName = [System.IO.Path]::GetFileName($imgFile)
        $newImgName = "veh_$imgName"
        $newStem = "veh_$stem"

        Copy-Item $imgFile -Destination "$imgOut\$newImgName"
        Set-Content -Path "$lblOut\$newStem.txt" -Value ($newLines -join "`n")

        $splitCopied++
    }

    Write-Host "  ${split}: $splitCopied alindi, $splitSkipped atildi"
    $vehCopied += $splitCopied; $vehSkipped += $splitSkipped
}
Write-Host "  vehicles.v2 TOPLAM: $vehCopied alindi, $vehSkipped atildi"

# ==================================================================
# ADIM 3: plateRecognition -> ID Donusumu + Split
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "ADIM 3: plateRecognition -> ID Donusumu + Split"
Write-Host ("=" * 60)

$pltImgDir = "$datasetsDir\plateRecognition\images"
$pltLblDir = "$datasetsDir\plateRecognition\label"

# Tum ciftleri topla
$pairs = @()
Get-ChildItem $pltLblDir -Filter "*.txt" | ForEach-Object {
    $stem = $_.BaseName
    foreach ($ext in @(".jpg",".jpeg",".png")) {
        $imgPath = "$pltImgDir\$stem$ext"
        if (Test-Path $imgPath) {
            $pairs += [PSCustomObject]@{ Img=$imgPath; Lbl=$_.FullName; Stem=$stem }
            break
        }
    }
}

Write-Host "  Toplam cift: $($pairs.Count)"

# Karistir (Fisher-Yates shuffle, seed=42)
$rng = [System.Random]::new(42)
for ($i = $pairs.Count - 1; $i -gt 0; $i--) {
    $j = $rng.Next(0, $i + 1)
    $tmp = $pairs[$i]; $pairs[$i] = $pairs[$j]; $pairs[$j] = $tmp
}

$n = $pairs.Count
$nTrain = [int]($n * 0.8)
$nValid = [int]($n * 0.1)

$splitAssign = @{
    "train" = $pairs[0..($nTrain-1)]
    "valid" = $pairs[$nTrain..($nTrain+$nValid-1)]
    "test"  = $pairs[($nTrain+$nValid)..($n-1)]
}

foreach ($splitName in @("train","valid","test")) {
    $splitPairs = $splitAssign[$splitName]
    $imgOut = "$mergedDir\$splitName\images"
    $lblOut = "$mergedDir\$splitName\labels"
    New-Item -ItemType Directory -Path $imgOut -Force | Out-Null
    New-Item -ItemType Directory -Path $lblOut -Force | Out-Null

    foreach ($p in $splitPairs) {
        $imgName = [System.IO.Path]::GetFileName($p.Img)
        $newImgName = "plt_$imgName"
        $newStem = "plt_$($p.Stem)"

        Copy-Item $p.Img -Destination "$imgOut\$newImgName"

        $lines = Get-Content $p.Lbl
        $newLines = @()
        foreach ($line in $lines) {
            $line = $line.Trim()
            if ($line -eq "") { continue }
            $parts = $line -split '\s+'
            if ($parts.Count -ge 5) {
                $rest = ($parts[1..($parts.Count-1)]) -join ' '
                $newLines += "20 $rest"  # Class 0 -> 20
            }
        }
        Set-Content -Path "$lblOut\$newStem.txt" -Value ($newLines -join "`n")
    }

    Write-Host "  ${splitName}: $($splitPairs.Count) goruntu"
}

# ==================================================================
# ADIM 4: colorRecognition -> Turkce Siniflandirma
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "ADIM 4: colorRecognition -> Turkce Siniflandirma"
Write-Host ("=" * 60)

$colorMap = @{
    "black"  = "siyah"
    "white"  = "beyaz"
    "grey"   = "gri"
    "silver" = "gri"
    "red"    = "kirmizi"
    "blue"   = "mavi"
    "yellow" = "sari"
    "gold"   = "sari"
    "green"  = "yesil"
    "orange" = "turuncu"
    "brown"  = "kahverengi"
    "tan"    = "kahverengi"
    "beige"  = "kahverengi"
}
$colorExclude = @("pink","purple")

$colorCopied = 0; $colorExcluded = 0
foreach ($split in @("train","val","test")) {
    $splitDir = "$datasetsDir\colorRecognition\$split"
    if (-not (Test-Path $splitDir)) { continue }

    Get-ChildItem $splitDir -Directory | ForEach-Object {
        $engName = $_.Name.ToLower()
        $srcPath = $_.FullName

        if ($colorExclude -contains $engName) {
            $cnt = (Get-ChildItem $srcPath -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png)$' }).Count
            $colorExcluded += $cnt
            Write-Host "  $split/$engName : $cnt CIKARILDI (kilavuzda yok)"
            return
        }

        if (-not $colorMap.ContainsKey($engName)) {
            Write-Host "  UYARI: Bilinmeyen renk atlandi: $engName"
            return
        }

        $trName = $colorMap[$engName]
        $outDir = "$colorDir\$split\$trName"
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null

        $cnt = 0
        Get-ChildItem $srcPath -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png)$' } | ForEach-Object {
            $newName = "${engName}_$($_.Name)"
            Copy-Item $_.FullName -Destination "$outDir\$newName"
            $cnt++
        }

        Write-Host "  $split/$engName -> $trName : $cnt goruntu"
        $colorCopied += $cnt
    }
}
Write-Host "  colorRecognition TOPLAM: $colorCopied kopyalandi, $colorExcluded cikarildi"

# ==================================================================
# ADIM 5: data.yaml Olustur
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "ADIM 5: data.yaml Olusturuluyor"
Write-Host ("=" * 60)

$yamlContent = @"
# ==============================================================================
# TEKNOFEST 2026 - ANKAAI
# Birlesik Detection Veri Seti (merged_detection)
# ==============================================================================
path: .
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
"@

Set-Content -Path "$mergedDir\data.yaml" -Value $yamlContent -Encoding UTF8
Write-Host "  data.yaml olusturuldu: $mergedDir\data.yaml"

# ==================================================================
# ADIM 6: Istatistik
# ==================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "SONUC ISTATISTIKLERI"
Write-Host ("=" * 60)

$classNames = @{ "0"="sedan"; "1"="suv"; "2"="hatchback"; "3"="pickup"; "4"="minibus"; "5"="panelvan"; "6"="kamyon"; "20"="plaka" }

Write-Host "`n--- merged_detection ---"
foreach ($split in @("train","valid","test")) {
    $imgDir = "$mergedDir\$split\images"
    $lblDir = "$mergedDir\$split\labels"
    if (-not (Test-Path $imgDir)) { continue }

    $imgCount = (Get-ChildItem $imgDir -File).Count
    $lblCount = if (Test-Path $lblDir) { (Get-ChildItem $lblDir -File).Count } else { 0 }

    Write-Host "  ${split}: $imgCount goruntu, $lblCount etiket"

    $classCounts = @{}
    if (Test-Path $lblDir) {
        Get-ChildItem $lblDir -Filter "*.txt" | ForEach-Object {
            Get-Content $_.FullName | ForEach-Object {
                $l = $_.Trim()
                if ($l -ne "") {
                    $cid = ($l -split '\s+')[0]
                    if ($classCounts.ContainsKey($cid)) { $classCounts[$cid]++ } else { $classCounts[$cid] = 1 }
                }
            }
        }
    }

    foreach ($cid in $classCounts.Keys | Sort-Object { [int]$_ }) {
        $name = if ($classNames.ContainsKey($cid)) { $classNames[$cid] } else { "class_$cid" }
        Write-Host "    Class $cid ($name): $($classCounts[$cid])"
    }
}

Write-Host "`n--- color_classification ---"
foreach ($split in @("train","val","test")) {
    $splitDir = "$colorDir\$split"
    if (-not (Test-Path $splitDir)) { continue }
    Write-Host "  ${split}:"
    Get-ChildItem $splitDir -Directory | Sort-Object Name | ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png)$' }).Count
        Write-Host "    $($_.Name): $cnt"
    }
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "TAMAMLANDI!"
Write-Host ("=" * 60)
