# Cloud-Shadow Segmentation pada Landsat dengan Anotasi Shadow Tidak Lengkap

Kode eksperimen untuk mengukur pengaruh anotasi cloud shadow yang tidak lengkap pada
benchmark L8 Biome dan L7 Irish, dengan SPARCS sebagai set uji eksternal berlabel lengkap,
serta partial label berbasis geometri matahari--awan. Eksperimen disusun per fase.

## Struktur

```
cloudshadow_revision/
├── run_experiments.py         # runner fase 1-4 (resume otomatis)
├── analyze.py                 # fase 5a: tabel, uji statistik, ANOVA, bootstrap, error analysis
├── make_figures.py            # fase 5b: confusion matrix, kurva, figure kualitatif dari TEST
├── scripts/
│   ├── phase0_check_setup.py      # fase 0a: environment, split scene, overlap footprint
│   ├── phase0_scan_train.py       # fase 0b: index patch + statistik + normalisasi
│   └── legacy_check_index_leakage.py   # cek kebocoran index versi lama
├── src/
│   ├── config.py      # semua konstanta (seed, split, bobot kelas, default training)
│   ├── experiment.py  # RunConfig + penamaan run deterministik
│   ├── phases.py      # definisi run per fase
│   ├── data.py        # DataModule L7 Irish & L8 Biome
│   ├── samplers.py    # random, oversample, ratio-weighted, class-aware
│   ├── losses.py      # ce, wce, cb, ohem, hybrid
│   ├── models.py      # U-Net(+CBAM), ResNet-18 U-Net (SSL4EO-L)
│   ├── metrics.py     # metrik dari confusion matrix (global & per scene)
│   ├── train.py       # training + evaluasi satu run
│   ├── geo_compat.py  # kompatibilitas versi TorchGeo (bbox & index)
│   └── utils.py
├── dataset/    # dataset mentah: l7irish/, l8biome/ (taruh manual / diunduh TorchGeo)
├── data/       # artefak fase 0 (index, stats, split)  (dibuat otomatis)
├── runs/       # hasil JSON, checkpoint, log    (dibuat otomatis)
├── analysis/   # tabel & statistik              (dibuat otomatis)
└── figures/
```

## Urutan menjalankan

Semua perintah dijalankan dari folder root proyek.

### Fase 0: fondasi (wajib, sekali saja)

```bash
python scripts/phase0_check_setup.py --dataset l7irish l8biome
python scripts/phase0_scan_train.py  --dataset l7irish l8biome --extra_splits val test
```

`phase0_check_setup.py` harus selesai tanpa error. Jika gagal di bagian bbox atau daftar
scene, berarti `src/geo_compat.py` perlu disesuaikan dengan versi TorchGeo kamu.

`phase0_scan_train.py` men-scan **seluruh** scene training (bisa 1–2 jam per dataset).
Outputnya dipakai semua fase berikutnya. Angka di `data/{ds}_train_stats.json` dipakai
langsung di paper: prevalensi shadow dengan dan tanpa Fill, π_shadow empiris, dan waktu
preprocessing.

### Fase 1–4: training

Selalu mulai dengan uji cepat, lalu lihat daftar run:

```bash
python run_experiments.py --phase 1 --quick_test
python run_experiments.py --phase 1 --dry_run
python run_experiments.py --phase 1
```

| Fase | Isi | Run | Estimasi GPU* |
|---|---|---|---|
| 1 | Factorial 2×2×2: sampling × CBAM × loss, 5 seed, 2 dataset | 80 | ~4,7 hari |
| 2 | Baseline lain: CE, class-balanced, OHEM, ratio-weighted, class-aware, random non-empty | 60 | ~3,5 hari |
| 3 | Sensitivitas p, τ, jitter (L7, 5 seed) | 40 | ~2,4 hari |
| 4 | Backbone pretrained × sampling: ResNet-18 SSL4EO-L (fine-tune, frozen, scratch) + DeepLabV3+ R50 ImageNet {random, OS} | 60 | ~3,5 hari |

\* asumsi ~85 menit/run di RTX 4060. Total 240 run (~14,2 hari GPU).

**Random non-empty** (`sampling="nonempty"`) mengambil patch seragam dari patch grid yang berisi
data valid, tanpa prioritas shadow. Ini memisahkan dua efek oversampling: menghindari patch
Fill (non-empty vs random) dan memprioritaskan shadow (OS vs non-empty). Perlu karena
`RandomGeoSampler` memilih sebanding luas footprint di EPSG:3857, sehingga di L7 Irish sekitar
86% patch acaknya hanya Fill (scene lintang tinggi membengkak di Web Mercator).

**DeepLabV3+** (`arch="deeplabv3plus"`, encoder ResNet-50 ImageNet, `segmentation_models_pytorch`)
adalah backbone kompetitif untuk menguji apakah efek sampling tetap ada di pipeline yang lebih
kuat. 26,7M parameter, ~19 GFLOPs per patch, ~1,6 GB memori training (batch 8).

Checkpoint terbaik & early stopping dipilih dengan **val mIoU** (`SELECT_METRIC` di
`src/config.py`; val loss tetap dipakai untuk LR scheduler), seragam untuk semua run.
Ubah ke `"loss"` untuk mereplikasi perilaku versi lama. GFLOPs (1 patch, forward) tersimpan
di JSON hasil dan `cost_table.csv`.

Run yang sudah selesai otomatis di-skip, jadi aman di-Ctrl+C lalu dijalankan ulang.
Untuk mengurangi waktu, batasi seed atau dataset, misalnya
`--seeds 42 123 2024` atau `--dataset l7irish`.

Prioritas kalau waktu terbatas: Fase 0 → 1 → 2 → 4 → 3.

### Fase 6: pilot partial label (label shadow tidak lengkap)

Shadow hanya dianotasi di sebagian scene (L7 Irish ~45/206, L8 Biome ~32/96); di scene lain
bayangan berlabel Clear. Mask turunan dibuat sekali, lalu pilot 12 run (1 seed):

```bash
python scripts/make_partial_masks.py --dataset l8biome l7irish --mode geom all
python run_experiments.py --phase 6 --quick_test --dataset l8biome
python run_experiments.py --phase 6
```

| `partial` | Piksel Clear di scene tanpa anotasi shadow |
|---|---|
| `none` | tetap Clear (mask asli) |
| `all` | semua menjadi "Clear atau Shadow" (setara Francis 2024, SEnSeI-v2) |
| `geom` | hanya yang berada di jalur proyeksi bayangan awan (mask awan ground truth digeser menjauhi matahari untuk tinggi awan 0,2–12 km; sudut matahari dari MTL) |

Piksel "Clear atau Shadow" dilatih dengan marginal loss −log(p_clear + p_shadow) (`losses.WeightedCE`,
hanya `ce`/`wce`). Evaluasi selalu memakai label asli. Mask turunan ada di `dataset/partial/{mode}/`
(citra di-hardlink), ringkasan per scene di `data/{ds}_partial_{mode}.json`, termasuk validasi geometri:
berapa persen shadow asli di scene berlabel yang tertutup jalur (`swath_shadow_recall_annotated`).

### Koreksi prior post-hoc (tanpa training ulang)

Oversampling menggeser prior kelas saat training (~60% patch ber-shadow) sehingga recall shadow
naik tetapi precision jatuh. `scripts/prior_correction.py` mengevaluasi ulang checkpoint yang sudah ada:

```bash
python scripts/prior_correction.py --pattern "*oversample*"
```

- `prior`: logit_c += log(pi_target_c / pi_train_c) (Saerens et al. 2002; Menon et al. 2021);
  pi_train diukur dari sampler run itu sendiri, pi_target dari piksel valid scene training. Tanpa tuning.
- `tuned`: bias pada logit shadow dipilih di validasi (F1 shadow di scene val yang berisi shadow).

Hasil: `runs/{ds}/prior/{nama}.json` (confusion base/prior/tuned, global + per scene, dan grid bias
untuk kurva precision–recall). Aman dijalankan paralel dengan training (hanya inferensi).

### Validasi eksternal SPARCS (model L8 Biome)

Test set Biome hanya berlabel shadow di ~1/3 scene, jadi precision shadow di sana bias ke bawah.
SPARCS (Hughes & Kennedy 2019; 78 subscene L8 1000×1000 px, label shadow lengkap, 7,4% piksel
shadow) dipakai sebagai test set eksternal. Unduh `sparcs_data_L8.zip` dari
https://emapr.ceoas.oregonstate.edu/sparcs/ ke `dataset/sparcs/`, lalu:

```bash
python scripts/prepare_sparcs.py   # sekali: dataset/sparcs/prepared/*.npz + data/sparcs_prepared.json
python scripts/eval_sparcs.py      # semua run L8 Biome; --pattern untuk subset
```

- Radiometri: DN // 256, sama dengan L8 Biome versi TorchGeo (bukan TOA; dicek dari band B9).
  B8 (tidak ada di SPARCS) diisi regresi B2–B4 dari scene train Biome (R² 0,999).
- Kelas: shadow (+ shadow di atas air) → Shadow; air/salju/darat/banjir → Clear; awan → Cloud.
  Thin + Cloud prediksi digabung; metrik 3 kelas (`mIoU3`, per kelas P/R/F1/IoU).
- Tidak ada scene SPARCS yang sama dengan scene Biome; 2 subscene berbagi path/row dengan scene
  train Biome (tanggal berbeda), tercatat di `data/sparcs_prepared.json`.

Hasil: `runs/l8biome/sparcs/{nama}.json`. Hanya inferensi; aman paralel dengan training.

### Fase 5: analisis & figure

```bash
python analyze.py                       # analysis/report.md + CSV
python make_figures.py --cm --curves
python make_figures.py --qualitative --dataset l7irish --seed 42
```

`analyze.py` bisa dijalankan kapan saja, termasuk saat fase belum selesai. Baris yang
belum ada hasilnya akan ditandai "belum ada".

## Perubahan penting dibanding kode lama

1. **Index shadow** dibangun dari scene **training saja** dan dari **seluruh** scene training.
   Versi lama men-scan `full_dataset` dan berhenti setelah 3000 lokasi.
2. **Validasi per epoch** memakai patch yang sama di semua epoch dan run. Versi lama berganti
   tiap epoch karena generator tidak dipakai, sehingga early stopping dan pemilihan checkpoint noisy.
3. **Baseline versi lama adalah weighted CE** dengan bobot [0, 5, 1, 1,5, 1]. Ini dipertahankan
   sebagai `wce`. CE tanpa bobot ditambahkan di Fase 2.
4. **Focal loss diperbaiki**: `pt` dihitung dari CE tanpa bobot dan rata-rata hanya atas piksel valid.
5. **SD dilaporkan dengan ddof=1** (versi lama ddof=0, sehingga SD lebih kecil ~18% untuk n=3).
6. **Uji berpasangan dipasangkan per seed** (versi lama mengikuti urutan file).
7. **Semua faktor tersimpan di JSON** (versi lama: run `--no_pretrained`, lr, augment berbeda
   bisa tercampur dalam satu baris tabel).
8. **Normalisasi z-score per band** (statistik dari scene training) dan **augmentasi per sampel**
   aktif di semua run.
9. **Encoder frozen**: BatchNorm encoder ikut dibekukan (versi lama statistik BN tetap berubah).
10. Checkpoint versi lama **tidak kompatibel** dengan proyek ini (nama layer CBAM berubah).
11. **Split berbasis lokasi** (WRS-2 path/row), bukan per scene. Versi lama (`random_bbox_assignment`)
    menaruh akuisisi berbeda dari lokasi yang sama di train dan test (L7 Irish: 13 dari 41 scene
    test, overlap ≥50%). Angka hasil karena itu tidak bisa dibandingkan langsung dengan versi lama.
    Sisa overlap hanyalah sidelap path/row bertetangga di lintang tinggi; jumlahnya dilaporkan
    di `*_split.json`. Set `SPLIT_MODE = "scene"` untuk mereplikasi split lama.
12. **Pemilihan checkpoint dengan val mIoU** (`SELECT_METRIC`), bukan val loss.
13. **`geo_compat.py` diperbaiki untuk TorchGeo 0.8.1** (index GeoDataFrame & kolom `filepath_1`
    pada L8 Biome); sebelumnya daftar scene kosong sehingga bootstrap per scene tidak tersedia.
14. **Patch validasi hanya yang berisi data.** 400 patch tetap dipilih dari kandidat acak
    yang punya piksel valid, di-cache di `data/{ds}_val_bboxes_*.pkl` dan dipakai semua run.
    Tanpa ini ~85% patch validasi acak hanya Fill dan val mIoU dihitung dari segelintir patch.
15. **Training minimal 20 epoch & seleksi checkpoint mulai epoch 5** (`min_epochs`,
    `select_from_epoch` di `TRAIN_DEFAULTS`). Val mIoU baseline random naik-turun ±10 poin
    antar epoch; tanpa aturan ini run random WCE berhenti di epoch 11 dengan checkpoint epoch 1.
    Hasil 6 run dengan protokol sebelumnya disimpan di `runs/_backup_old_protocol_20261101/`.
16. **Partial label untuk anotasi shadow yang hilang** (`partial` di `RunConfig`, Fase 6). Run lama
    (`partial="none"`) namanya tidak berubah.
17. **Data dibaca di CRS asli (30 m), tidak lagi lewat TorchGeo** (`src/data.py`, 2026-11-02).
    TorchGeo 0.8.1 menyatukan semua scene ke EPSG:3857 dengan resolusi file pertama (L8 105 m,
    L7 52 m): scene tropis dilatih/diuji di ~105 m, scene Antartika di-upsample, dan bobot piksel
    antar scene berbeda sampai ~60x (3 scene Antartika = 73% piksel test L8). Sekarang patch
    dibaca langsung dari raster asli; key patch = (scene, baris, kolom).
18. **Split lokasi distratifikasi per grup** (biome L8, zona lintang L7): L8 8/2/2 scene per biome,
    L7 117/46/43. Split sebelumnya tidak menaruh biome barren dan wetlands di test L8.
    Hasil, index, dan statistik dengan pipeline lama ada di `runs/_backup_epsg3857_20261102/` dan
    `data/_backup_epsg3857_20261102/`.
19. **Desain ringkas (Fase 7, 2026-11-03)**: hanya U-Net + WCE; random, lalu partial
    label {none, all, geom} x sampling {non-empty, oversample}, 5 seed. CBAM/hybrid (Fase 1) hanya
    ada untuk seed 42 (kedua dataset) dan seed 123 sebagian untuk L7 (lampiran, satu seed). Fase 2–4
    belum dijalankan; sebagai gantinya Fase 8 = backbone pretrained (ResNet-18 SSL4EO-L MoCo,
    DeepLabV3+ R50 ImageNet) x partial {none, geom}, sampling non-empty, 3 seed. Rantai:
    `D:\Reexperiment\grsl_chain.ps1` (Fase 7) lalu `grsl_chain_p8.ps1` (menunggu Fase 7). Filter geometri pasca-inferensi (`scripts/geom_filter.py`) hanya menaikkan IoU
    shadow L8 0,1–0,8 poin (seed 42) dan tidak dipakai sebagai metode.
20. **Analisis utama (2026-11-05)**: Fase 7 & 8 selesai (rantai `grsl_chain.ps1`, Fase 8 ikut di
    rantai yang sama). `analyze_grsl.py` -> `analysis/grsl/` menghitung tiap run dalam 4 view:
    semua scene, scene beranotasi shadow, metrik sadar-geometri (`geom_filter.py` varian
    `geomaware`: di scene tanpa anotasi shadow, Clear di jalur bayangan awan GT diabaikan), dan
    SPARCS (`eval_sparcs.py`, label lengkap; `prepare_sparcs.py` dijalankan ulang untuk split baru,
    tidak ada scene SPARCS yang sama dengan Biome). Uji t berpasangan per seed + Holm per
    keluarga x view x metrik, CI bootstrap per scene. Metrik sadar-geometri dan koreksi prior
    (run oversample tanpa partial) dijalankan lewat `D:\Reexperiment\geomaware_chain.ps1` lalu
    `prior_chain.ps1` (keduanya via WMI).
21. **Fase 9 (2026-11-05)**: sensitivitas p oversampling (0.2/0.4/0.8; p = 0.6 dari Fase 7), U-Net WCE
    tanpa partial label, 3 seed. Rantai `D:\Reexperiment\grsl_p9_chain.ps1` (WMI) juga
    menjalankan eval SPARCS dan metrik sadar-geometri untuk run ini. `analyze_grsl.py` keluarga F4.
22. **Fase 10 (2026-11-05):** faktorial 2×2×2 sampling {random, oversample} × CBAM × loss {WCE, hybrid}
    plus CB loss, OHEM, class-aware, 3 seed, L8 + L7 (54 run; 12 sudah ada). Rantai
    `D:\Reexperiment\grsl_p10_chain.ps1` menunggu rantai geomaware dan prior, lalu menjalankan training,
    eval SPARCS, dan metrik sadar-geometri; menulis `grsl_p10.done` saat selesai. `analyze_grsl.py` kini
    memuat semua loss dan CBAM (keluarga F5 CBAM × loss, F6 baseline imbalance).
23. **Fase 11 (2026-11-07):** OHEM Fase 10 memakai CE tanpa bobot kelas dan kolaps (IoU shadow val ≈ 0).
    Loss baru `ohem_w` = OHEM dengan bobot kelas WCE; 6 run (random, 3 seed, L8 + L7). Rantai
    `D:\Reexperiment\grsl_p11_chain.ps1` menunggu Fase 10, lalu training, SPARCS, metrik geometri;
    menulis `grsl_p11.done`. `analyze_grsl.py` F6 memuat "weighted OHEM".

## Hal yang perlu kamu verifikasi sendiri

- **Normalisasi untuk encoder pretrained.** Default `zscore`. Cek konvensi input SSL4EO-L di
  dokumentasi TorchGeo, lalu kalau perlu ubah default `norm` di `RunConfig` sebelum Fase 4
  (jangan `--set norm=...`: nama run berubah dan `analyze.py` tidak menemukannya).
- **Bobot pretrained.** Cek `model_load_info` di JSON hasil Fase 4: `missing_keys` seharusnya
  hanya `fc` (atau kosong), dan `conv1_adapted` sebaiknya `false`.
- **Overlap footprint test–train** di `*_split.json` (sisa sidelap path/row bertetangga). Laporkan di paper.
- **Biome per split** (`groups` di `*_split.json`). Split distratifikasi per biome/zona (lihat #18).
- **Bobot class-balanced (`cb`)** memakai β = 1 − 1/n_max karena β = 0,9999 jenuh untuk hitungan piksel.
  Nilai bobot tercetak di log; laporkan di paper.
