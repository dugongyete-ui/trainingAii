Jalankan:

bash install.sh
Script ini akan:

Mencari Python 3.11.
Memasang PyTorch versi CPU agar tidak mengunduh CUDA besar.
Memasang semua dependency dari requirements.txt.
Mengunduh fondasi model CPU-friendly Qwen2.5-0.5B-Instruct jika belum tersedia.
Memeriksa PyTorch, Transformers, dan sintaks kode.
Menampilkan perintah untuk menjalankan AI.
Setelah instalasi selesai, jalankan:

python3.11 eval_llm.py --device cpu
Pilih:

[1] Input manual
Lalu tulis pertanyaan dalam Bahasa Indonesia.

## Memahami nama model

- **Dzeck Large ID** adalah nama model/API di proyek ini.
- **Qwen2.5-0.5B-Instruct** adalah model dasar ringan yang dipakai sebelum training Dzeck.
- **Qwen2.5-3B-Instruct** tetap dapat dipakai sebagai model besar jika checkpoint-nya lengkap.
- Folder `dzeck-small-id` yang lama adalah salinan Qwen2.5-0.5B, bukan model Dzeck
  hasil training sendiri. Folder itu tetap dipertahankan agar kompatibel, tetapi
  tidak lagi menjadi pilihan default.
- Model yang benar-benar menjadi milik Anda adalah checkpoint yang dihasilkan
  setelah menjalankan pretraining/SFT/LoRA dan mengekspornya ke format Transformers.

Untuk memakai model lain tanpa mengubah kode:

```bash
DZECK_MODEL_PATH=/path/ke/checkpoint-Anda python3.11 eval_llm.py --device cpu
```

## Versi web untuk HP

Proyek ini juga memiliki WebUI berbasis Streamlit. HP hanya dipakai untuk
membuka browser; model dan proses CPU tetap berjalan di workspace/server.

Jalankan dari Shell:

```bash
bash run_web.sh
```

Setelah proses berjalan, buka **Preview/Webview** di Replit. Jika workspace
memiliki workflow web, cukup jalankan workflow tersebut.

`run_web.sh` otomatis memilih Qwen 0.5B jika foldernya tersedia. Pengaturan CPU
yang dapat dipakai:

```bash
DZECK_DEVICE=cpu DZECK_CPU_THREADS=4 DZECK_MAX_NEW_TOKENS=256 bash run_web.sh
```

`DZECK_MAX_NEW_TOKENS` yang lebih kecil membuat jawaban lebih cepat. WebUI
tetap memakai model lokal di folder proyek dan tidak mengirim percakapan ke
layanan AI eksternal.