Jalankan:

bash install.sh
Script ini akan:

Mencari Python 3.11.
Memasang PyTorch versi CPU agar tidak mengunduh CUDA besar.
Memasang semua dependency dari requirements.txt.
Mengunduh fondasi model besar Qwen2.5-3B-Instruct jika belum tersedia.
Memeriksa PyTorch, Transformers, dan sintaks kode.
Menampilkan perintah untuk menjalankan AI.
Setelah instalasi selesai, jalankan:

python3.11 eval_llm.py --device cpu
Pilih:

[1] Input manual
Lalu tulis pertanyaan dalam Bahasa Indonesia.

## Memahami nama model

- **Dzeck Large ID** adalah nama model/API di proyek ini.
- **Qwen2.5-3B-Instruct** adalah model dasar yang dipakai sebelum training Dzeck.
- Folder `dzeck-small-id` yang lama adalah salinan Qwen2.5-0.5B, bukan model Dzeck
  hasil training sendiri. Folder itu tetap dipertahankan agar kompatibel, tetapi
  tidak lagi menjadi pilihan default.
- Model yang benar-benar menjadi milik Anda adalah checkpoint yang dihasilkan
  setelah menjalankan pretraining/SFT/LoRA dan mengekspornya ke format Transformers.

Untuk memakai model lain tanpa mengubah kode:

```bash
DZECK_MODEL_PATH=/path/ke/checkpoint-Anda python3.11 eval_llm.py --device cpu
```