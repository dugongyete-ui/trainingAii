Jalankan:

bash install.sh
Script ini akan:

Mencari Python 3.11.
Memasang PyTorch versi CPU agar tidak mengunduh CUDA besar.
Memasang semua dependency dari requirements.txt.
Mengunduh model multilingual Qwen jika belum tersedia.
Memeriksa PyTorch, Transformers, dan sintaks kode.
Menampilkan perintah untuk menjalankan AI.
Setelah instalasi selesai, jalankan:

python3.11 eval_llm.py --device cpu
Pilih:

[1] Input manual
Lalu tulis pertanyaan dalam Bahasa Indonesia.