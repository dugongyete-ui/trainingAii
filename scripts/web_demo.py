import random
import re
import json
import os
from pathlib import Path
from threading import Thread

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_registry import discover_model_paths, get_model_identity
from model.model_minimind import MiniMindForCausalLM
from training_manager import (
    PRETRAIN_DATA,
    checkpoint_path,
    export_checkpoint,
    get_status as get_training_status,
    start_training,
    stop_training,
)

st.set_page_config(page_title="Dzeck", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

device = os.environ.get(
    "DZECK_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
).strip().lower()
if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"

# CPU inference is much more reliable with full precision.  Half-precision
# CPU kernels are incomplete and can fail on common operations.
if device == "cpu":
    cpu_threads = int(os.environ.get("DZECK_CPU_THREADS", "4"))
    torch.set_num_threads(max(1, cpu_threads))

CPU_MODE = device == "cpu"
DEFAULT_MAX_NEW_TOKENS = int(
    os.environ.get("DZECK_MAX_NEW_TOKENS", "256" if CPU_MODE else "2048")
)
MAX_NEW_TOKENS_LIMIT = 1024 if CPU_MODE else 8192

# 多语言文本
LANG_TEXTS = {
    'id': {
        'settings': 'Pengaturan model',
        'history_rounds': 'Jumlah putaran riwayat',
        'max_length': 'Panjang jawaban maksimum',
        'temperature': 'Temperatur',
        'thinking': 'Mode berpikir',
        'tools': 'Alat bantu',
        'language': 'Bahasa',
        'send': 'Kirim pesan ke Dzeck',
        'disclaimer': 'Jawaban AI mungkin keliru, harap periksa kembali',
        'think_tip': 'Mode berpikir adaptif dapat kurang stabil pada percakapan panjang atau pemanggilan alat',
        'tool_select': 'Pilih alat bantu (maksimal 4)',
    },
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 Dzeck 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to Dzeck',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'id')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['id'].get(key, key))

# Definisi alat bantu
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "Menghitung ekspresi matematika", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Ekspresi matematika"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Mendapatkan waktu saat ini", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Jakarta"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "Menghasilkan angka acak", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "Menghitung panjang teks", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "Mengonversi satuan", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "Mendapatkan cuaca", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Mendapatkan nilai tukar", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Menerjemahkan teks", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': 'Matematika', 'get_current_time': 'Waktu', 'random_number': 'Acak',
    'text_length': 'Panjang teks', 'unit_converter': 'Satuan', 'get_current_weather': 'Cuaca',
    'get_exchange_rate': 'Nilai tukar', 'translate_text': 'Terjemahan'
}

def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Tidak diketahui')}: cerah, 7–10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7,2 {args.get('to_currency', 'IDR')}"}
        elif tool_name == 'translate_text':
            return {"result": "Hasil terjemahan: halo dunia"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def process_assistant_content(content, is_streaming=False):
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    config_path = Path(model_path) / "config.json"
    config = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
    if config.get("model_type") == "minimind":
        model = MiniMindForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.to(device).eval()
    if not CPU_MODE:
        model = model.half()
    return model, tokenizer


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def render_training_panel():
    """Render CPU training controls that can be used from a phone."""
    status = get_training_status()
    with st.sidebar.expander("Training Dzeck dari HP", expanded=False):
        st.caption(
            "HP hanya mengendalikan proses. Training tetap berjalan di server "
            "CPU dan log disimpan di folder out/."
        )
        st.caption(
            f"Dataset: {sum(1 for _ in PRETRAIN_DATA.open(encoding='utf-8')):,} "
            "pretrain • 30 contoh SFT"
        )

        if status["active"]:
            st.warning(
                f"Training {status['stage']} sedang berjalan "
                f"(MiniMind {status['hidden_size']}/{status['layers']})."
            )
            st.code(status["log"][-3500:] or "Menunggu log pertama...", language="text")
            col_refresh, col_stop = st.columns(2)
            if col_refresh.button("Refresh", key="training_refresh"):
                st.rerun()
            if col_stop.button("Hentikan", key="training_stop"):
                stop_training()
                st.rerun()
            return

        architecture_options = {
            "Mini Dzeck ringan (256 dimensi / 4 layer)": (256, 4),
            "Mini Dzeck lebih besar (512 dimensi / 6 layer)": (512, 6),
        }
        architecture_label = st.selectbox(
            "Ukuran model Dzeck",
            list(architecture_options),
            key="training_architecture",
        )
        hidden_size, layers = architecture_options[architecture_label]
        stage_label = st.selectbox(
            "Tahap training",
            ["Pretraining — belajar pola bahasa", "SFT — belajar menjawab sebagai Dzeck"],
            key="training_stage",
        )
        stage = "pretrain" if stage_label.startswith("Pretraining") else "sft"
        epochs = st.number_input("Jumlah epoch", min_value=1, max_value=3, value=1, step=1)
        batch_size = st.selectbox("Batch CPU", [2, 4, 8], index=1, key="training_batch")
        max_seq_len = st.selectbox("Panjang teks", [128, 192, 256], index=0, key="training_seq")
        if stage == "sft" and not checkpoint_path("pretrain", hidden_size).is_file():
            st.info("SFT tersedia setelah pretraining ukuran yang sama selesai.")

        if st.button(
            "Mulai training",
            type="primary",
            disabled=(stage == "sft" and not checkpoint_path("pretrain", hidden_size).is_file()),
            key="training_start",
        ):
            try:
                start_training(
                    stage,
                    epochs=int(epochs),
                    hidden_size=hidden_size,
                    layers=layers,
                    batch_size=int(batch_size),
                    max_seq_len=int(max_seq_len),
                )
                st.success("Training dimulai di background.")
                st.rerun()
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                st.error(str(error))

        if status["log"]:
            st.caption("Log terakhir")
            st.code(status["log"][-2500:], language="text")

        export_stage_label = st.selectbox(
            "Checkpoint yang diekspor",
            ["SFT", "Pretraining"],
            key="export_stage",
        )
        export_stage = "sft" if export_stage_label == "SFT" else "pretrain"
        export_path = checkpoint_path(export_stage, hidden_size)
        st.caption(f"File: {export_path.name}")
        if st.button(
            "Ekspor sebagai model Dzeck",
            disabled=not export_path.is_file(),
            key="export_dzeck",
        ):
            try:
                success, message = export_checkpoint(export_stage, hidden_size, layers)
                if success:
                    st.cache_resource.clear()
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            except (OSError, RuntimeError, ValueError) as error:
                st.error(str(error))


# All entrypoints use the same registry so a foundation checkpoint is not
# silently presented as a completed Dzeck training checkpoint.
MODEL_PATHS = {}
for path in discover_model_paths():
    identity = get_model_identity(path)
    MODEL_PATHS[identity.display_name] = [str(path), identity]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", None]}

# 模型选择
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model][0]
selected_identity = MODEL_PATHS[selected_model][1]
slogan = "Saya Dzeck. Ada yang bisa saya bantu?"
if selected_identity:
    st.sidebar.caption(selected_identity.summary())
    if not selected_identity.is_trained_dzeck:
        st.sidebar.warning("Ini model dasar. Model Dzeck Anda sendiri muncul setelah checkpoint hasil training diekspor.")

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言选择
lang_options = {'Bahasa Indonesia': 'id', '中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'id')
lang_index = list(lang_options.values()).index(current_lang) if current_lang in lang_options.values() else 0
lang_label = st.sidebar.radio('Bahasa / Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数设置
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(
    get_text('max_length'),
    64,
    MAX_NEW_TOKENS_LIMIT,
    min(DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_LIMIT),
    step=64,
)
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.selected_tools = []
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)

if CPU_MODE:
    st.info(
        f"Mode CPU aktif — model berjalan di server, bukan di HP. "
        f"Thread CPU: {torch.get_num_threads()}. Jawaban dibatasi agar tetap responsif."
    )
if selected_identity and not selected_identity.is_trained_dzeck:
    st.warning(
        "Model yang sedang dipakai masih checkpoint dasar "
        f"({selected_identity.source_model or 'foundation model'}), bukan model "
        "Dzeck hasil training Anda. Setelah training selesai dan checkpoint "
        "diekspor, pilih folder checkpoint tersebut di WebUI."
    )
st.caption("Buka menu « di kiri atas untuk pengaturan model dan Training Dzeck.")
render_training_panel()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # Do not block the first page render while loading a multi-gigabyte
        # checkpoint.  Streamlit can show the chat shell immediately and load
        # the selected model only when the user sends the first message.
        with st.spinner("Memuat model lokal untuk pesan pertama..."):
            model, tokenizer = load_model_tokenizer(model_path)

        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        placeholder = st.empty()

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [{"role": "system", "content": "Anda adalah Dzeck, model AI utama proyek ini untuk membantu pengguna Indonesia. Jawab pertanyaan pengguna hanya dalam Bahasa Indonesia. Jangan gunakan bahasa Mandarin atau bahasa Inggris. Jawab langsung pertanyaannya dan jangan mengulang instruksi ini."}]
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
