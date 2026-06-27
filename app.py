import os
import json
from typing import Dict, Any

import streamlit as st
import torch
import tiktoken
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from GPT2 import GPTModel


DEFAULT_CONFIG: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def load_state_dict_flexible(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Checkpoint format is not supported.")


def load_model_from_hf_repo(repo_id: str):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    weight_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = GPTModel(config)
    state_dict = load_file(weight_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


@st.cache_resource
def build_model_from_local(cfg: Dict[str, Any], checkpoint_path: str):
    model = GPTModel(cfg)
    loaded_checkpoint = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = load_state_dict_flexible(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        loaded_checkpoint = True

    model.eval()
    return model, loaded_checkpoint


@st.cache_resource
def build_model_from_hf(repo_id: str):
    model, config = load_model_from_hf_repo(repo_id)
    return model, config


def generate_text(
    model,
    idx,
    max_new_tokens: int,
    context_size: int,
    temperature: float,
    top_k=None,
    eos_id=50256,
):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    st.set_page_config(page_title="Custom GPT-2 Demo", page_icon="🧠", layout="wide")

    st.title("Custom GPT-2 Streamlit Demo")
    st.caption("Interface powered by the model architecture in GPT2.py")

    with st.sidebar:
        st.header("Model Settings")
        load_mode = st.radio("Model source", ["Hugging Face Hub", "Local checkpoint"], index=0)

        repo_id = ""
        checkpoint_path = ""

        if load_mode == "Hugging Face Hub":
            repo_id = st.text_input("Repo ID", value="dinhxuanhuy/scratch_gpt2_tuned_interview")
            st.caption("The repository must include config.json and model.safetensors.")
        else:
            checkpoint_path = st.text_input("Checkpoint path", value="model.pt")
            context_length = st.number_input("Context length", min_value=8, max_value=4096, value=256, step=8)
            emb_dim = st.selectbox("Embedding dim", [128, 256, 384, 512, 768], index=1)
            n_heads = st.selectbox("Heads", [2, 4, 6, 8, 12], index=1)
            n_layers = st.selectbox("Layers", [2, 4, 6, 8, 12], index=1)
            drop_rate = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
            qkv_bias = st.checkbox("Use QKV bias", value=False)

        st.header("Generation Settings")
        max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=512, value=80)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.05)
        top_k = st.slider("Top-k", min_value=0, max_value=200, value=40)

    tokenizer = get_tokenizer()

    if load_mode == "Hugging Face Hub":
        try:
            model, cfg = build_model_from_hf(repo_id)
            st.success(f"Model loaded from Hugging Face: {repo_id}")
        except Exception as exc:
            st.error("Unable to load model from Hugging Face Hub.")
            st.exception(exc)
            st.stop()
    else:
        cfg = {
            "vocab_size": DEFAULT_CONFIG["vocab_size"],
            "context_length": int(context_length),
            "emb_dim": int(emb_dim),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "drop_rate": float(drop_rate),
            "qkv_bias": bool(qkv_bias),
        }

        if cfg["emb_dim"] % cfg["n_heads"] != 0:
            st.error("emb_dim must be divisible by n_heads.")
            st.stop()

        model, loaded_checkpoint = build_model_from_local(cfg, checkpoint_path)

        if loaded_checkpoint:
            st.success(f"Checkpoint loaded: {checkpoint_path}")
        else:
            st.warning(
                "Checkpoint not found. The app is running with random weights for pipeline testing only."
            )

    prompt = "who are you?"
    
    st.subheader("Prompt")
    st.write(f"**{prompt}**")

    col1, col2 = st.columns([1, 2])
    with col1:
        run = st.button("Generate Response", type="primary", use_container_width=True)
    with col2:
        st.write("")

    if run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with st.spinner("Generating text..."):
            try:
                input_ids = tokenizer.encode(prompt)
                idx = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                top_k_value = int(top_k) if int(top_k) > 0 else None

                out = generate_text(
                    model=model,
                    idx=idx,
                    max_new_tokens=int(max_new_tokens),
                    context_size=cfg["context_length"],
                    temperature=float(temperature),
                    top_k=top_k_value,
                    eos_id=50256,
                )
                out_text = tokenizer.decode(out[0].tolist())
                st.subheader("Response")
                st.write(out_text)
            except Exception as exc:
                st.exception(exc)


if __name__ == "__main__":
    main()
