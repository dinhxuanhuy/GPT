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

EOS_ID = 50256


def load_state_dict_flexible(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]

        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]

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


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    context_length: int,
    device,
):
    """
    Greedy decoding giống logic:

    Question: ...
    Answer:
    """

    model.eval()

    prompt = f"Question: {question}\nAnswer:"

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        # Nếu prompt + output gần vượt context thì dừng
        if input_ids.shape[1] >= context_length:
            break

        idx_cond = input_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # Greedy search: lấy token có xác suất cao nhất
        next_id = torch.argmax(logits, dim=-1, keepdim=True)

        # Dừng nếu model sinh EOS
        if next_id.item() == EOS_ID:
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    output_text = tokenizer.decode(input_ids[0].tolist())

    # Chỉ lấy phần sau Answer:
    if "Answer:" in output_text:
        answer = output_text.split("Answer:", 1)[-1].strip()
    else:
        answer = output_text.strip()

    # Xóa token đặc biệt nếu còn sót
    answer = answer.replace("<|endoftext|>", "").strip()

    return answer, prompt, output_text


def main():
    st.set_page_config(
        page_title="Custom GPT-2 Demo",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Custom GPT-2 Interview Demo")
    st.caption("Ask questions and get answers from your fine-tuned GPT-2 model.")

    with st.sidebar:
        st.header("Model Settings")

        load_mode = st.radio(
            "Model source",
            ["Hugging Face Hub", "Local checkpoint"],
            index=0,
        )

        repo_id = ""
        checkpoint_path = ""

        if load_mode == "Hugging Face Hub":
            repo_id = st.text_input(
                "Repo ID",
                value="dinhxuanhuy/scratch_gpt2_tuned_interview",
            )
            st.caption("The repository must include config.json and model.safetensors.")

        else:
            checkpoint_path = st.text_input("Checkpoint path", value="model.pt")

            context_length = st.number_input(
                "Context length",
                min_value=8,
                max_value=4096,
                value=256,
                step=8,
            )

            emb_dim = st.selectbox(
                "Embedding dim",
                [128, 256, 384, 512, 768],
                index=4,
            )

            n_heads = st.selectbox(
                "Heads",
                [2, 4, 6, 8, 12],
                index=4,
            )

            n_layers = st.selectbox(
                "Layers",
                [2, 4, 6, 8, 12],
                index=4,
            )

            drop_rate = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
            )

            qkv_bias = st.checkbox("Use QKV bias", value=False)

        st.header("Generation Settings")

        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=1,
            max_value=512,
            value=80,
        )

        st.caption("Decoding method: Greedy Search")

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

    context_length = int(cfg.get("context_length", DEFAULT_CONFIG["context_length"]))

    st.subheader("Ask a question")

    user_question = st.text_area(
        "Your question",
        value="who are you?",
        height=100,
        placeholder="Example: what is your gpa?",
    )

    run = st.button(
        "Generate Response",
        type="primary",
        use_container_width=True,
    )

    if run:
        if not user_question.strip():
            st.warning("Please enter a question.")
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with st.spinner("Generating answer..."):
            try:
                answer, used_prompt, full_output = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=user_question.strip(),
                    max_new_tokens=int(max_new_tokens),
                    context_length=context_length,
                    device=device,
                )

                st.subheader("Response")
                st.write(answer)

                with st.expander("Show prompt and raw output"):
                    st.markdown("**Prompt used:**")
                    st.code(used_prompt)

                    st.markdown("**Raw model output:**")
                    st.code(full_output)

            except Exception as exc:
                st.exception(exc)


if __name__ == "__main__":
    main()