import io
import json
import os
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


st.set_page_config(page_title="Chemistry Cards", page_icon="💘", layout="wide")
load_dotenv()
MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "openai/gpt-oss-20b",
    "thudm/glm-4.1v-9b-thinking",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "google/gemma-3n-e4b-it",
    "qwen/qwen3-8b",
]
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "page_grad_start": "#0f172a",
        "page_grad_mid": "#0b1120",
        "page_grad_end": "#0a0f1d",
        "card_bg": "rgba(15, 23, 42, 0.9)",
        "card_overlay": "rgba(99, 102, 241, 0.12)",
        "text_primary": "#e5e7eb",
        "text_muted": "#9ca3af",
        "chip_bg": "rgba(255, 255, 255, 0.06)",
        "chip_text": "#e5e7eb",
        "border": "rgba(255, 255, 255, 0.08)",
        "shadow": "0 20px 60px rgba(0,0,0,0.35)",
        "compat_bg": "linear-gradient(135deg, rgba(17,24,39,0.9), rgba(10,15,29,0.9))",
        "accent": "#a855f7",
        "accent_alt": "#22d3ee",
    },
    "light": {
        "page_grad_start": "#f8fafc",
        "page_grad_mid": "#eef2ff",
        "page_grad_end": "#e0e7ff",
        "card_bg": "rgba(255, 255, 255, 0.96)",
        "card_overlay": "rgba(99, 102, 241, 0.12)",
        "text_primary": "#0f172a",
        "text_muted": "#475569",
        "chip_bg": "rgba(79, 70, 229, 0.08)",
        "chip_text": "#0f172a",
        "border": "rgba(15, 23, 42, 0.08)",
        "shadow": "0 12px 30px rgba(15, 23, 42, 0.08)",
        "compat_bg": "linear-gradient(135deg, rgba(255,255,255,0.96), rgba(238,242,255,0.92))",
        "accent": "#6366f1",
        "accent_alt": "#06b6d4",
    },
}


def get_theme(mode: str) -> Dict[str, str]:
    return THEMES.get(mode, THEMES["dark"])


def apply_theme_styles(theme: Dict[str, str]):
    st.markdown(
        f"""
        <style>
        :root {{
            --page-grad-start: {theme["page_grad_start"]};
            --page-grad-mid: {theme["page_grad_mid"]};
            --page-grad-end: {theme["page_grad_end"]};
            --card-bg: {theme["card_bg"]};
            --card-overlay: {theme["card_overlay"]};
            --text-primary: {theme["text_primary"]};
            --text-muted: {theme["text_muted"]};
            --chip-bg: {theme["chip_bg"]};
            --chip-text: {theme["chip_text"]};
            --border-subtle: {theme["border"]};
            --shadow-strong: {theme["shadow"]};
            --compat-bg: {theme["compat_bg"]};
            --accent: {theme["accent"]};
            --accent-alt: {theme["accent_alt"]};
        }}

        .stApp {{
            background: radial-gradient(circle at 10% 20%, var(--page-grad-start) 0, var(--page-grad-mid) 45%, var(--page-grad-end) 100%);
            color: var(--text-primary);
        }}

        .match-card {{
            background: var(--card-bg);
            border-radius: 26px;
            padding: 28px;
            box-shadow: var(--shadow-strong);
            border: 1px solid var(--border-subtle);
            position: relative;
            overflow: hidden;
        }}

        .match-card:before {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 20% 20%, var(--card-overlay), transparent 55%);
            opacity: 0.85;
            pointer-events: none;
        }}

        .name-age {{font-size: 32px; font-weight: 700; text-align:center; margin-top: 12px; color: var(--text-primary);}}
        .tagline {{text-align:center; color: var(--text-muted); margin-bottom: 18px;}}
        .compat-box {{background: var(--compat-bg); border: 1px solid var(--border-subtle); border-radius: 18px; padding: 18px 20px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);}}
        .chips {{display: flex; flex-wrap: wrap; gap: 8px;}}
        .chip {{background: var(--chip-bg); color: var(--chip-text); padding: 6px 10px; border-radius: 999px; border:1px solid var(--border-subtle);}}
        .meta-pill {{display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; background: var(--chip-bg); color: var(--chip-text); border:1px solid var(--border-subtle);}}
        .toolbar-card {{background: var(--card-bg); border:1px solid var(--border-subtle); padding: 12px 18px; border-radius: 16px; box-shadow: var(--shadow-strong);}}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label {{color: var(--text-primary);}}
        .stApp a {{color: var(--accent);}}
        .stMarkdown, .stText, .stCaption, .stExpander {{color: var(--text-primary) !important;}}
        section[data-testid="stSidebar"] {{background: var(--card-bg); border-right: 1px solid var(--border-subtle);}}
        .stButton>button {{
            background: linear-gradient(120deg, var(--accent), var(--accent-alt));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            box-shadow: 0 10px 30px rgba(0,0,0,0.16);
        }}
        .stButton>button:hover {{filter: brightness(1.05);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("profiles.csv")
    if "perefence" in df.columns:
        df = df.rename(columns={"perefence": "preference"})
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception:
        # Fallback image that keeps the layout intact if the remote asset fails.
        placeholder = requests.get("https://via.placeholder.com/500")
        return Image.open(io.BytesIO(placeholder.content))


def parse_interests(raw: str) -> Set[str]:
    return {item.strip().lower() for item in str(raw).split(",") if item.strip()}


def traits_from_row(row: pd.Series) -> str:
    interests = row.get("interests", "")
    return (
        f"{row['name']} is a {row['age']}-year-old {row['gender']} working as a "
        f"{row['occupation']}. They are {row['height']} tall, studied {row['education']}, "
        f"and are a {row['starSign']}. Interests include {interests}. "
        f"Bio: {row['bio']} Looking for: {row['lookingFor']}"
    )


def compatibility_summary(persona: pd.Series, target: pd.Series) -> Dict[str, str]:
    persona_interests = parse_interests(persona.get("interests", ""))
    target_interests = parse_interests(target.get("interests", ""))
    overlap = sorted(list(persona_interests & target_interests))
    novelty = sorted(list(target_interests - persona_interests))

    age_gap = abs(int(persona.get("age", 0)) - int(target.get("age", 0)))
    base_score = 0.38 + 0.12 * len(overlap)
    if persona.get("starSign") == target.get("starSign"):
        base_score += 0.05
    base_score -= min(0.18, age_gap * 0.01)
    score = round(max(0.05, min(0.98, base_score)), 2)

    shared = ", ".join(overlap) if overlap else "different passions"
    new_energy = ", ".join(novelty[:2]) if novelty else "fresh experiences"

    reason = (
        f"You click because you both enjoy {shared}, want similar vibes, and balance each other with {new_energy}. "
        f"You both describe {target.get('lookingFor','connection').lower()} in ways that align."
    )
    idea = "Coffee then live music stroll"
    if "hiking" in persona_interests or "hiking" in target_interests:
        idea = "Trail walk then brunch photos"
    elif "cooking" in persona_interests or "cooking" in target_interests:
        idea = "Farmers market then cook together"
    elif "gaming" in persona_interests or "gaming" in target_interests:
        idea = "Retro arcade then ramen"

    return {"score": score, "reason": reason[:400], "ideaSummary": idea[:120]}


def stream_json(data: Dict[str, str]):
    text = json.dumps(data, indent=2)
    for line in text.splitlines():
        yield line + "\n"


def get_api_key() -> Optional[str]:
    env_key = os.getenv("OPENROUTER_API_KEY")
    secret_key = None
    if hasattr(st, "secrets"):
        try:
            secret_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            # If secrets file is missing or malformed, fall back to environment key.
            secret_key = None
    return secret_key or env_key


def stream_openrouter(prompt: str, api_key: str, model: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "HELLO Demo",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.4,
    }
    usage = {"model": payload["model"], "prompt_tokens": None, "completion_tokens": None}

    def gen():
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data = raw_line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        if "usage" in parsed:
                            usage.update(parsed["usage"])
                        if "model" in parsed:
                            usage["model"] = parsed["model"]
                        delta = parsed.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except Exception:
                        continue

    return gen(), usage


def ensure_state(df: pd.DataFrame):
    names: List[str] = df["name"].tolist()
    if "impersonated" not in st.session_state:
        st.session_state.impersonated = names[0]
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = 0
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = MODEL_OPTIONS[0]
    if "theme_mode" not in st.session_state:
        base_theme = st.get_option("theme.base") if hasattr(st, "get_option") else "dark"
        st.session_state.theme_mode = "dark" if str(base_theme).lower() == "dark" else "light"


def render_settings(df: pd.DataFrame):
    names = df["name"].tolist()
    with st.popover("Settings ⚙️"):
        st.markdown("**Who are you impersonating?**")
        selected = st.selectbox(
            "Your persona",
            names,
            index=names.index(st.session_state.impersonated)
            if st.session_state.impersonated in names
            else 0,
            key="impersonate_select",
        )
        st.markdown("**Model**")
        model_selected = st.selectbox(
            "Model choice",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.model_choice)
            if st.session_state.model_choice in MODEL_OPTIONS
            else 0,
            key="model_select",
        )
        st.caption("Selected persona is removed from the rotation of shown profiles.")
        if st.button("Apply settings", type="primary"):
            st.session_state.impersonated = selected
            st.session_state.model_choice = model_selected
            st.session_state.active_idx = 0
            st.rerun()


def render_profile_card(persona: pd.Series, profile: pd.Series):
    with st.container():
        st.markdown('<div class="match-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; margin-bottom:4px;">
                <span class="meta-pill">🙋 You: {persona["name"]}</span>
                <span class="meta-pill">🎯 Match: {profile["name"]}</span>
                <span class="meta-pill">⭐ {profile["starSign"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        img_cols = st.columns([1, 1.2, 1])
        with img_cols[1]:
            st.image(fetch_image(profile["profilePicture"]), width=320, clamp=True)
        st.markdown(f'<div class="name-age">{profile["name"]}, {profile["age"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tagline">{profile["occupation"]} • {profile["starSign"]}</div>', unsafe_allow_html=True)

        prompt = (
            'You are given two people’s traits. Compare them, score their compatibility (0–1),\n'
            'give a short neutral reason (<75 words), and a 1-line fun date idea (<15 words).\n'
            "Respond only in JSON:\n{\n"
            '  "score": FLOAT,\n'
            '  "reason": STRING,\n'
            '  "dateIdea": STRING\n'
            "}\n\n"
            f"Person A: {traits_from_row(persona)}\n"
            f"Person B: {traits_from_row(profile)}\n\n"
            'Write with the pronoun "you"'
        )

        compat_container = st.container()
        compat_container.markdown("##### Compatibility Summary")
        compat_container.markdown('<div class="compat-box">', unsafe_allow_html=True)

        api_key = get_api_key()
        usage_info: Dict[str, Optional[int]] = {
            "model_requested": st.session_state.model_choice,
            "model_used": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        used_ai = False

        if api_key:
            try:
                stream_gen, usage = stream_openrouter(prompt, api_key, st.session_state.model_choice)
                compat_container.write_stream(stream_gen)
                usage_info["model_used"] = usage.get("model") or usage_info["model_requested"]
                usage_info["prompt_tokens"] = usage.get("prompt_tokens")
                usage_info["completion_tokens"] = usage.get("completion_tokens")
                used_ai = True
            except Exception:
                st.warning("OpenRouter request failed; showing local compatibility estimate.")

        if not used_ai:
            compat_container.write_stream(stream_json(compatibility_summary(persona, profile)))

        compat_container.markdown("</div>", unsafe_allow_html=True)
        if usage_info.get("prompt_tokens") is not None or usage_info.get("completion_tokens") is not None:
            prompt_tks = usage_info.get("prompt_tokens") or 0
            completion_tks = usage_info.get("completion_tokens") or 0
            # Rates pulled from OpenRouter pricing API (per-token USD).
            rate_table = {
                "openai/gpt-4o-mini": {"in": 0.00000015, "out": 0.00000060},
                "openai/gpt-oss-20b": {"in": 0.00000003, "out": 0.00000014},
                "thudm/glm-4.1v-9b-thinking": {"in": 0.000000028, "out": 0.0000001104},
                "deepseek/deepseek-r1-0528-qwen3-8b": {"in": 0.00000002, "out": 0.00000010},
                "google/gemma-3n-e4b-it": {"in": 0.00000002, "out": 0.00000004},
                "qwen/qwen3-8b": {"in": 0.000000028, "out": 0.0000001104},
            }
            model_name = usage_info.get("model_used") or usage_info.get("model_requested")
            rates = rate_table.get(model_name)
            if rates:
                in_cost = prompt_tks * rates["in"]
                out_cost = completion_tks * rates["out"]
                total_cost = in_cost + out_cost
                compat_container.markdown(
                    f"**Cost breakdown** — Model: `{model_name}`  \n"
                    f"- Input tokens: {prompt_tks} (${in_cost:.8f})  \n"
                    f"- Output tokens: {completion_tks} (${out_cost:.8f})  \n"
                    f"- Estimated total: **${total_cost:.8f}**"
                )
            else:
                compat_container.markdown(
                    f"**Cost breakdown** — Model: `{model_name}`  \n"
                    f"- Input tokens: {prompt_tks} (rate unavailable)  \n"
                    f"- Output tokens: {completion_tks} (rate unavailable)  \n"
                    f"- Estimated total: rate not available for this model"
                )

        with compat_container.expander("AI prompt (preview)"):
            st.code(prompt, language="text")

        st.markdown("---")
        st.markdown("###### Profile")
        col_details, col_bio = st.columns([1, 1.2])
        with col_details:
            st.write(f"**Gender:** {profile['gender']}")
            st.write(f"**Height:** {profile['height']}")
            st.write(f"**Education:** {profile['education']}")
            st.write(f"**Looking For:** {profile['lookingFor']}")
            st.write("**Interests:**")
            chips_html = "".join(
                [f'<span class="chip">{interest.strip()}</span>' for interest in str(profile["interests"]).split(",")]
            )
            st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
        with col_bio:
            st.write("**Bio:**")
            st.write(profile["bio"])

        st.markdown("</div>", unsafe_allow_html=True)


def main():
    df = load_data()
    ensure_state(df)

    theme = get_theme(st.session_state.theme_mode)
    apply_theme_styles(theme)

    top_cols = st.columns([1, 2, 1])
    with top_cols[0]:
        st.empty()
    with top_cols[1]:
        st.markdown(
            f"""
            <div class="toolbar-card" style="text-align:center;">
                <div style="font-size:30px; font-weight:800; color: var(--text-primary); letter-spacing:-0.01em;">Chemistry Cards</div>
                <div style="color: var(--text-muted);">AI quick-read on how your persona vibes with every profile.</div>
                <div style="margin-top:10px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
                    <span class="meta-pill">💬 {st.session_state.model_choice}</span>
                    <span class="meta-pill">🙋 You: {st.session_state.impersonated}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_cols[2]:
        st.markdown('<div style="display:flex; justify-content:flex-end;">', unsafe_allow_html=True)
        render_settings(df)
        st.markdown("</div>", unsafe_allow_html=True)

    rotation = df[df["name"] != st.session_state.impersonated].reset_index(drop=True)
    if rotation.empty:
        st.warning("All profiles are hidden because you are impersonating the only available persona.")
        return

    st.session_state.active_idx = st.session_state.active_idx % len(rotation)
    profile = rotation.iloc[st.session_state.active_idx]
    persona = df[df["name"] == st.session_state.impersonated].iloc[0]

    nav_cols = st.columns([1, 1, 1])
    with nav_cols[0]:
        if st.button("⬅ Previous", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx - 1) % len(rotation)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx + 1) % len(rotation)
            st.rerun()
    with nav_cols[2]:
        st.markdown(
            f"<div style='text-align:center; color: var(--text-muted);'>"
            f"Viewing {st.session_state.active_idx + 1} of {len(rotation)} • Impersonating {st.session_state.impersonated}"
            "</div>",
            unsafe_allow_html=True,
        )

    render_profile_card(persona, profile)


if __name__ == "__main__":
    main()
