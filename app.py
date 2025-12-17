import base64
import io
import json
import os
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
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

MODEL_COSTS = {
    "openai/gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.6},
    "openai/gpt-oss-20b": {"input_per_million": 0.15, "output_per_million": 0.6},
    "thudm/glm-4.1v-9b-thinking": {"input_per_million": 0.2, "output_per_million": 0.8},
    "deepseek/deepseek-r1-0528-qwen3-8b": {"input_per_million": 0.1, "output_per_million": 0.4},
    "google/gemma-3n-e4b-it": {"input_per_million": 0.1, "output_per_million": 0.4},
    "qwen/qwen3-8b": {"input_per_million": 0.1, "output_per_million": 0.4},
}

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "page_grad_start": "#1a1a2e",
        "page_grad_mid": "#16213e",
        "page_grad_end": "#0f3460",
        "card_bg": "rgba(255, 255, 255, 0.08)",
        "card_overlay": "rgba(255, 20, 147, 0.15)",
        "text_primary": "#ffffff",
        "text_muted": "#b0b0b0",
        "chip_bg": "rgba(255,255,255,0.1)",
        "chip_text": "#ffffff",
        "border": "rgba(255,255,255,0.1)",
        "shadow": "0 8px 32px rgba(0,0,0,0.4)",
        "compat_bg": "linear-gradient(135deg, rgba(255,20,147,0.9), rgba(138,43,226,0.9))",
        "accent": "#ff1493",
        "accent_alt": "#9370db",
    },
    "light": {
        "page_grad_start": "#ffe4e1",
        "page_grad_mid": "#ffb6c1",
        "page_grad_end": "#ff69b4",
        "card_bg": "rgba(255,255,255,0.95)",
        "card_overlay": "rgba(255,20,147,0.1)",
        "text_primary": "#333333",
        "text_muted": "#666666",
        "chip_bg": "rgba(255,20,147,0.1)",
        "chip_text": "#333333",
        "border": "rgba(255,20,147,0.2)",
        "shadow": "0 4px 20px rgba(255,20,147,0.15)",
        "compat_bg": "linear-gradient(135deg, rgba(255,182,193,0.9), rgba(255,105,180,0.9))",
        "accent": "#ff1493",
        "accent_alt": "#ffb6c1",
    },
}


# ------------------------------------------------------------------------------
# THEME ENGINE
# ------------------------------------------------------------------------------

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
    background: radial-gradient(circle at 20% 50%,
        var(--page-grad-start) 0%,
        var(--page-grad-mid) 50%,
        var(--page-grad-end) 100%);
    color: var(--text-primary);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

.page-shell {{
    max-width: 800px;
    margin: 0 auto;
    padding: 20px 16px 40px;
}}

.toolbar-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    padding: 24px 32px;
    border-radius: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
    text-align: center;
}}

.match-card {{
    background: var(--card-bg);
    border-radius: 32px;
    padding: 0;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    position: relative;
    margin: 20px auto 0;
    max-width: 500px;
    overflow: hidden;
}}

.match-header {{
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    padding: 16px 20px;
    background: rgba(0,0,0,0.5);
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    z-index: 2;
}}

.image-wrap {{
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;
}}

.profile-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 0;
}}

.score-badge {{
    position: absolute;
    top: 20px;
    right: 20px;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow);
    background: var(--card_bg);
    border: 2px solid var(--accent);
}}

.score-inner {{
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent);
    color: white;
    font-weight: bold;
    font-size: 18px;
}}

.name-age-overlay {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    padding: 60px 20px 20px;
    color: white;
    text-align: center;
}}

.name-age {{
    font-size: 32px;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}}

.tagline {{
    font-size: 18px;
    margin: 4px 0 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    opacity: 0.9;
}}

.summary-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    padding: 20px 24px;
    border-radius: 20px;
    max-width: 500px;
    margin: 20px auto;
    box-shadow: var(--shadow);
}}

.meta-pill {{
    display: inline-flex;
    padding: 8px 16px;
    border-radius: 20px;
    background: var(--chip-bg);
    border: 1px solid var(--border);
    font-size: 0.9rem;
    color: var(--chip-text);
    font-weight: 500;
}}

.chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}}

.chip {{
    padding: 6px 12px;
    background: var(--chip-bg);
    border-radius: 16px;
    border: 1px solid var(--border);
    color: var(--chip-text);
    font-size: 0.85rem;
}}

.profile-details {{
    display: flex;
    gap: 24px;
    margin-top: 24px;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}}

.profile-details > div {{
    flex: 1;
}}

@media(max-width:768px){{
    .profile-img {{ }}
    .name-age {{ font-size: 28px; }}

    .page-shell {{ padding: 20px 16px 40px; }}
    .toolbar-card {{ padding: 20px 24px; font-size: 24px; }}
    .match-card {{ margin: 20px auto; }}
    .summary-card {{ padding: 20px 24px; }}
    .meta-pill {{ font-size: 0.9rem; padding: 8px 12px; }}
    .chip {{ padding: 8px 12px; }}
    .name-age {{ margin-top: 0; }}
    .tagline {{ margin-bottom: 0; }}

    .stButton button {{ min-height: 48px; font-size: 16px; }}
    .stSelectbox {{ font-size: 16px; }}
}}

@media(max-width:480px){{
    .image-wrap {{ height: 350px; }}
    .name-age {{ font-size: 24px; }}
    .tagline {{ font-size: 16px; }}
    .toolbar-card {{ font-size: 20px; padding: 16px 20px; }}
    .match-card {{ }}
    .summary-card {{ padding: 16px 20px; }}
    .meta-pill {{ font-size: 0.8rem; padding: 6px 10px; }}
    .chip {{ padding: 6px 10px; font-size: 0.8rem; }}
}}

@media(max-width:768px){{
    .profile-details {{
        flex-direction: column;
        gap: 16px;
    }}
}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("profiles.csv")
    if "perefence" in df.columns:
        df = df.rename(columns={"perefence": "preference"})
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_image(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.thumbnail((402, 612), Image.Resampling.LANCZOS)
        return img
    except Exception:
        fallback = requests.get("https://via.placeholder.com/500")
        img = Image.open(io.BytesIO(fallback.content))
        img.thumbnail((402, 612), Image.Resampling.LANCZOS)
        return img


def parse_interests(raw: str) -> Set[str]:
    return {x.strip().lower() for x in str(raw).split(",") if x.strip()}


def traits_from_row(row: pd.Series) -> str:
    interests = row.get("interests", "")
    return (
        f"{row['name']} is a {row['age']}-year-old {row['gender']} working as a "
        f"{row['occupation']}. They are {row['height']} tall, studied {row['education']}, "
        f"and are a {row['starSign']}. Interests include {interests}. "
        f"Bio: {row['bio']} Looking for: {row['lookingFor']}"
    )


def compatibility_summary(persona: pd.Series, target: pd.Series) -> Dict[str, str]:
    a = parse_interests(persona.get("interests", ""))
    b = parse_interests(target.get("interests", ""))

    overlap = sorted(list(a & b))
    novelty = sorted(list(b - a))

    age_gap = abs(int(persona["age"]) - int(target["age"]))
    score = 0.38 + 0.12 * len(overlap)
    if persona.get("starSign") == target.get("starSign"):
        score += 0.05
    score -= min(0.18, age_gap * 0.01)
    score = max(0.05, min(0.98, score))

    shared = ", ".join(overlap) if overlap else "different passions"
    new_energy = ", ".join(novelty[:2]) if novelty else "fresh experiences"

    return {
        "score": round(score, 2),
        "reason": f"You click because you both enjoy {shared}, and complement each other with {new_energy}.",
        "dateIdea": "Coffee then live music stroll",
    }


def get_api_key() -> Optional[str]:
    env_key = os.getenv("OPENROUTER_API_KEY")
    try:
        secret = st.secrets.get("OPENROUTER_API_KEY", None)
    except:
        secret = None
    return secret or env_key


def parse_summary_text(text: str) -> Dict[str, str]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e+1])
            except:
                return {}
        return {}


def call_llm(prompt: str, model: str) -> dict:
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key"}
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return {"content": result["choices"][0]["message"]["content"], "usage": result.get("usage", {})}
        else:
            return {"error": f"API error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def score_to_color(score: float) -> str:
    hue = int(120 * max(0, min(1, score)))
    return f"hsl({hue}, 80%, 50%)"


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ensure_state(df: pd.DataFrame):
    names = df["name"].tolist()
    if "impersonated" not in st.session_state:
        st.session_state.impersonated = names[0]
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = 0
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = MODEL_OPTIONS[0]
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"


# ------------------------------------------------------------------------------
# SETTINGS UI
# ------------------------------------------------------------------------------

def render_settings(df: pd.DataFrame):
    names = df["name"].tolist()
    with st.popover("Settings ⚙️"):
        st.markdown("**Who are you impersonating?**")
        selected = st.selectbox(
            "Persona",
            names,
            index=names.index(st.session_state.impersonated),
            key="impersonate_select"
        )
        st.markdown("**Model**")
        model = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.model_choice),
            key="model_select"
        )
        st.markdown("**Theme**")
        theme = st.selectbox(
            "Theme",
            ["dark", "light"],
            index=["dark", "light"].index(st.session_state.theme_mode),
            key="theme_select"
        )
        if st.button("Apply settings", type="primary"):
            st.session_state.impersonated = selected
            st.session_state.model_choice = model
            st.session_state.theme_mode = theme
            st.session_state.active_idx = 0
            st.rerun()


# ------------------------------------------------------------------------------
# PROFILE CARD RENDERING
# ------------------------------------------------------------------------------

def render_profile_card(persona: pd.Series, profile: pd.Series):

    picture = fetch_image(profile["profilePicture"])

    # Generate compatibility summary using LLM
    persona_traits = traits_from_row(persona)
    target_traits = traits_from_row(profile)
    prompt = f"You are {persona['name']}. Analyze your compatibility with this person based on the profiles below. Provide a JSON response with 'score' (float 0-1), 'reason' (string written in second person using 'You' to refer to yourself), 'dateIdea' (string).\n\nYour profile: {persona_traits}\n\nTarget profile: {target_traits}"

    result = call_llm(prompt, st.session_state.model_choice)
    if "error" in result:
        summary = compatibility_summary(persona, profile)
        st.warning("Using local summary due to API error.")
        show_costs = False
    else:
        summary_text = result["content"]
        summary = parse_summary_text(summary_text)
        if not summary:
            summary = compatibility_summary(persona, profile)
            st.warning("Failed to parse LLM response, using local summary.")
            show_costs = False
        else:
            show_costs = True
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            model = st.session_state.model_choice
            costs = MODEL_COSTS.get(model, MODEL_COSTS["openai/gpt-4o-mini"])
            input_cost = (input_tokens / 1e6) * costs["input_per_million"]
            output_cost = (output_tokens / 1e6) * costs["output_per_million"]
            total_cost = input_cost + output_cost
            cost_1000 = total_cost * 1000

    score = summary["score"]
    pct = int(score * 100)
    color = score_to_color(score)

    img64 = image_to_base64(picture)

    st.markdown(
        f"""
        <div class="image-wrap">
            <div class="match-header">
                <span class="meta-pill">🎯 Match</span>
                <span class="meta-pill">Impersonating {persona["name"]}</span>
            </div>
            <img src="data:image/png;base64,{img64}" class="profile-img"/>
            <div class="score-badge"
                 style="background: var(--card-bg);">
                <div class="score-inner">{pct}%</div>
            </div>
            <div class="name-age-overlay">
                <div class="name-age">{profile["name"]}, {profile["age"]}</div>
                <div class="tagline">{profile["occupation"]} • {profile["starSign"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
                <span class="meta-pill">Compatibility</span>
                <span style="color:{color}; font-weight:800; font-size: 18px;">{pct}%</span>
            </div>
            <p style="margin: 0 0 12px 0; line-height: 1.5;">{summary["reason"]}</p>
            <span style="color:var(--text-muted); font-style: italic;">💡 {summary["dateIdea"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_costs:
        st.markdown("**LLM Cost Breakdown**")
        st.markdown(f"""
        <pre style="background-color: #000000; color: #00FF00; font-family: 'Courier New', monospace; padding: 15px; border-radius: 8px; border: 1px solid #333; overflow-x: auto;">
        Input tokens: {input_tokens}
        Output tokens: {output_tokens}
        Input cost: ${input_cost:.6f}
        Output cost: ${output_cost:.6f}
        Total cost: ${total_cost:.6f}
        Model: {model}
        Cost for 1000 calls: ${cost_1000:.4f}
        </pre>
        """, unsafe_allow_html=True)

    st.markdown("#### Profile Details")
    st.markdown('<div class="profile-details">', unsafe_allow_html=True)

    st.markdown('<div>', unsafe_allow_html=True)
    st.write(f"**Gender:** {profile['gender']}")
    st.write(f"**Height:** {profile['height']}")
    st.write(f"**Education:** {profile['education']}")
    st.write(f"**Looking For:** {profile['lookingFor']}")
    st.write("**Interests:**")

    chips = "".join([f"<span class='chip'>{x.strip()}</span>" for x in str(profile["interests"]).split(",")])
    st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div>', unsafe_allow_html=True)
    st.write("**Bio:**")
    st.write(profile["bio"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------

def main():
    df = load_data()
    ensure_state(df)

    theme = get_theme(st.session_state.theme_mode)
    apply_theme_styles(theme)

    st.markdown('<div class="page-shell">', unsafe_allow_html=True)

    # HEADER + SETTINGS
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
            <div class="toolbar-card">
                <div style="font-size:28px; font-weight:800; text-align:center;">
                    💕 Chemistry Cards
                </div>
                <div style="text-align:center; color:var(--text-muted); font-size: 16px;">
                    Find your perfect match with AI-powered compatibility analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        render_settings(df)

    # PROFILE ROTATION
    rotation = df[df["name"] != st.session_state.impersonated].reset_index(drop=True)
    if rotation.empty:
        st.warning("No profiles to show.")
        return

    st.session_state.active_idx %= len(rotation)
    profile = rotation.iloc[st.session_state.active_idx]
    persona = df[df["name"] == st.session_state.impersonated].iloc[0]

    # NAVIGATION
    nav1, nav2, nav3 = st.columns([1.2, 1.6, 1.2])
    with nav1:
        if st.button("❌ Nope", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx - 1) % len(rotation)
            st.rerun()
    with nav2:
        st.markdown(
            f"""
            <div style="text-align:center; color:var(--text-muted); font-size:0.9rem;">
                Viewing {st.session_state.active_idx+1} of {len(rotation)}<br/>
                Impersonating <strong>{st.session_state.impersonated}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("❤️ Like", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx + 1) % len(rotation)
            st.rerun()

    # RENDER PROFILE
    render_profile_card(persona, profile)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
