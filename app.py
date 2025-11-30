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


def stream_openrouter(prompt: str, api_key: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Chemistry Cards",
    }
    payload = {
        "model": "openai/gpt-4o-mini",
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
        st.caption("Selected persona is removed from the rotation of shown profiles.")
        if st.button("Use this persona", type="primary"):
            st.session_state.impersonated = selected
            st.session_state.active_idx = 0
            st.rerun()


def render_profile_card(persona: pd.Series, profile: pd.Series):
    st.markdown(
        """
        <style>
        .stApp {background: radial-gradient(circle at 10% 20%, #111827 0, #0b1120 50%, #0a0f1d 100%); color: #e5e7eb;}
        .match-card {background: #0f172a; border-radius: 26px; padding: 28px; box-shadow: 0 20px 60px rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.05);}
        .name-age {font-size: 32px; font-weight: 700; text-align:center; margin-top: 16px; color: #f8fafc;}
        .tagline {text-align:center; color: #9ca3af; margin-bottom: 18px;}
        .compat-box {background: linear-gradient(135deg, #111827, #0b162b); border: 1px solid #1f2937; border-radius: 18px; padding: 18px 20px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);}
        .chips {display: flex; flex-wrap: wrap; gap: 8px;}
        .chip {background: #111827; color: #e5e7eb; padding: 6px 10px; border-radius: 999px; border:1px solid #1f2937;}
        .stMarkdown, .stText, .stCaption, .stExpander {color: #e5e7eb !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="match-card">', unsafe_allow_html=True)
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
            '  "ideaSummary": STRING\n'
            "}\n\n"
            f"Person A: {traits_from_row(persona)}\n"
            f"Person B: {traits_from_row(profile)}\n\n"
            'Write with the pronoun "you"'
        )

        compat_container = st.container()
        compat_container.markdown("##### Compatibility Summary")
        compat_container.markdown('<div class="compat-box">', unsafe_allow_html=True)

        api_key = get_api_key()
        usage_info: Dict[str, Optional[int]] = {"model": "openai/gpt-4o-mini", "prompt_tokens": None, "completion_tokens": None}
        used_ai = False

        if api_key:
            try:
                stream_gen, usage = stream_openrouter(prompt, api_key)
                compat_container.write_stream(stream_gen)
                usage_info.update(usage)
                used_ai = True
            except Exception:
                st.warning("OpenRouter request failed; showing local compatibility estimate.")

        if not used_ai:
            compat_container.write_stream(stream_json(compatibility_summary(persona, profile)))

        compat_container.markdown("</div>", unsafe_allow_html=True)
        if usage_info.get("prompt_tokens") is not None or usage_info.get("completion_tokens") is not None:
            prompt_tks = usage_info.get("prompt_tokens") or 0
            completion_tks = usage_info.get("completion_tokens") or 0
            in_rate = 0.000150  # dollars per 1k input tokens for gpt-4o-mini
            out_rate = 0.000600  # dollars per 1k output tokens for gpt-4o-mini
            in_cost = (prompt_tks / 1000) * in_rate
            out_cost = (completion_tks / 1000) * out_rate
            total_cost = in_cost + out_cost
            compat_container.markdown(
                f"**Cost breakdown** — Model: `{usage_info.get('model','openai/gpt-4o-mini')}`  \n"
                f"- Input tokens: {prompt_tks} (${in_cost:.5f})  \n"
                f"- Output tokens: {completion_tks} (${out_cost:.5f})  \n"
                f"- Estimated total: **${total_cost:.5f}**"
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
    st.header("Chemistry Cards")
    df = load_data()
    ensure_state(df)

    top_cols = st.columns([5, 1.5])
    with top_cols[1]:
        render_settings(df)

    rotation = df[df["name"] != st.session_state.impersonated].reset_index(drop=True)
    if rotation.empty:
        st.warning("All profiles are hidden because you are impersonating the only available persona.")
        return

    st.session_state.active_idx = st.session_state.active_idx % len(rotation)
    profile = rotation.iloc[st.session_state.active_idx]
    persona = df[df["name"] == st.session_state.impersonated].iloc[0]

    nav_cols = st.columns([1, 1, 4])
    with nav_cols[0]:
        if st.button("⬅ Previous", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx - 1) % len(rotation)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.active_idx = (st.session_state.active_idx + 1) % len(rotation)
            st.rerun()
    with nav_cols[2]:
        st.write(
            f"Viewing {st.session_state.active_idx + 1} of {len(rotation)} "
            f"(you are impersonating {st.session_state.impersonated})"
        )

    render_profile_card(persona, profile)


if __name__ == "__main__":
    main()
