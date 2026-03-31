import anthropic
import streamlit as st

# system prompt

SYSTEM_PROMPT = """You are an AI assistant embedded in a fundraising strategy tool built for a national charity.
The tool analyzes IRS ZIP code charitable giving data to identify the best gala invite targets.

You have access to a summary of the current filtered dataset, including top ZIP codes by generosity,
cluster segments, and donor momentum leaders. Use this context to answer questions clearly and concisely.

Always frame answers in terms of fundraising strategy — which ZIP codes to prioritize,
why a cluster is valuable, or what a trend means for the charity's outreach.

Do not speculate beyond the data provided. If a ZIP code is not in the context, say so."""


# context builder - compact text summary of current app state to pass as context

def build_context(top_n_df, seg_summary, rising_df, selected_states, selected_years_range):
    """serializes filtered app state into plain text for claude"""
    state_label = ", ".join(selected_states) if selected_states else "All States"
    year_label  = f"{min(selected_years_range)}–{max(selected_years_range)}"

    lines = [
        f"Active filters: {state_label} | Years: {year_label}",
        "",
        "=== TOP ZIP CODES (by current ranking metric) ===",
    ]

    for _, row in top_n_df.head(20).iterrows():
        lines.append(
            f"  ZIP {row['zipcode']} ({row['STATE']}) | "
            f"GI: {row['generosity_index']:.2%} | "
            f"PR: {row['participation_rate']:.2%} | "
            f"Donors: {int(row['N19700']):,} | "
            f"AGI: ${row['A00100']:,.0f}k"
        )

    lines += ["", "=== CLUSTER SEGMENTS (K-Means) ==="]
    for _, row in seg_summary.iterrows():
        lines.append(
            f"  {row['Segment']} | ZIP count: {int(row['ZIP_Codes']):,} | "
            f"Avg GI: {row['Avg_GI']:.2f}% | "
            f"Avg PR: {row['Avg_PR']:.2f}%"
        )

    if not rising_df.empty:
        lines += ["", "=== DONOR MOMENTUM LEADERS (rising GI trend) ==="]
        for _, row in rising_df.head(10).iterrows():
            lines.append(
                f"  ZIP {row['zipcode']} ({row['STATE']}) | "
                f"Momentum: {row['momentum_score']:.5f} | "
                f"Current GI: {row['generosity_index']:.2%}"
            )

    return "\n".join(lines)


# natural language query - sends question + dataset context to claude

def ask_assistant(user_question, context_text, history=None):
    """sends user question + dataset context to claude, returns response - supports multi-turn history"""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # include fresh dataset context each turn
    user_message = f"Here is the current dataset context:\n\n{context_text}\n\nQuestion: {user_question}"

    messages = []
    if history:
        # cap at last 10 messages to keep token usage manageable
        for m in history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    return response.content[0].text


# zip report generator - short strategic brief for single target zip

def generate_zip_report(zip_row, context_text):
    """generates 3-5 sentence strategic brief for specific zip code"""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    zip_detail = (
        f"ZIP Code: {zip_row['zipcode']} | State: {zip_row['STATE']}\n"
        f"Generosity Index: {zip_row['generosity_index']:.2%}\n"
        f"Participation Rate: {zip_row['participation_rate']:.2%}\n"
        f"Itemizing Donors (N19700): {int(zip_row['N19700']):,}\n"
        f"Total AGI (A00100): ${zip_row['A00100']:,.0f}k\n"
        f"Total Filers (N1): {int(zip_row['N1']):,}"
    )

    prompt = (
        f"Generate a 3-5 sentence gala target brief for this ZIP code. "
        f"Explain why it is or isn't a strong candidate based on the metrics. "
        f"Reference how it compares to the broader dataset context below.\n\n"
        f"ZIP Details:\n{zip_detail}\n\n"
        f"Dataset Context:\n{context_text}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
