import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import html as _html
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ai_assistant import ask_assistant, build_context, generate_zip_report

# clear chat if triggered via URL param
if st.query_params.get('_cc') == '1':
    st.session_state.chat_messages = []
    st.query_params.clear()

# process pending chat message submitted from right-panel input
_pending_msg = st.query_params.get('_msg', '')
if _pending_msg:
    st.query_params.clear()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    st.session_state.chat_messages.append({"role": "user", "content": _pending_msg})
    _ctx = st.session_state.get('_ai_context', 'No dataset context available yet.')
    _history = st.session_state.chat_messages[:-1]  # history before current message
    _response = ask_assistant(_pending_msg, _ctx, history=_history)
    st.session_state.chat_messages.append({"role": "assistant", "content": _response})

# init chat session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

def _chat_messages_html(messages):
    if not messages:
        return (
            '<div class="chat-empty">'
            '<div class="ei">&#129302;</div>'
            '<p>Ask me anything about the fundraising data.</p>'
            '</div>'
        )
    parts = []
    for m in messages:
        content = _html.escape(m["content"]).replace('\n', '<br>')
        if m["role"] == "user":
            parts.append(
                '<div class="msg-row row-user">'
                '<div class="av av-u">YOU</div>'
                '<div class="bbl bbl-u">' + content + '</div>'
                '</div>'
            )
        else:
            parts.append(
                '<div class="msg-row row-bot">'
                '<div class="av av-b">AI</div>'
                '<div class="bbl bbl-b">' + content + '</div>'
                '</div>'
            )
    return "\n".join(parts)

# set layout
st.set_page_config(page_title="Generosity Intelligence", layout="wide")

import streamlit.components.v1 as _components

# right panel — 3-part injection: CSS via st.markdown, HTML via st.markdown, JS via components
_DEF_W = 260

# ── 1. CSS (includes panel layout + chat input docking) ───────────────────────
st.markdown("""
<style>
    [data-testid="stSidebarResizeHandle"] { display:none !important; pointer-events:none !important; }
    .block-container { transition: padding-right 0.25s ease; padding-right: 18rem !important; }

    /* panel shell */
    .right-panel {
        position: fixed; top: 0; right: 0;
        width: 260px; height: 100vh;
        background: rgb(38,39,48);
        border-left: 1px solid rgba(250,250,250,0.1);
        z-index: 100;
        display: flex; flex-direction: column;
        overflow: hidden;
        transition: right 0.3s ease, width 0.2s ease;
        box-sizing: border-box;
    }
    .right-panel.collapsed { right: -110vw; }

    /* drag handle */
    .rp-resize-handle {
        position: absolute; top: 0; left: 0;
        width: 5px; height: 100%;
        cursor: col-resize; z-index: 102;
        background: transparent; transition: background 0.15s;
    }
    .rp-resize-handle:hover, .rp-resize-handle.dragging { background: rgba(56,189,248,0.45); }

    /* side tab toggle */
    .rp-toggle {
        position: absolute; top: 50%; left: -1.4rem;
        transform: translateY(-50%);
        width: 1.4rem; height: 2.8rem;
        background: rgb(38,39,48);
        border: 1px solid rgba(250,250,250,0.1); border-right: none;
        border-radius: 6px 0 0 6px;
        color: #fafafa; font-size: 1rem;
        cursor: pointer; user-select: none;
        display: flex; align-items: center; justify-content: center;
        z-index: 101;
    }
    .rp-toggle:hover { background: rgb(55,57,70); }

    /* header */
    .rp-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.8rem 0.85rem 0.7rem;
        border-bottom: 1px solid rgba(250,250,250,0.1);
        flex-shrink: 0;
    }
    .rp-title { color: #fafafa; font-size: 0.9rem; font-weight: 600; margin: 0; }
    .rp-actions { display: flex; gap: 0.3rem; }
    .rp-btn {
        background: none; border: 1px solid rgba(250,250,250,0.13); border-radius: 4px;
        color: rgba(250,250,250,0.55); width: 1.5rem; height: 1.5rem; font-size: 0.72rem;
        cursor: pointer; display: flex; align-items: center; justify-content: center;
        transition: background 0.15s, color 0.15s;
    }
    .rp-btn:hover { background: rgba(250,250,250,0.09); color: #fafafa; }
    .rp-btn.active { color: #38bdf8; border-color: rgba(56,189,248,0.4); }

    /* messages area */
    .chat-msgs {
        flex: 1; overflow-y: auto;
        padding: 0.75rem 0.8rem 0.5rem;
        display: flex; flex-direction: column; gap: 0.6rem;
        scrollbar-width: thin; scrollbar-color: rgba(250,250,250,0.12) transparent;
    }
    .chat-msgs::-webkit-scrollbar { width: 3px; }
    .chat-msgs::-webkit-scrollbar-thumb { background: rgba(250,250,250,0.12); border-radius: 2px; }

    /* empty state */
    .chat-empty {
        display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
        margin-top: 2.5rem; color: rgba(250,250,250,0.28);
        font-size: 0.78rem; text-align: center; line-height: 1.5;
    }
    .chat-empty .ei { font-size: 2rem; }

    /* message rows */
    .msg-row { display: flex; gap: 0.4rem; align-items: flex-end; }
    .msg-row.row-user { flex-direction: row-reverse; }

    /* avatars */
    .av {
        width: 1.5rem; height: 1.5rem; border-radius: 50%;
        font-size: 0.55rem; font-weight: 800; letter-spacing: 0.02em;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .av-u { background: #1e3a8a; color: #bae6fd; }
    .av-b { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.22); }

    /* chat bubbles */
    .bbl { padding: 0.42rem 0.65rem; font-size: 0.79rem; line-height: 1.5; word-break: break-word; max-width: calc(100% - 2.2rem); }
    .bbl-u { background: #1e3a8a; color: #dbeafe; border-radius: 10px 10px 2px 10px; }
    .bbl-b { background: rgba(255,255,255,0.06); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.09); border-radius: 10px 10px 10px 2px; }

    /* chat input bar — docked at panel bottom */
    .chat-input-bar {
        flex-shrink: 0;
        padding: 0.5rem 0.75rem 0.65rem;
        border-top: 1px solid rgba(250,250,250,0.1);
        display: flex; flex-direction: column; gap: 0.4rem;
    }
    .chat-input-wrap { display: flex; gap: 0.4rem; align-items: flex-end; }
    .chat-ta {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 6px;
        color: #e2e8f0; font-size: 0.79rem;
        padding: 0.42rem 0.6rem;
        resize: none; min-height: 2rem;
        overflow: hidden; outline: none;
        font-family: inherit; line-height: 1.4;
        box-sizing: border-box;
    }
    .chat-ta::placeholder { color: rgba(255,255,255,0.28); }
    .chat-ta:focus { border-color: rgba(56,189,248,0.45); }
    .chat-send-btn {
        background: #1e3a8a; border: none; border-radius: 6px;
        color: #bae6fd; width: 2rem; height: 2rem;
        cursor: pointer; font-size: 0.85rem;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; transition: background 0.15s;
    }
    .chat-send-btn:hover { background: #1d4ed8; }
    .chat-clear-btn {
        background: none; border: 1px solid rgba(250,250,250,0.13); border-radius: 6px;
        color: rgba(250,250,250,0.45); width: 2rem; height: 2rem;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 0.78rem; cursor: pointer;
        text-decoration: none; transition: background 0.15s, color 0.15s;
    }
    .chat-clear-btn:hover { background: rgba(250,250,250,0.08); color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── 2. Panel HTML (rebuilt each rerun with fresh messages) ────────────────────
_msgs_html = _chat_messages_html(st.session_state.chat_messages)
st.markdown(
    '<div class="right-panel" id="rightPanel">'
    '<div class="rp-resize-handle" id="rpResizeHandle"></div>'
    '<button class="rp-toggle" id="rpToggle" title="Toggle">&#8250;</button>'
    '<div class="rp-header">'
    '<span class="rp-title">&#129302; AI Assistant</span>'
    '<div class="rp-actions">'
    '<button class="rp-btn" id="btnExpand" title="Expand">&#x26F6;</button>'
    '</div></div>'
    '<div class="chat-msgs" id="chatMsgs">' + _msgs_html + '</div>'
    '<div class="chat-input-bar">'
    '<div class="chat-input-wrap">'
    '<a class="chat-clear-btn" href="?_cc=1" title="Clear chat">&#10005;</a>'
    '<form method="GET" action="" style="display:flex;flex:1;gap:0.4rem;margin:0;padding:0;">'
    '<textarea name="_msg" id="chatInput" class="chat-ta" rows="1" placeholder="Ask about the data\u2026"'
    ' oninput="this.style.height=\'auto\';this.style.height=this.scrollHeight+\'px\';"'
    ' onkeydown="if(event.key===\'Enter\'&&!event.shiftKey){event.preventDefault();if(this.value.trim())this.form.submit();}"'
    '></textarea>'
    '<button type="submit" id="chatSend" class="chat-send-btn"'
    ' onclick="if(!document.getElementById(\'chatInput\').value.trim()){event.preventDefault();}"'
    '>&#10148;</button>'
    '</form>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# ── 3. Behavior JS via components (executes scripts reliably) ─────────────────
_components.html(
    '<script>'
    'var p = window.parent.document;'
    'var panel = p.getElementById("rightPanel");'
    'var toggle = p.getElementById("rpToggle");'
    'var handle = p.getElementById("rpResizeHandle");'
    'var btnExp = p.getElementById("btnExpand");'
    'var MIN_W = 180, DEF_W = ' + str(_DEF_W) + ';'
    'var MAX_W = Math.round(window.parent.innerWidth * 0.85);'
    'var EXP_W = Math.max(460, Math.round(window.parent.innerWidth * 0.42));'
    'var expanded = false;'
    'function getMain(){ return p.querySelector(".block-container"); }'
    'function syncWidth(px){'
    '  if(!panel) return;'
    '  panel.style.width = px + "px";'
    '  var m = getMain(); if(m && !panel.classList.contains("collapsed")) m.style.paddingRight = (px+32)+"px";'
    '}'
    'function toggleExpand(){'
    '  expanded = !expanded;'
    '  panel.style.transition = "right 0.3s ease, width 0.2s ease";'
    '  syncWidth(expanded ? EXP_W : DEF_W);'
    '  if(btnExp) btnExp.classList.toggle("active", expanded);'
    '}'
    'function togglePanel(){'
    '  var c = panel.classList.toggle("collapsed");'
    '  var m = getMain(); if(m) m.style.paddingRight = c ? "2rem" : (panel.offsetWidth+32)+"px";'
    '  if(toggle) toggle.innerHTML = c ? "&#8249;" : "&#8250;";'
    '}'
    'if(toggle) toggle.onclick = togglePanel;'
    'if(btnExp) btnExp.onclick = toggleExpand;'
    'var sx, sw;'
    'if(handle) handle.addEventListener("mousedown", function(e){'
    '  if(panel.classList.contains("collapsed")) return;'
    '  e.preventDefault(); sx = e.clientX; sw = panel.offsetWidth;'
    '  handle.classList.add("dragging"); panel.style.transition = "none";'
    '  function mv(e){ syncWidth(Math.min(MAX_W, Math.max(MIN_W, sw + sx - e.clientX))); }'
    '  function up(){ handle.classList.remove("dragging"); panel.style.transition="right 0.3s ease, width 0.2s ease"; window.parent.removeEventListener("mousemove",mv); window.parent.removeEventListener("mouseup",up); }'
    '  window.parent.addEventListener("mousemove",mv);'
    '  window.parent.addEventListener("mouseup",up);'
    '});'
    'var msgs = p.getElementById("chatMsgs");'
    'if(msgs) msgs.scrollTop = msgs.scrollHeight;'
    '</script>',
    height=0
)

# load data
@st.cache_data
def load_data(filepath='data/zpallagi_cleaned.csv'):
    df = pd.read_csv(filepath, dtype={'zipcode': str})
    df['year'] = df['year'].astype(int)
    return df

# setup titles
st.title("Philanthropy Advisor Project")
st.markdown("An analysis of charitable giving patterns across US ZIP codes. Metrics are based on IRS data: **Generosity Index** (Total Charitable Contributions / Total AGI) and **Participation Rate** (Returns with Charitable Contributions / Total Returns).")

# data disclaimer
with st.expander("Data Limitations & Key Tax Insights - Read Before Interpreting Results"):
    st.markdown(
        "**1. The Itemization Floor:** This analysis relies strictly on IRS Schedule A charitable deductions. "
        "Because only taxpayers who itemize their deductions report charitable contributions to the IRS, "
        "standard deduction filers are not captured. This means results reflect generosity specifically "
        "among itemizing filers and may underrepresent middle-income communities.\n\n"
        "**2. The TCJA Data Gap (2017-2018):** The 2017 Tax Cuts and Jobs Act (TCJA) roughly tripled the "
        "standard deduction, dropping the national share of itemizing filers from ~30% to ~10%. "
        "**Data before and after 2017 is not directly comparable.** A sharp decline in giving metrics "
        "crossing that threshold essentially reflects this reporting floor change, not a literal collapse in charitable behavior.\n\n"
        "**3. Outlier Exclusion:** ZIP codes where Generosity Index exceeds 100% (A19700 / A00100 > 1) are excluded. "
        "These extreme values typically result from large non-cash donations or donor-advised fund activity concentrated in very small ZIP codes "
        "and would produce misleading rankings.\n\n"
        "**4. Nominal Dollar Values:** A00100 (Total AGI) and A19700 (Total Charitable Contributions) are reported in nominal thousands of dollars "
        "and are not inflation-adjusted. Raw dollar figures are not directly comparable across years. "
        "The Generosity Index ratio (A19700 / A00100) partially mitigates this since both values inflate together, "
        "but absolute dollar charts and tooltips should be interpreted as nominal."
    )

# try loading data
try:
    full_df = load_data()
except FileNotFoundError:
    st.error("Cleaned data not found. Please run data_processing.py first.")
    st.stop()

# global filters section
st.sidebar.header("Global Filters")

# setup themes
alt.theme.enable('dark')
alt.data_transformers.enable('vegafusion')
THEME_PALETTE = alt.Scale(range=['#38bdf8', '#1e3a8a'])
chart_bg = '#1a1d24'
axis_color = '#fafafa'

# year filter - above state so it scopes available states correctly
all_years = sorted(full_df['year'].unique())
year_options = ["All Available Years"] + list(all_years)

st.sidebar.markdown("**Select Year:**")
st.sidebar.markdown("<p style='font-size: 0.8rem; color: gray; margin-top: -10px; margin-bottom: 5px;'>Includes range bounds and datapoint display</p>", unsafe_allow_html=True)

def handle_year_selection():
    opts = st.session_state['year_filter']
    if "All Available Years" in opts and len(opts) > 1:
        st.session_state['year_filter'] = ["All Available Years"]

if 'year_filter' not in st.session_state:
    st.session_state['year_filter'] = [max(all_years)]

selected_option = st.sidebar.multiselect(
    "Select Year:",
    options=year_options,
    key='year_filter',
    on_change=handle_year_selection,
    label_visibility="collapsed"
)

if not selected_option or "All Available Years" in selected_option:
    selected_years = all_years
else:
    selected_years = [y for y in selected_option if y != "All Available Years"]

# calc full range between min and max selected years
min_year = min(selected_years)
max_year = max(selected_years)
selected_years_range = [y for y in all_years if min_year <= y <= max_year]
# last item the user picked in the multiselect — used for single-year snapshot charts
display_year = selected_years[-1]

df = full_df[full_df['year'].isin(selected_years_range)]

# setup state filter
all_states = sorted(df['STATE'].unique())
selected_states = st.sidebar.multiselect("Select State(s):", options=all_states, default=[])

# apply filter
if selected_states:
    df = df[df['STATE'].isin(selected_states)]

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# cross-era warning when selected range spans the TCJA boundary
pre_tcja_years = [y for y in selected_years_range if y <= 2016]
post_tcja_years = [y for y in selected_years_range if y >= 2018]
spans_tcja = min_year <= 2017 and max_year >= 2018
if pre_tcja_years and post_tcja_years:
    st.error(
        "**Cross-era comparison active.** Your selected year range includes both pre-TCJA years "
        f"({', '.join(str(y) for y in pre_tcja_years)}) and post-TCJA years "
        f"({', '.join(str(y) for y in post_tcja_years)}). "
        "These represent different filer populations (~30% itemizing pre-2017 vs. ~10% post-2017). "
        "Metrics are not directly comparable across this boundary. Aggregate values and state rankings "
        "will reflect the mix of both regimes. Use a single-regime year range for valid trend comparisons."
    )


# calc averages
avg_gen = df['generosity_index'].mean()
med_gen = df['generosity_index'].median()
avg_part = df['participation_rate'].mean()
total_zips = len(df)

# display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average Generosity Index", f"{avg_gen:.2%}")
col2.metric("Average Participation", f"{avg_part:.2%}")
col3.metric("Total ZIP Codes Analyzed", f"{total_zips:,}")

with st.expander("How are these metrics calculated?"):
    st.markdown(
        "**Average Generosity Index** = mean(A19700 / A00100) across all ZIP codes in the selected filter\n\n"
        "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
        "**Average Participation** = mean(N19700 / N1) across all ZIP codes in the selected filter\n\n"
        "- N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "- N1 = Total Returns - total number of tax returns filed"
    )

# divider
st.divider()
st.subheader("Top N ZIP Codes")
if len(selected_years_range) > 1:
    st.caption(f"Showing data for **{display_year}** (last selected year). Multi-year data is not layered here to avoid duplicate ZIP entries.")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    selected_metric_label = st.segmented_control(
        "Rank Top N ZIP Codes By:",
        options=["Generosity", "Participation"],
        default="Generosity"
    )
    selected_metric_label = selected_metric_label or "Generosity"
with col_ctrl2:
    top_n = st.slider("Top N ZIP Codes to Display:", min_value=5, max_value=100, value=20, step=5)
metric_map = {"Generosity": "generosity_index", "Participation": "participation_rate"}
selected_metric = metric_map[selected_metric_label]

# expander info
with st.expander("How is this calculated?"):
    st.markdown(
        f"ZIP codes ranked in descending order by **{'Generosity Index' if selected_metric_label == 'Generosity' else 'Participation Rate'}**.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "  - N1 = Total Returns - total number of tax returns filed"
    )

# get top n metrics — use only the most recent year when multiple years are selected
top_n_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
top_n_df = top_n_source.nlargest(top_n, selected_metric)
min_top_n = float(top_n_df[selected_metric].min()) if not top_n_df.empty else 0
max_top_n = float(top_n_df[selected_metric].max()) if not top_n_df.empty else 0

# parse tooltips
def create_tooltip(fields):
    return [alt.Tooltip(f, title=t, format=fmt) if fmt else alt.Tooltip(f, title=t) for f, t, fmt in fields]

# set shared tooltips
tooltip_common = create_tooltip([
    ('zipcode:N', 'ZIP Code', None),
    ('STATE:N', 'State', None),
    ('year:O', 'Year', None),
    ('A00100:Q', 'Total AGI ($ thousands)', ','),
    ('A19700:Q', 'Total Charitable Contributions ($ thousands)', ','),
    ('generosity_index:Q', 'Generosity Index', '.2%'),
    ('participation_rate:Q', 'Participation Rate', '.2%')
])

# render bar chart
chart1 = alt.Chart(top_n_df).mark_bar().encode(
    y=alt.Y('zipcode:N', sort='-x', title='ZIP Code', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
    x=alt.X(f'{selected_metric}:Q', title=f'{selected_metric_label} (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    color=alt.Color(f'{selected_metric}:Q', scale=THEME_PALETTE, legend=None),
    tooltip=tooltip_common
).properties(height=500).configure(background=chart_bg)

# embed chart
st.altair_chart(chart1, width='stretch', theme=None)

# gala target table
st.markdown("##### Gala Target List")
table_df = top_n_df[['zipcode', 'STATE', 'year', 'generosity_index', 'participation_rate', 'N19700', 'A19700', 'N1']].copy()
table_df['generosity_index']   = (table_df['generosity_index']   * 100).round(2)
table_df['participation_rate'] = (table_df['participation_rate'] * 100).round(2)
st.dataframe(
    table_df,
    width='stretch',
    hide_index=True,
    column_config={
        'zipcode':            st.column_config.TextColumn('ZIP Code'),
        'STATE':              st.column_config.TextColumn('State'),
        'year':               st.column_config.NumberColumn('Year', format='%d'),
        'generosity_index':   st.column_config.NumberColumn('Generosity Index % (A19700 / A00100)', format='%.2f%%'),
        'participation_rate': st.column_config.NumberColumn('Participation Rate % (N19700 / N1)',   format='%.2f%%'),
        'N19700':             st.column_config.NumberColumn('Itemizing Donors (N19700)',            format='%d'),
        'A19700':             st.column_config.NumberColumn('Total Giving in $k (A19700)',          format='$%,.0f'),
        'N1':                 st.column_config.NumberColumn('Total Filers (N1)',                    format='%d'),
    }
)

# AI ZIP Brief — wires generate_zip_report into the chat panel
st.markdown("##### AI ZIP Code Brief")
if not top_n_df.empty:
    _col_zip, _col_btn = st.columns([3, 1])
    with _col_zip:
        _brief_zip = st.selectbox(
            "Select ZIP Code for Brief:",
            options=top_n_df['zipcode'].tolist(),
            label_visibility="collapsed",
            key="brief_zip_select"
        )
    with _col_btn:
        st.write("")
        if st.button("Get AI Brief", use_container_width=True, key="gen_brief_btn"):
            _zip_row = top_n_df[top_n_df['zipcode'] == _brief_zip].iloc[0]
            _ctx = st.session_state.get('_ai_context', 'No dataset context available yet.')
            _brief_text = generate_zip_report(_zip_row, _ctx)
            st.session_state.chat_messages.append({"role": "user", "content": f"Give me a gala brief for ZIP {_brief_zip}."})
            st.session_state.chat_messages.append({"role": "assistant", "content": f"Gala Brief — ZIP {_brief_zip} ({_zip_row['STATE']}):\n\n{_brief_text}"})
            st.rerun()

# divider
st.divider()
st.subheader("Participation vs. Generosity")
if len(selected_years_range) > 1:
    st.caption(f"Showing data for **{display_year}** (last selected year).")
with st.expander("How is this calculated?"):
    st.markdown(
        "Scatter plot comparing **Generosity Index** (vertical axis, depth) against **Participation Rate** (horizontal axis, reach) per ZIP code bin. "
        "Bubble size encodes the sum of N1 (Total Returns) within each bin.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "Dashed red lines mark the dataset averages for each axis. White dashed lines trace the user-set target thresholds. Quadrant shading reflects combinations of above/below target reach and depth:\n"
        "- **Top Left (Blue)**: Low Reach, High Depth\n"
        "- **Top Right (Green)**: High Reach, High Depth\n"
        "- **Bottom Left (Red)**: Low Reach, Low Depth\n"
        "- **Bottom Right (Yellow)**: High Reach, Low Depth"
    )

st.markdown("#### Set Target Thresholds")
col_t1, col_t2, col_t3 = st.columns(3)
min_income_k = col_t1.slider("Min Avg Household Income ($k)", min_value=0, max_value=500, value=100, step=10, format="$%dk") * 1000
min_gen    = col_t2.slider("Min Generosity Index (%)", min_value=0.0, max_value=20.0, value=float(avg_gen * 100), step=0.1, format="%.1f%%") / 100
min_part   = col_t3.slider("Min Participation Rate (%)", min_value=0.0, max_value=50.0, value=float(avg_part * 100), step=1.0, format="%.1f%%") / 100

st.markdown("##### Target Statistics")
scatter_source = df[df['year'] == display_year].copy() if len(selected_years_range) > 1 else df.copy()
scatter_source['avg_hh_income'] = (scatter_source['A00100'] * 1000) / scatter_source['N1']
target_df = scatter_source[
    (scatter_source['avg_hh_income'] >= min_income_k) &
    (scatter_source['generosity_index'] >= min_gen) &
    (scatter_source['participation_rate'] >= min_part)
].copy()

target_count = len(target_df)
target_avg_gen = target_df['generosity_index'].mean() if target_count > 0 else 0
target_avg_part = target_df['participation_rate'].mean() if target_count > 0 else 0
give_score = target_avg_gen / target_avg_part if target_avg_part > 0 else 0
target_avg_agi = ((target_df['A00100'].sum() * 1000) / target_df['N1'].sum()) if target_count > 0 and target_df['N1'].sum() > 0 else 0

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Targeted ZIP-Years", f"{target_count:,}")
col_m2.metric("Average Give Score (GI/PR)", f"{give_score:.2f}")
col_m3.metric("Average Target AGI", f"${target_avg_agi:,.0f}")

# set binned tooltips
tooltip_binned = create_tooltip([
    ('mean(participation_rate):Q', 'PR (%)', '.2%'),
    ('mean(generosity_index):Q', 'GI (%)', '.2%'),
    ('sum(N1):Q', 'Total Population', ','),
    ('count()', 'Number of ZIP Codes', None)
])

# define avg lines (static global references)
avg_df = pd.DataFrame({'generosity_index': [avg_gen], 'participation_rate': [avg_part]})
rule_x = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='participation_rate:Q')
rule_y = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(y='generosity_index:Q')

# create new target tracking lines (dynamic user input)
target_lines_df = pd.DataFrame({'generosity_index': [min_gen], 'participation_rate': [min_part]})
rule_x_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(x='participation_rate:Q')
rule_y_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(y='generosity_index:Q')

# render background binned scatter
chart_binned_scatter = alt.Chart(scatter_source).mark_circle(opacity=1.0).encode(
    x=alt.X('participation_rate:Q', bin=alt.Bin(maxbins=30), title='Charitable Participation Rate (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    y=alt.Y('generosity_index:Q', bin=alt.Bin(maxbins=30), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    size=alt.Size('sum(N1):Q', title='Total Population', scale=alt.Scale(range=[10, 1000]), legend=None),
    color=alt.Color('sum(N1):Q', scale=THEME_PALETTE, legend=None),
    tooltip=tooltip_binned
).properties(height=400)

# color quadrants based on user targets
max_p = scatter_source['participation_rate'].max() * 1.05 if not scatter_source.empty else 1
max_g = scatter_source['generosity_index'].max() * 1.05 if not scatter_source.empty else 1

quads = pd.DataFrame([
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#ef4444'},       # Bottom Left (Red)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#eab308'},   # Bottom Right (Yellow)
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#3b82f6'},   # Top Left (Blue)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#10b981'}# Top Right (Green)
])
chart_quads = alt.Chart(quads).mark_rect(opacity=0.1).encode(
    x='participation_rate:Q',
    x2='participation_rate2:Q',
    y='generosity_index:Q',
    y2='generosity_index2:Q',
    color=alt.Color('color:N', scale=None)
)

# dark overlay for non-targeted quadrants
overlay_quads = pd.DataFrame([
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen},       # Bottom Left (Red)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen},   # Bottom Right (Yellow)
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g}    # Top Left (Blue)
])
chart_overlay = alt.Chart(overlay_quads).mark_rect(color='#1a1d24', opacity=0.45).encode(
    x='participation_rate:Q', x2='participation_rate2:Q', y='generosity_index:Q', y2='generosity_index2:Q'
)

# overlay layered scatter
layered_scatter = (chart_quads + rule_x + rule_y + chart_binned_scatter + chart_overlay + rule_x_target + rule_y_target).configure(background=chart_bg)
st.altair_chart(layered_scatter, width='stretch', theme=None)

# divider
st.divider()
st.subheader("State Averages")
if len(selected_years_range) > 1:
    st.caption(f"Showing data for **{display_year}** (last selected year).")
with st.expander("How is this calculated?"):
    st.markdown(
        "State-level averages, computed as the mean across all ZIP codes within each state for the selected filters.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "  - N1 = Total Returns - total number of tax returns filed"
    )

map_metric_label = st.segmented_control(
    "Color State Map By:",
    options=["Generosity", "Participation"],
    default="Generosity"
)
map_metric_label = map_metric_label or "Generosity"
map_metric_map = {"Generosity": "Generosity Index", "Participation": "Participation Rate"}
map_metric = map_metric_map[map_metric_label]

# setup map axes
if map_metric == 'Generosity Index':
    metric_col, metric_title, map_colors = 'generosity_index', 'Avg Generosity', ['#e9d5ff', '#4c1d95']
else:
    metric_col, metric_title, map_colors = 'participation_rate', 'Avg Participation', ['#fca5a5', '#7f1d1d']

# group state avgs — use single display year when multiple years selected
state_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
state_avg = state_source.groupby('STATE', as_index=False)[['generosity_index', 'participation_rate']].mean().round(4)

# build fips dict
state_fips = {
    'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8, 'CT': 9, 'DE': 10, 'FL': 12, 'GA': 13,
    'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19, 'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24,
    'MA': 25, 'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30, 'NE': 31, 'NV': 32, 'NH': 33, 'NJ': 34,
    'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39, 'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45,
    'SD': 46, 'TN': 47, 'TX': 48, 'UT': 49, 'VT': 50, 'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56,
    'DC': 11
}
# assign map lookups
state_avg['id'] = state_avg['STATE'].map(state_fips)
state_avg = state_avg.dropna(subset=['id'])

# load map json
url = 'https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/us-10m.json'
states = alt.topo_feature(url, 'states')

# render map chart
chart_map = alt.Chart(states).mark_geoshape().encode(
    color=alt.Color(f'{metric_col}:Q', scale=alt.Scale(range=map_colors), title=metric_title, legend=alt.Legend(labelColor=axis_color, titleColor=axis_color, format='.1%')),
    tooltip=[
        alt.Tooltip('STATE:N', title='State'),
        alt.Tooltip(f'{metric_col}:Q', title=metric_title, format='.1%')
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(state_avg, 'id', [metric_col, 'STATE'])
).project('albersUsa').properties(height=500).configure(background=chart_bg)

# embed map
st.altair_chart(chart_map, width='stretch', theme=None)

# divider
st.divider()

col_dist, col_zip = st.columns(2)

with col_dist:
    st.subheader("Generosity Distribution")
    if len(selected_years_range) > 1:
        st.caption(f"Showing data for **{display_year}** (last selected year).")
    with st.expander("How is this calculated?"):
        st.markdown(
            "Histogram of **Generosity Index** = A19700 / A00100 across all ZIP codes in the selected filter.\n\n"
            "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
            "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
            "Each bar counts the number of ZIP codes whose Generosity Index falls within that bin. The dashed red line marks the dataset average."
        )
    hist_bins = st.slider("Histogram Detail Level (Bins):", min_value=10, max_value=100, value=50, step=10)
    hist_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
    # render histogram
    chart3 = alt.Chart(hist_source).mark_bar().encode(
        x=alt.X('generosity_index:Q', bin=alt.Bin(maxbins=hist_bins), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        y=alt.Y('count()', title='Number of ZIP Codes', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
        color=alt.Color('count()', scale=THEME_PALETTE, title='Count of ZIP Codes', legend=None),
        tooltip=[alt.Tooltip('count()', title='Count')]
    ).properties(height=400)

    # overlay histogram
    rule_hist = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='avg_gen:Q')
    layered_hist = (chart3 + rule_hist).configure(background=chart_bg)
    st.altair_chart(layered_hist, width='stretch', theme=None)

with col_zip:
    st.subheader("Reportable Giving ZIP Codes by Year")
    with st.expander("How is this calculated?"):
        st.markdown(
            "Count of ZIP codes per tax year satisfying the reportability threshold: A19700 > 0 and N1 >= 100.\n\n"
            "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
            "- N1 = Total Returns - total number of tax returns filed"
        )
    if len(selected_years_range) < 2:
        st.info("Select a range of two or more years in the sidebar to view the trend chart.")

    # load full unfiltered dataset for this chart so year/state filters don't distort the trend
    @st.cache_data
    def get_zip_counts(data):
        return data.groupby('year', as_index=False).size().rename(columns={'size': 'zip_count'})

    zip_counts = get_zip_counts(full_df)
    # filter to dynamic global year range
    zip_counts = zip_counts[zip_counts['year'].isin(selected_years_range)].copy()

    # add era column so line breaks over 2017-2018 TCJA gap
    zip_counts['era'] = zip_counts['year'].apply(lambda y: 'pre' if y <= 2017 else 'post')

    chart_floor = alt.Chart(zip_counts).mark_line(point=alt.OverlayMarkDef(opacity=1), color='#38bdf8').encode(
        x=alt.X('year:Q', title='Tax Year', scale=alt.Scale(domain=[min_year, max_year], zero=False, nice=False), axis=alt.Axis(format='d', tickMinStep=1, labelColor=axis_color, titleColor=axis_color)),
        y=alt.Y('zip_count:Q', title='ZIP Codes with Reportable Giving', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
        detail='era:N',
        tooltip=[
            alt.Tooltip('year:Q', title='Year'),
            alt.Tooltip('zip_count:Q', title='ZIP Count', format=',')
        ]
    ).properties(height=350)

    max_y = zip_counts['zip_count'].max() if not zip_counts.empty else 1000

    if spans_tcja:
        tcja_rect = alt.Chart(pd.DataFrame({'start': [2017], 'end': [2018]})).mark_rect(
            color='#4b5563', opacity=0.3
        ).encode(x='start:Q', x2='end:Q')
        tcja_label = alt.Chart(pd.DataFrame({'x': [2017.5], 'y': [max_y * 0.95], 'label': ['TCJA Gap']})).mark_text(
            color='#9ca3af', fontSize=10, fontWeight='bold', dy=-5
        ).encode(x='x:Q', y='y:Q', text='label:N')
        layered_floor = (tcja_rect + chart_floor + tcja_label).configure(background=chart_bg)
    else:
        layered_floor = chart_floor.configure(background=chart_bg)
    st.altair_chart(layered_floor, width='stretch', theme=None)

# divider
st.divider()
st.subheader("Generosity Trend by State")
with st.expander("How is this calculated?"):
    st.markdown(
        "Average **Generosity Index** = A19700 / A00100 per state per tax year, computed across all ZIP codes within the state.\n\n"
        "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
        "Use the state filter to compare or isolate specific states."
    )

# only show trend if multiple years in the range
if len(selected_years_range) < 2:
    st.info("Select a range of two or more years in the sidebar to view the trend chart.")
else:
    # use dataframe already filtered by year range and states
    trend_df = df.groupby(['year', 'STATE'], as_index=False)['generosity_index'].mean().round(4)
    
    trend_df['era'] = trend_df['year'].apply(lambda y: 'pre' if y <= 2017 else 'post')

    highlight = alt.selection_point(on='pointerover', fields=['STATE'], nearest=True)

    base_line = alt.Chart(trend_df).encode(
        x=alt.X('year:Q', title='Tax Year', scale=alt.Scale(domain=[min_year, max_year], zero=False, nice=False), axis=alt.Axis(format='d', tickMinStep=1, labelColor=axis_color, titleColor=axis_color)),
        y=alt.Y('generosity_index:Q', title='Avg Generosity Index', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        detail=alt.Detail(['STATE:N', 'era:N']),
        tooltip=[
            alt.Tooltip('STATE:N', title='State'),
            alt.Tooltip('year:Q', title='Year'),
            alt.Tooltip('generosity_index:Q', title='Avg Generosity Index', format='.2%')
        ]
    )

    lines = base_line.mark_line(point=alt.OverlayMarkDef(opacity=1)).encode(
        color=alt.condition(highlight, alt.value('#38bdf8'), alt.value('#2563eb')),
        size=alt.condition(~highlight, alt.value(1), alt.value(3)),
        opacity=alt.condition(~highlight, alt.value(0.7), alt.value(1))
    )

    points = base_line.mark_circle().encode(
        opacity=alt.value(0)
    ).add_params(highlight)
    
    max_y_trend = trend_df['generosity_index'].max() if not trend_df.empty else 0.1

    if spans_tcja:
        tcja_rect_trend = alt.Chart(pd.DataFrame({'start': [2017], 'end': [2018]})).mark_rect(
            color='#4b5563', opacity=0.3
        ).encode(x='start:Q', x2='end:Q')
        tcja_label_trend = alt.Chart(pd.DataFrame({'x': [2017.5], 'y': [max_y_trend * 0.95], 'label': ['TCJA Gap']})).mark_text(
            color='#9ca3af', fontSize=10, fontWeight='bold', dy=-5
        ).encode(x='x:Q', y='y:Q', text='label:N')
        interactive_trend = (tcja_rect_trend + tcja_label_trend + lines + points).properties(height=400).configure(background=chart_bg)
    else:
        interactive_trend = (lines + points).properties(height=400).configure(background=chart_bg)

    st.altair_chart(interactive_trend, width='stretch', theme=None)

# divider
st.divider()
st.subheader("Donor Momentum")
with st.expander("How is this calculated?"):
    st.markdown(
        "Dumbbell chart showing each ZIP code's **Generosity Index** at the start and end of the selected year range.\n\n"
        "**Momentum** = GI at end year − GI at start year, for ZIP codes present in both boundary years. "
        "Each row is one ZIP code. The left dot marks the starting GI, the right dot marks the ending GI, "
        "and the connecting bar spans the change. "
        "Gainers (end GI > start GI) are shown in **blue**; decliners in **red**.\n\n"
        "- **Generosity Index (GI)** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)"
    )

if len(selected_years_range) < 2:
    st.info("Select a range of two or more years in the sidebar to view the momentum chart.")
else:
    mom_col1, mom_col2 = st.columns(2)
    with mom_col1:
        momentum_n = st.slider("Top N ZIP Codes by Momentum:", min_value=3, max_value=30, value=15, step=1)
    with mom_col2:
        start_year, end_year = st.select_slider(
            "Momentum Year Range:",
            options=selected_years_range,
            value=(min(selected_years_range), max(selected_years_range))
        )
        if start_year == end_year:
            st.warning("Select two different years to calculate momentum.")
            st.stop()
    gi_start = df[df['year'] == start_year][['zipcode', 'STATE', 'generosity_index']].rename(columns={'generosity_index': 'gi_start'})
    gi_end   = df[df['year'] == end_year  ][['zipcode', 'generosity_index']].rename(columns={'generosity_index': 'gi_end'})
    momentum_base = gi_start.merge(gi_end, on='zipcode')
    momentum_base['momentum'] = momentum_base['gi_end'] - momentum_base['gi_start']

    # top N by absolute momentum (biggest movers either direction)
    momentum_zips = momentum_base.reindex(momentum_base['momentum'].abs().nlargest(momentum_n).index)['zipcode'].tolist()

    dumbbell_df = momentum_base[momentum_base['zipcode'].isin(momentum_zips)].copy()
    dumbbell_df = dumbbell_df.sort_values('momentum', ascending=True)
    zip_order = dumbbell_df['zipcode'].tolist()
    chart_height = max(300, len(zip_order) * 24)

    shared_tooltip = [
        alt.Tooltip('zipcode:N', title='ZIP Code'),
        alt.Tooltip('STATE:N',   title='State'),
        alt.Tooltip('gi_start:Q', title=f'GI {start_year}', format='.2%'),
        alt.Tooltip('gi_end:Q',   title=f'GI {end_year}',   format='.2%'),
        alt.Tooltip('momentum:Q', title='GI Change',         format='+.2%'),
    ]

    # connecting bar — neutral mid tone
    bar_layer = alt.Chart(dumbbell_df).mark_rule(strokeWidth=2, color='#334155').encode(
        y=alt.Y('zipcode:N', sort=zip_order, title='ZIP Code',
                axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelFontSize=11)),
        x=alt.X('gi_start:Q', title='Generosity Index (%)',
                axis=alt.Axis(labelColor=axis_color, titleColor=axis_color,
                              labelExpr="format(datum.value * 100, '.1f')", tickCount=6)),
        x2=alt.X2('gi_end:Q'),
        tooltip=shared_tooltip,
    )

    # start node — dark blue (#1e3a8a)
    dot_start = alt.Chart(dumbbell_df).mark_circle(size=70, color='#1e3a8a', opacity=1.0).encode(
        y=alt.Y('zipcode:N', sort=zip_order),
        x=alt.X('gi_start:Q'),
        tooltip=shared_tooltip,
    )

    # end node — light blue (#38bdf8)
    dot_end = alt.Chart(dumbbell_df).mark_circle(size=70, color='#38bdf8', opacity=1.0).encode(
        y=alt.Y('zipcode:N', sort=zip_order),
        x=alt.X('gi_end:Q'),
        tooltip=shared_tooltip,
    )

    mom_chart = (bar_layer + dot_start + dot_end).properties(height=chart_height).configure(background=chart_bg)
    st.altair_chart(mom_chart, width='stretch', theme=None)

# divider
st.divider()
st.subheader("ZIP Code Cluster Analysis")
with st.expander("How is this calculated?"):
    st.markdown(
        "K-Means clustering groups ZIP codes into **k** segments based on Generosity Index, Participation Rate, and Average Household Income. "
        "Features are standardized (zero mean, unit variance) before clustering so no single metric dominates.\n\n"
        "- **Generosity Index** = A19700 / A00100\n"
        "- **Participation Rate** = N19700 / N1\n"
        "- **Avg Household Income** = (A00100 × 1,000) / N1\n\n"
        "Clusters are ranked by a composite score (GI + Participation Rate) and assigned meaningful labels from lowest to highest engagement. "
        "Color runs **dark blue → light blue** from lowest to highest engagement. "
        "When multiple years are selected, values are averaged per ZIP code before clustering."
    )

k_col1, k_col2, k_col3 = st.columns(3)
with k_col1:
    k_clusters = st.slider("Number of Clusters (k):", min_value=1, max_value=4, value=3, step=1)
with k_col2:
    min_filers = st.slider("Min Filers per ZIP (N1):", min_value=0, max_value=1000, value=100, step=50)
with k_col3:
    cluster_rank_by = st.segmented_control(
        "Rank ZIP Codes By:",
        options=["Generosity", "Participation", "HH Income"],
        default="Generosity"
    )
    cluster_rank_by = cluster_rank_by or "Generosity"

cluster_rank_col = {"Generosity": "generosity_index", "Participation": "participation_rate", "HH Income": "avg_hh_income"}

# label definitions per k — ordered lowest → highest engagement
CLUSTER_LABELS = {
    1: [
        ("General Population",    "All ZIP codes fall into a single segment. Increase k to reveal sub-groups.")
    ],
    2: [
        ("Low Engagement",        "Below-average generosity and participation. Fewer itemizing donors relative to total filers."),
        ("High Engagement",       "Above-average generosity and participation. Strong donor base relative to total filers."),
    ],
    3: [
        ("Emerging Donors",       "Low generosity and participation. Likely standard-deduction-dominated or lower-income ZIP codes with limited itemizing activity."),
        ("Moderate Givers",       "Mid-range generosity and participation. Solid donor presence but not yet standout giving depth."),
        ("High-Value Donors",     "High generosity and participation. Core target audience — deep giving culture with broad filer involvement."),
    ],
    4: [
        ("Low Engagement",        "Low generosity and participation with modest household incomes. Limited itemizing donor activity."),
        ("Broad but Shallow",     "Higher participation but lower generosity per filer. Many donors, but smaller average gift relative to income."),
        ("Deep but Narrow",       "High generosity index among a smaller share of filers. Concentrated giving — fewer but high-value donors."),
        ("High-Value Donors",     "Top generosity and participation with higher household incomes. Strongest prospect segment for major gift outreach."),
    ],
}

# aggregate per zip across years before clustering
cluster_input = df.groupby('zipcode', as_index=False).agg(
    generosity_index=('generosity_index', 'mean'),
    participation_rate=('participation_rate', 'mean'),
    N1=('N1', 'mean'),
    A00100=('A00100', 'mean'),
).copy()
cluster_input['avg_hh_income'] = (cluster_input['A00100'] * 1000) / cluster_input['N1']
cluster_input = cluster_input.dropna(subset=['generosity_index', 'participation_rate', 'avg_hh_income'])

# apply filters
cluster_input = cluster_input[cluster_input['N1'] >= min_filers]

rank_col = cluster_rank_col[cluster_rank_by]

st.caption(f"Clustering **{len(cluster_input):,}** ZIP codes — filtered to N1 ≥ {min_filers:,}, ranked by {cluster_rank_by}.")

if cluster_input.empty:
    st.warning("No ZIP codes match the current filters. Lower the minimum filers threshold.")
    st.stop()

features = cluster_input[['generosity_index', 'participation_rate', 'avg_hh_income']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
cluster_input['cluster_id'] = kmeans.fit_predict(features_scaled)

# rank clusters by user-selected metric low → high, map to labels
cluster_stats = cluster_input.groupby('cluster_id')[['generosity_index', 'participation_rate', 'avg_hh_income']].mean()
rank_order = cluster_stats[rank_col].rank(method='first').astype(int) - 1  # 0-indexed rank
label_list = CLUSTER_LABELS[k_clusters]
cluster_input['cluster_label'] = cluster_input['cluster_id'].map(
    lambda cid: label_list[rank_order[cid]][0]
)

# dark → light blue palette using app colors (lowest → highest engagement)
all_cluster_colors = ['#1e3a8a', '#1d4ed8', '#38bdf8', '#bae6fd']
ordered_labels = [label_list[i][0] for i in range(k_clusters)]
cluster_scale = alt.Scale(domain=ordered_labels, range=all_cluster_colors[:k_clusters])

cluster_chart = alt.Chart(cluster_input).mark_circle(opacity=0.55, size=25).encode(
    x=alt.X('participation_rate:Q', title='Participation Rate (%)',
            axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    y=alt.Y('generosity_index:Q', title='Generosity Index (%)',
            axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    color=alt.Color('cluster_label:N', scale=cluster_scale, title='Cluster',
                    legend=alt.Legend(labelColor=axis_color, titleColor=axis_color)),
    tooltip=[
        alt.Tooltip('zipcode:N', title='ZIP Code'),
        alt.Tooltip('cluster_label:N', title='Cluster'),
        alt.Tooltip('participation_rate:Q', title='Participation Rate', format='.2%'),
        alt.Tooltip('generosity_index:Q', title='Generosity Index', format='.2%'),
        alt.Tooltip('avg_hh_income:Q', title='Avg HH Income', format='$,.0f'),
    ]
).properties(height=450).configure(background=chart_bg)

st.altair_chart(cluster_chart, width='stretch', theme=None)

# cluster legend with explanations
st.markdown("**Cluster Descriptions**")
legend_cols = st.columns(k_clusters)
for i, col in enumerate(legend_cols):
    label, desc = label_list[i]
    color = all_cluster_colors[i]
    text_color = '#111827' if color in ('#38bdf8', '#bae6fd') else '#ffffff'
    col.markdown(
        f"<div style='background:{color}; color:{text_color}; padding:10px 12px; border-radius:6px;'>"
        f"<strong>{label}</strong><br><span style='font-size:0.82rem'>{desc}</span></div>",
        unsafe_allow_html=True
    )

# ─── AI Chat Logic ────────────────────────────────────────────────────────────
_seg_df = pd.DataFrame()
_rising_df = pd.DataFrame()

try:
    _seg_df = cluster_input.groupby('cluster_label', as_index=False).agg(
        ZIP_Codes=('zipcode', 'count'),
        Avg_GI=('generosity_index', 'mean'),
        Avg_PR=('participation_rate', 'mean'),
        Avg_Income=('avg_hh_income', 'mean')
    ).rename(columns={'cluster_label': 'Segment'})
    _seg_df['Avg_GI'] = (_seg_df['Avg_GI'] * 100).round(2)
    _seg_df['Avg_PR'] = (_seg_df['Avg_PR'] * 100).round(2)
except Exception:
    pass

try:
    _rising_df = dumbbell_df[dumbbell_df['momentum'] > 0].copy()
    _rising_df = _rising_df.rename(columns={'momentum': 'momentum_score', 'gi_end': 'generosity_index'})[
        ['zipcode', 'STATE', 'momentum_score', 'generosity_index']
    ].sort_values('momentum_score', ascending=False)
except Exception:
    pass

# update context in session state so the always-visible chat input can use it
st.session_state['_ai_context'] = build_context(top_n_df, _seg_df, _rising_df, selected_states, selected_years_range)
