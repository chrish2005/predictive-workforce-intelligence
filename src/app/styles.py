"""
Executive HR Theme and Design System for Streamlit Application.
Provides modern glassmorphic styling, responsive layout tokens, and Plotly chart themes.
"""

# Modern Design Tokens
PRIMARY_NAVY = "#0F172A"
SLATE_DARK = "#1E293B"
ACCENT_BLUE = "#3B82F6"
ACCENT_INDIGO = "#6366F1"
SUCCESS_GREEN = "#10B981"
WARNING_AMBER = "#F59E0B"
DANGER_RED = "#EF4444"
BG_LIGHT = "#F8FAFC"
CARD_BG = "#FFFFFF"
BORDER_SUBTLE = "#E2E8F0"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#64748B"

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Executive Header Banner */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    /* Active Cohort Pill */
    .cohort-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }

    /* Polished Metric Cards */
    .metric-card {
        background: #ffffff;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
        border-color: #cbd5e1;
    }
    .metric-card.accent-red { border-top: 4px solid #ef4444; }
    .metric-card.accent-blue { border-top: 4px solid #3b82f6; }
    .metric-card.accent-green { border-top: 4px solid #10b981; }
    .metric-card.accent-amber { border-top: 4px solid #f59e0b; }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 700;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.3rem;
        letter-spacing: -0.02em;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Status Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid #fca5a5;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid #fcd34d;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        border: 1px solid #6ee7b7;
    }

    /* Card Panels */
    .card-panel {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Section Subheadings */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
</style>
"""

PLOTLY_LAYOUT_TEMPLATE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#1E293B", "size": 12},
    "margin": {"l": 40, "r": 25, "t": 35, "b": 35},
    "xaxis": {
        "gridcolor": "#F1F5F9",
        "linecolor": "#E2E8F0",
        "showline": True,
        "zeroline": False,
    },
    "yaxis": {
        "gridcolor": "#F1F5F9",
        "linecolor": "#E2E8F0",
        "showline": True,
        "zeroline": False,
    },
}
