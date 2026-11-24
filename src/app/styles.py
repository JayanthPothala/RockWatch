"""
RockWatch Web Application - Custom CSS Styling & Theme Configuration
Provides custom CSS to give Streamlit a modern environmental intelligence UI.
"""

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base Settings */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* Top Header Branding Banner */
    .rock-brand-banner {
        background: linear-gradient(135deg, #132a1e 0%, #1e4533 50%, #2b5942 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(19, 42, 30, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .rock-brand-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .rock-banner-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        border: 1px solid rgba(110, 231, 183, 0.3);
        margin-bottom: 0.8rem;
    }

    .rock-banner-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        line-height: 1.2;
    }

    .rock-banner-subtitle {
        font-size: 1.02rem;
        color: #cbd5e1;
        margin-top: 0.6rem;
        margin-bottom: 0;
        max-width: 850px;
        line-height: 1.5;
    }

    /* Metric Cards */
    .rock-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }

    .rock-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }

    .rock-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    .rock-metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .rock-metric-subtext {
        font-size: 0.82rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Tier Badges */
    .tier-badge {
        display: inline-block;
        font-weight: 700;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        font-size: 0.85rem;
    }

    .tier-low {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }

    .tier-moderate {
        background: #fefce8;
        color: #854d0e;
        border: 1px solid #fef08a;
    }

    .tier-high {
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }

    .tier-very-high {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* General Content Cards */
    .rock-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.04);
    }

    .rock-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom Table Styling */
    .dataframe {
        font-size: 0.88rem !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background: #0f1f16 !important;
        border-right: 1px solid #1c3527 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background: transparent;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        transition: background 0.15s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.07);
    }

    /* Button Styling */
    .stButton>button {
        background: #1e4533;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px -2px rgba(30, 69, 51, 0.4);
    }

    .stButton>button:hover {
        background: #2b5942;
        box-shadow: 0 6px 15px -2px rgba(30, 69, 51, 0.6);
        transform: translateY(-1px);
    }

    /* Progress bar custom color */
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }

    /* Alert / Warning custom box */
    .rock-disclaimer-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #334155;
    }
</style>
"""


def get_metric_card(label: str, value: str, subtext: str = None, is_positive: bool = True) -> str:
    """Generate HTML for an environmental intelligence metric card."""
    subtext_html = ""
    if subtext:
        sub_color = "#10b981" if is_positive else "#ef4444"
        subtext_html = f'<div class="rock-metric-subtext" style="color: {sub_color};">{subtext}</div>'

    return f"""
    <div class="rock-metric-card">
        <div class="rock-metric-label">{label}</div>
        <div class="rock-metric-value">{value}</div>
        {subtext_html}
    </div>
    """


def get_tier_badge(tier_name: str) -> str:
    """Return styled HTML badge for hazard tiers."""
    t_lower = tier_name.lower().replace(" ", "-")
    return f'<span class="tier-badge tier-{t_lower}">{tier_name.upper()}</span>'

