import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import fpdf
from io import StringIO
from io import BytesIO
from fpdf import FPDF
from datetime import datetime


def process_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = [
            'Right Step Time (sec)', 'Right Step Length (meters)', 'Right Cadence (steps/min)',
            'Right Swing Time (sec)', 'Right Stance Time (sec)', 'GaitSpeed Rtable (mph*10)',
            'Right Stride Time (sec)', 'Right Stride Length (meters)', 'Left Step Time (sec)',
            'Left Step Length (meters)', 'Left Cadence (steps/min)', 'Left Swing Time (sec)',
            'Left Stance Time (sec)', 'GaitSpeed Ltable (mph*10)', 'Left Stride Time (sec)',
            'Left Stride Length (meters)'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        df = df.dropna(how='all')
        for col in required_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in required_columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
        if 'Time' not in df.columns:
            df['Time'] = range(len(df))
        return df, None
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"


def generate_fake_foot_tracking_data(num_steps: int = 10):
    steps = []
    y_position = 0.0
    step_length = 0.75
    lateral_offset = 0.15
    for i in range(num_steps):
        y_position += step_length
        foot = "Right" if i % 2 == 0 else "Left"
        x = lateral_offset if foot == "Right" else -lateral_offset
        steps.append({"step": i, "foot": foot, "x": x, "y": y_position, "phase": "contact"})
    return pd.DataFrame(steps)

def create_overhead_foot_tracking_visualization(current_step: int = 0, num_steps: int = 10,
                                                target_step_length: float = 0.75,
                                                is_patient: bool = False):
    """Clean overhead footstep graphic – dots only, white background, no lines."""
    df_track = generate_fake_foot_tracking_data(num_steps)

    fig = go.Figure()

    color_map = {"Right": "#d32f2f", "Left": "#1565C0"}

    for idx, row in df_track.iterrows():
        if idx == current_step:
            size = 38
            opacity = 1.0
            border_width = 4
        elif idx < current_step:
            size = 26
            opacity = 0.65
            border_width = 2
        else:
            size = 20
            opacity = 0.25
            border_width = 1

        fig.add_trace(
            go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color_map[row["foot"]],
                    opacity=opacity,
                    symbol="circle",
                    line=dict(width=border_width, color="white"),
                ),
                name=f"{row['foot']} Foot",
                hovertemplate=(
                    f"<b>{row['foot']} Foot</b><br>"
                    f"Step: {row['step']}<br>"
                    f"Position: {row['y']:.2f} m<br>"
                    "<extra></extra>"
                ),
                showlegend=(idx in [0, 1]),
            )
        )

    fig.add_annotation(
        x=0,
        y=num_steps * target_step_length + 0.6,
        text="↑ Direction of Travel",
        showarrow=False,
        font=dict(size=14, color="#555555"),
    )

    title_text = (
        "<b>Your Stepping Target</b>" if is_patient
        else f"<b>Stepping Target  (Step {current_step})</b>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color="#d32f2f")),
        xaxis=dict(
            title=dict(text="Left  ←  |  →  Right (m)", font=dict(size=13, color="#222222")),
            range=[-0.45, 0.45],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#cccccc",
            tickfont=dict(size=12, color="#333333"),
        ),
        yaxis=dict(
            title=dict(text="Forward (m)", font=dict(size=13, color="#222222")),
            range=[-0.3, num_steps * target_step_length + 1.0],
            showgrid=False,
            tickfont=dict(size=12, color="#333333"),
        ),
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#222222"),
        ),
        margin=dict(l=70, r=70, t=80, b=90),
    )

    return fig

def create_hero_summary(stats, session_time, distance, is_patient=False):
    if is_patient:
        hero_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 36px; border-radius: 18px; color: white; text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <h1 style="margin:0; font-size:2.8em; font-weight:bold;">Great Work!</h1>
            <p style="font-size:1.2em; opacity:0.85; margin-top:8px;">Here's how you did today</p>
            <div style="display:flex; justify-content:space-around; margin-top:28px; flex-wrap:wrap;">
                <div style="flex:1; min-width:150px; margin:12px;">
                    <div style="font-size:3.2em; font-weight:bold;">{distance:.0f}</div>
                    <div style="font-size:1.3em; opacity:0.9;">meters walked</div>
                </div>
                <div style="flex:1; min-width:150px; margin:12px;">
                    <div style="font-size:3.2em; font-weight:bold;">{session_time:.1f}</div>
                    <div style="font-size:1.3em; opacity:0.9;">minutes</div>
                </div>
                <div style="flex:1; min-width:150px; margin:12px;">
                    <div style="font-size:3.2em; font-weight:bold;">{stats['avg_speed']:.1f}</div>
                    <div style="font-size:1.3em; opacity:0.9;">mph walking speed</div>
                </div>
                <div style="flex:1; min-width:150px; margin:12px;">
                    <div style="font-size:3.2em; font-weight:bold;">{stats['avg_right_cadence']:.0f}</div>
                    <div style="font-size:1.3em; opacity:0.9;">steps per minute</div>
                </div>
            </div>
        </div>
        """
    else:
        hero_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <h1 style="margin:0; font-size:2.5em; font-weight:bold;">Session Complete!</h1>
            <div style="display:flex; justify-content:space-around; margin-top:30px; flex-wrap:wrap;">
                <div style="flex:1; min-width:150px; margin:10px;">
                    <div style="font-size:3em; font-weight:bold;">{distance:.2f}</div>
                    <div style="font-size:1.2em; opacity:0.9;">meters</div>
                    <div style="font-size:0.9em; opacity:0.7;">Total Distance</div>
                </div>
                <div style="flex:1; min-width:150px; margin:10px;">
                    <div style="font-size:3em; font-weight:bold;">{session_time:.1f}</div>
                    <div style="font-size:1.2em; opacity:0.9;">minutes</div>
                    <div style="font-size:0.9em; opacity:0.7;">Session Time</div>
                </div>
                <div style="flex:1; min-width:150px; margin:10px;">
                    <div style="font-size:3em; font-weight:bold;">{stats['avg_speed']:.2f}</div>
                    <div style="font-size:1.2em; opacity:0.9;">mph</div>
                    <div style="font-size:0.9em; opacity:0.7;">Average Speed</div>
                </div>
                <div style="flex:1; min-width:150px; margin:10px;">
                    <div style="font-size:3em; font-weight:bold;">{stats['avg_right_cadence']:.0f}</div>
                    <div style="font-size:1.2em; opacity:0.9;">steps/min</div>
                    <div style="font-size:0.9em; opacity:0.7;">Average Cadence</div>
                </div>
            </div>
        </div>
        """
    return hero_html


def create_real_time_symmetry_bars(left_value, right_value, metric_name,
                                   target_min=None, target_max=None,
                                   left_label="Left", right_label="Right"):
    fig = go.Figure()

    def get_color(value, tmn, tmx):
        if tmn is not None and tmx is not None:
            if tmn <= value <= tmx:
                return "#28a745"
            elif (tmn * 0.9 <= value < tmn) or (tmx < value <= tmx * 1.1):
                return "#ffc107"
            else:
                return "#dc3545"
        return "#6c757d"

    left_color = get_color(left_value, target_min, target_max) if target_min else "#4ECDC4"
    right_color = get_color(right_value, target_min, target_max) if target_min else "#FF6B6B"

    fig.add_trace(go.Bar(x=[left_value], y=[left_label], orientation='h', name=left_label,
                         marker=dict(color=left_color), text=[f'{left_value:.2f}'], textposition='auto'))
    fig.add_trace(go.Bar(x=[right_value], y=[right_label], orientation='h', name=right_label,
                         marker=dict(color=right_color), text=[f'{right_value:.2f}'], textposition='auto'))

    if target_min and target_max:
        fig.add_vrect(x0=target_min, x1=target_max, fillcolor="green", opacity=0.1,
                      layer="below", line_width=0)

    fig.update_layout(title=f"Real-Time Balance: {metric_name}", xaxis_title=metric_name,
                      height=200, showlegend=False,
                      margin=dict(l=80, r=40, t=60, b=40), plot_bgcolor="white")
    return fig


def calculate_asymmetry_percentage(left_value, right_value):
    if left_value == 0 or right_value == 0:
        return 0
    avg = (left_value + right_value) / 2
    diff = abs(right_value - left_value)
    return (diff / avg) * 100


def create_fatigue_analysis(df, metric_name):
    first_third = df.iloc[:len(df) // 3]
    last_third = df.iloc[-len(df) // 3:]
    start_mean = first_third[metric_name].mean()
    end_mean = last_third[metric_name].mean()
    change = ((end_mean - start_mean) / start_mean) * 100
    fig = go.Figure()
    fig.add_trace(go.Box(y=first_third[metric_name], name="Start", marker_color="#4ECDC4"))
    fig.add_trace(go.Box(y=last_third[metric_name], name="End", marker_color="#FF6B6B"))
    fig.update_layout(title=f"Fatigue Analysis: {metric_name}<br><sub>Change: {change:+.1f}%</sub>",
                      yaxis_title=metric_name, height=400, showlegend=True)
    return fig

def initialize_session_state():
    defaults = {
        'user_mode': None,
        'logged_in': False,
        'user_email': "",
        'patient_profiles': {},
        'session_history': [],
        'favorite_metrics': [],
        'target_zones': {},
        'self_pacing_enabled': False,
        'admin_target_step_length': 0.75,
        'admin_color_coding': True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(
    page_title="Gait Analysis System",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── shared ── */
    .main-header {
        font-size: 2.5em; color: #d32f2f; text-align: center;
        margin-bottom: 1rem; font-weight: bold;
    }
    .sub-header { font-size: 1.8rem; color: #0055cc; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }

    .large-metric { font-size: 4em; font-weight: bold; text-align: center; padding: 30px; border-radius: 15px; }
    .medium-metric { font-size: 2em; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; }

    .status-green  { background-color: #28a745; color: white; }
    .status-yellow { background-color: #ffc107; color: black; }
    .status-red    { background-color: #dc3545; color: white; }
    .status-gray   { background-color: #6c757d; color: white; }

    .assessment-box { background-color: transparent; color: #d32f2f; border: 1px solid #d32f2f; border-radius: 5px; padding: 14px; margin-bottom: 10px; }
    .assessment-box p { color: #FFFFFF; }
    .statistics-group { border: 2px solid #6c757d; border-radius: 5px; padding: 15px; margin-top: 20px; }
    .metric-separator { border-top: 1px solid #000; margin: 10px 0; }
    .stButton>button { width: 100%; height: 60px; font-size: 1.2em; }

    /* ── patient-specific overrides ── */
    .patient-large-metric {
        font-size: 5em; font-weight: bold; text-align: center;
        padding: 36px; border-radius: 18px; line-height: 1.1;
        border: 3px solid rgba(255,255,255,0.4);
    }
    .patient-medium-metric {
        font-size: 3em; font-weight: bold; text-align: center;
        padding: 26px; border-radius: 14px; line-height: 1.1;
        border: 3px solid rgba(255,255,255,0.4);
    }
    .patient-label { font-size: 0.38em; opacity: 0.85; letter-spacing: 0.05em; text-transform: uppercase; }
    .patient-unit  { font-size: 0.45em; opacity: 0.9; }
    .patient-section { font-size: 1.6em; color: #d32f2f; font-weight: bold; margin: 1.2rem 0 0.6rem; }
    .patient-score-box {
        background: #1a1a2e; color: #eee; border-radius: 16px;
        padding: 24px; text-align: center; border: 2px solid #d32f2f;
    }
    .patient-score-value { font-size: 4em; font-weight: bold; color: #ffffff; }
    .patient-score-label { font-size: 1.1em; color: #aaa; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

def calculate_statistics(df):
    stats = {}
    stats['avg_right_cadence'] = df['Right Cadence (steps/min)'].mean()
    stats['avg_right_speed'] = df['GaitSpeed Rtable (mph*10)'].mean() / 10
    stats['avg_right_step_length'] = df['Right Step Length (meters)'].mean()
    stats['avg_right_stride_length'] = df['Right Stride Length (meters)'].mean()
    stats['avg_left_cadence'] = df['Left Cadence (steps/min)'].mean()
    stats['avg_left_speed'] = df['GaitSpeed Ltable (mph*10)'].mean() / 10
    stats['avg_left_step_length'] = df['Left Step Length (meters)'].mean()
    stats['avg_left_stride_length'] = df['Left Stride Length (meters)'].mean()
    stats['avg_speed'] = (stats['avg_right_speed'] + stats['avg_left_speed']) / 2
    stats['right_swing_stance_ratio'] = df['Right Swing Time (sec)'].mean() / df['Right Stance Time (sec)'].mean()
    stats['left_swing_stance_ratio'] = df['Left Swing Time (sec)'].mean() / df['Left Stance Time (sec)'].mean()
    stats['temporal_asymmetry'] = calculate_asymmetry_percentage(
        df['Left Step Time (sec)'].mean(), df['Right Step Time (sec)'].mean())
    stats['spatial_asymmetry'] = calculate_asymmetry_percentage(
        df['Left Step Length (meters)'].mean(), df['Right Step Length (meters)'].mean())
    return stats

def render_patient_ui(df, stats, session_time, total_distance,
                      target_cadence_min, target_cadence_max,
                      target_step_length_min, target_step_length_max,
                      color_coding_on, admin_target_step_length):
    """Render the full patient-facing interface."""

    def status_class(value, tmin, tmax):
        if not color_coding_on:
            return "status-gray"
        if tmin <= value <= tmax:
            return "status-green"
        elif (tmin * 0.9 <= value < tmin) or (tmax < value <= tmax * 1.1):
            return "status-yellow"
        return "status-red"

    tab_live, tab_done, tab_history, tab_target = st.tabs([
        "🏃 Live Session",
        "✅ Session Summary",
        "📈 My Progress",
        "👣 Stepping Target",
    ])

    with tab_live:
        st.markdown('<div class="patient-section">Current Session</div>', unsafe_allow_html=True)

        top_col1, top_col2, top_col3 = st.columns(3)

        spd_cls = status_class(stats['avg_speed'], 1.5, 4.0)
        with top_col1:
            st.markdown(f'''
                <div class="patient-medium-metric {spd_cls}">
                    <div class="patient-label">Walking Speed</div>
                    <div>{stats["avg_speed"]:.1f}</div>
                    <div class="patient-unit">mph</div>
                </div>''', unsafe_allow_html=True)

        with top_col2:
            st.markdown(f'''
                <div class="patient-medium-metric status-gray">
                    <div class="patient-label">Time Elapsed</div>
                    <div>{session_time:.1f}</div>
                    <div class="patient-unit">minutes</div>
                </div>''', unsafe_allow_html=True)

        cad_cls = status_class(stats['avg_right_cadence'], target_cadence_min, target_cadence_max)
        with top_col3:
            st.markdown(f'''
                <div class="patient-medium-metric {cad_cls}">
                    <div class="patient-label">Steps per Minute</div>
                    <div>{stats["avg_right_cadence"]:.0f}</div>
                    <div class="patient-unit">steps/min</div>
                </div>''', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="patient-section">Your Key Metrics</div>', unsafe_allow_html=True)
        km1, km2, km3 = st.columns(3)

        sl_cls = status_class(stats['avg_right_step_length'], target_step_length_min, target_step_length_max)
        with km1:
            st.markdown(f'''
                <div class="patient-large-metric {sl_cls}">
                    <div class="patient-label">Step Length</div>
                    <div>{stats["avg_right_step_length"]:.2f}</div>
                    <div class="patient-unit">meters per step</div>
                </div>''', unsafe_allow_html=True)

        asym_pct = (100 - min(stats['temporal_asymmetry'], 100))
        asym_cls = "status-green" if stats['temporal_asymmetry'] < 10 else ("status-yellow" if stats['temporal_asymmetry'] < 20 else "status-red")
        if not color_coding_on:
            asym_cls = "status-gray"
        with km2:
            st.markdown(f'''
                <div class="patient-large-metric {asym_cls}">
                    <div class="patient-label">Left–Right Balance</div>
                    <div>{asym_pct:.0f}%</div>
                    <div class="patient-unit">symmetry</div>
                </div>''', unsafe_allow_html=True)

        with km3:
            st.markdown(f'''
                <div class="patient-large-metric status-gray">
                    <div class="patient-label">Distance So Far</div>
                    <div>{total_distance:.0f}</div>
                    <div class="patient-unit">meters</div>
                </div>''', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="patient-section">Left vs. Right Balance</div>', unsafe_allow_html=True)
        sb1, sb2 = st.columns(2)
        with sb1:
            fig_s1 = create_real_time_symmetry_bars(
                stats['avg_left_cadence'], stats['avg_right_cadence'],
                "Steps per Minute",
                target_cadence_min, target_cadence_max,
                "Left Side", "Right Side")
            st.plotly_chart(fig_s1, use_container_width=True)
        with sb2:
            fig_s2 = create_real_time_symmetry_bars(
                stats['avg_left_step_length'], stats['avg_right_step_length'],
                "Step Length (m)",
                target_step_length_min, target_step_length_max,
                "Left Side", "Right Side")
            st.plotly_chart(fig_s2, use_container_width=True)

    with tab_done:
        st.markdown('<div class="patient-section">You Did It!</div>', unsafe_allow_html=True)
        st.markdown(create_hero_summary(stats, session_time, total_distance, is_patient=True),
                    unsafe_allow_html=True)

        st.markdown("---")
        filtered_df = df.copy()
        filtered_df['Sample Index'] = range(len(filtered_df))
        filtered_df['Temporal_Asymmetry'] = filtered_df.apply(
            lambda row: calculate_asymmetry_percentage(row['Left Step Time (sec)'], row['Right Step Time (sec)']), axis=1)
        filtered_df['Spatial_Asymmetry'] = filtered_df.apply(
            lambda row: calculate_asymmetry_percentage(row['Left Step Length (meters)'], row['Right Step Length (meters)']), axis=1)

        pt1, pt2, pt3, pt4 = st.tabs([
            "Steps per Minute", "Left–Right Balance", "How You Paced Yourself", "How You Did vs. Last Time"
        ])

        with pt1:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], y=filtered_df['Right Cadence (steps/min)'],
                mode='lines', name='Steps per Minute', line=dict(width=3, color='#d32f2f')))
            fig_c.add_hrect(y0=target_cadence_min, y1=target_cadence_max,
                            fillcolor="green", opacity=0.1, layer="below", line_width=0,
                            annotation_text="Target Zone", annotation_font_size=14)
            fig_c.update_layout(title="Steps per Minute During Session",
                                xaxis_title="Time", yaxis_title="Steps per Minute",
                                height=420, hovermode='x unified',
                                font=dict(size=15))
            st.plotly_chart(fig_c, use_container_width=True)

        with pt2:
            fig_sym = go.Figure()
            fig_sym.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], y=filtered_df['Temporal_Asymmetry'],
                mode='lines', name='Timing Balance', line=dict(color='#FF6B6B', width=2)))
            fig_sym.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], y=filtered_df['Spatial_Asymmetry'],
                mode='lines', name='Step Length Balance', line=dict(color='#4ECDC4', width=2)))
            fig_sym.add_hline(y=10, line_dash="dash", line_color="orange",
                              annotation_text="10% Difference Threshold",
                              annotation_font_size=13)
            fig_sym.update_layout(title="Left vs. Right Balance Over Time (lower = more balanced)",
                                  xaxis_title="Time", yaxis_title="Difference (%)",
                                  height=420, hovermode='x unified', font=dict(size=15))
            st.plotly_chart(fig_sym, use_container_width=True)

        with pt3:
            st.subheader("Start vs. Finish Comparison")
            first_third = df.iloc[:len(df) // 3]
            last_third = df.iloc[-len(df) // 3:]
            start_cad = first_third['Right Cadence (steps/min)'].mean()
            end_cad = last_third['Right Cadence (steps/min)'].mean()
            change = end_cad - start_cad
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Steps/min at Start", f"{start_cad:.0f}")
            fc2.metric("Steps/min at End", f"{end_cad:.0f}")
            fc3.metric("Change", f"{change:+.1f}", delta_color="normal")
            fig_fat = go.Figure()
            fig_fat.add_trace(go.Box(y=first_third['Right Cadence (steps/min)'],
                                     name="Beginning", marker_color="#4ECDC4"))
            fig_fat.add_trace(go.Box(y=last_third['Right Cadence (steps/min)'],
                                     name="End", marker_color="#FF6B6B"))
            fig_fat.update_layout(title="Your Pace: Beginning vs. End",
                                  yaxis_title="Steps per Minute", height=380, font=dict(size=15))
            st.plotly_chart(fig_fat, use_container_width=True)

        with pt4:
            if len(st.session_state.session_history) > 1:
                comp_df = pd.DataFrame([
                    {'Session': i + 1, 'Date': s['date'],
                     'Speed (mph)': s['avg_speed'], 'Steps/min': s['avg_cadence'],
                     'Distance (m)': s['distance']}
                    for i, s in enumerate(st.session_state.session_history)
                ])
                st.dataframe(comp_df, use_container_width=True)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(
                    x=comp_df['Session'], y=comp_df['Steps/min'],
                    mode='lines+markers', name='Steps per Minute',
                    line=dict(width=3, color='#667eea'), marker=dict(size=10)))
                fig_cmp.update_layout(title="Steps per Minute Across Sessions",
                                      xaxis_title="Session #", yaxis_title="Steps per Minute",
                                      height=400, font=dict(size=15))
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.info("Complete more sessions to compare your progress over time.")

        st.markdown("---")
        session_notes = st.text_area("Any notes for today? (optional)", height=120)
        if st.button("💾  Save This Session"):
            session_data = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'patient_id': st.session_state.get('patient_id', 'USER'),
                'duration': session_time,
                'distance': total_distance,
                'avg_speed': stats['avg_speed'],
                'avg_cadence': stats['avg_right_cadence'],
                'notes': session_notes,
                'stats': stats
            }
            st.session_state.session_history.append(session_data)
            st.success("✅ Session saved!")

    with tab_history:
        st.markdown('<div class="patient-section">My Progress Over Time</div>', unsafe_allow_html=True)

        if st.session_state.session_history:
            history_df = pd.DataFrame([
                {
                    'Date': s['date'],
                    'Time (min)': f"{s['duration']:.1f}",
                    'Distance (m)': f"{s['distance']:.0f}",
                    'Speed (mph)': f"{s['avg_speed']:.1f}",
                    'Steps/min': f"{s['avg_cadence']:.0f}",
                    'Notes': s['notes'][:50] + '…' if len(s['notes']) > 50 else s['notes']
                }
                for s in st.session_state.session_history
            ])
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("My Trends")
            hist_metric = st.selectbox("Show trend for:", ["Walking Speed", "Steps per Minute", "Distance", "Time"])
            if len(st.session_state.session_history) > 1:
                dates = [s['date'] for s in st.session_state.session_history]
                val_map = {
                    "Walking Speed": ([s['avg_speed'] for s in st.session_state.session_history], "Speed (mph)"),
                    "Steps per Minute": ([s['avg_cadence'] for s in st.session_state.session_history], "Steps/min"),
                    "Distance": ([s['distance'] for s in st.session_state.session_history], "Distance (m)"),
                    "Time": ([s['duration'] for s in st.session_state.session_history], "Minutes"),
                }
                vals, ylabel = val_map[hist_metric]
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(x=dates, y=vals, mode='lines+markers',
                                            line=dict(width=3, color='#667eea'), marker=dict(size=10)))
                fig_tr.update_layout(title=f"{hist_metric} Over Time",
                                     xaxis_title="Session Date", yaxis_title=ylabel,
                                     height=400, font=dict(size=15))
                st.plotly_chart(fig_tr, use_container_width=True)

            st.markdown("---")
            st.subheader("🏆 My Personal Bests")
            pr1, pr2, pr3 = st.columns(3)
            pr1.metric("Fastest Speed", f"{max(s['avg_speed'] for s in st.session_state.session_history):.1f} mph")
            pr2.metric("Longest Walk", f"{max(s['distance'] for s in st.session_state.session_history):.0f} m")
            pr3.metric("Longest Session", f"{max(s['duration'] for s in st.session_state.session_history):.1f} min")

            st.markdown("---")
            report_type = st.selectbox("Report Type", ["Daily", "Weekly", "Monthly"])
            if st.button(f"📄  Get My {report_type} Summary (PDF)"):
                st.success(f"✅ Your {report_type} summary has been sent to {st.session_state.user_email}")
        else:
            st.info("No sessions saved yet. Complete a session and tap **Save This Session** to start tracking your progress.")

    with tab_target:
        st.markdown('<div class="patient-section">Stepping Target</div>', unsafe_allow_html=True)
        st.markdown("Try to match your steps to the dots below. Your therapist has set a target step length for you.")

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            current_step = st.slider("Step progress (demo)", 0, 9, 0, key="patient_step_slider")
            fig_oh = create_overhead_foot_tracking_visualization(
                current_step=current_step,
                num_steps=10,
                target_step_length=admin_target_step_length,
                is_patient=True
            )
            st.plotly_chart(fig_oh, use_container_width=True)

        with sc2:
            on_target_pct = np.random.randint(75, 95)
            score = np.random.randint(80, 100)

            st.markdown(f'''
                <div class="patient-score-box" style="margin-bottom:16px;">
                    <div class="patient-score-value">{on_target_pct}%</div>
                    <div class="patient-score-label">On Target</div>
                </div>
                <div class="patient-score-box">
                    <div class="patient-score-value">{score}</div>
                    <div class="patient-score-label">Your Score</div>
                </div>''', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("🔴 **Right foot**")
            st.markdown("🔵 **Left foot**")
            st.markdown("Bigger dot = current step")

        st.markdown("---")
        st.subheader("Your Stepping Score Over This Session")
        steps_range = list(range(10))
        accuracy_scores = [np.random.randint(70, 100) for _ in range(10)]
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=steps_range, y=accuracy_scores,
            mode='lines+markers', name='Accuracy',
            line=dict(color='#28a745', width=3), marker=dict(size=12)))
        fig_sp.add_hline(y=90, line_dash="dash", line_color="green",
                         annotation_text="Goal: 90%", annotation_font_size=14,
                         annotation_font_color="#111111")
        fig_sp.update_layout(
            title=dict(text="Stepping Accuracy per Step", font=dict(size=18, color="#222222")),
            xaxis=dict(title=dict(text="Step Number", font=dict(size=14, color="#222222")),
                       tickfont=dict(size=13, color="#333333")),
            yaxis=dict(title=dict(text="Accuracy (%)", font=dict(size=14, color="#222222")),
                       tickfont=dict(size=13, color="#333333"), range=[0, 100]),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=14, color="#222222"),
        )
        st.plotly_chart(fig_sp, use_container_width=True)


def render_admin_ui(df, stats, session_time, total_distance,
                    target_cadence_min, target_cadence_max,
                    target_step_length_min, target_step_length_max,
                    patient_id):
    filtered_df = df.copy()
    filtered_df['Sample Index'] = range(len(filtered_df))
    filtered_df['Temporal_Asymmetry'] = filtered_df.apply(
        lambda row: calculate_asymmetry_percentage(row['Left Step Time (sec)'], row['Right Step Time (sec)']), axis=1)
    filtered_df['Spatial_Asymmetry'] = filtered_df.apply(
        lambda row: calculate_asymmetry_percentage(row['Left Step Length (meters)'], row['Right Step Length (meters)']), axis=1)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Real-Time Display", "Hero Summary", "Post-Analysis", "Progress Tracking", "Stepping Target"
    ])

    with tab1:
        st.markdown('<h2 style="color:#d32f2f;">Live Session Metrics</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 1])

        cadence_value = stats['avg_right_cadence']
        if 'cadence' in st.session_state.target_zones:
            tmin, tmax = st.session_state.target_zones['cadence']
            if tmin <= cadence_value <= tmax:
                sc = "status-green"
            elif (tmin * 0.9 <= cadence_value < tmin) or (tmax < cadence_value <= tmax * 1.1):
                sc = "status-yellow"
            else:
                sc = "status-red"
        else:
            sc = "status-gray"

        with col1:
            st.markdown(f'''
                <div class="large-metric {sc}">
                    <div style="font-size:0.4em;opacity:0.8;">PRIMARY METRIC</div>
                    <div>{cadence_value:.0f}</div>
                    <div style="font-size:0.5em;">steps/min</div>
                    <div style="font-size:0.4em;opacity:0.8;">Cadence</div>
                </div>''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
                <div class="medium-metric status-green">
                    <div style="font-size:0.5em;opacity:0.8;">SPEED</div>
                    <div>{stats["avg_speed"]:.1f}</div>
                    <div style="font-size:0.6em;">mph</div>
                </div>''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
                <div class="medium-metric status-green">
                    <div style="font-size:0.5em;opacity:0.8;">STEP LENGTH</div>
                    <div>{stats["avg_right_step_length"]:.2f}</div>
                    <div style="font-size:0.6em;">meters</div>
                </div>''', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Real-Time Symmetry Analysis")
        sym_col1, sym_col2 = st.columns(2)
        with sym_col1:
            fig_sym_cadence = create_real_time_symmetry_bars(
                stats['avg_left_cadence'], stats['avg_right_cadence'], "Cadence (steps/min)",
                target_cadence_min if 'cadence' in st.session_state.target_zones else None,
                target_cadence_max if 'cadence' in st.session_state.target_zones else None)
            st.plotly_chart(fig_sym_cadence, use_container_width=True)
        with sym_col2:
            fig_sym_step = create_real_time_symmetry_bars(
                stats['avg_left_step_length'], stats['avg_right_step_length'], "Step Length (meters)",
                target_step_length_min if 'step_length' in st.session_state.target_zones else None,
                target_step_length_max if 'step_length' in st.session_state.target_zones else None)
            st.plotly_chart(fig_sym_step, use_container_width=True)

        st.markdown("---")
        st.subheader("Asymmetry Indicators")
        ac1, ac2 = st.columns(2)
        with ac1:
            st.metric("Temporal Asymmetry", f"{stats['temporal_asymmetry']:.1f}%",
                      delta="Good" if stats['temporal_asymmetry'] < 10 else "Needs Attention",
                      delta_color="inverse" if stats['temporal_asymmetry'] < 10 else "normal")
        with ac2:
            st.metric("Spatial Asymmetry", f"{stats['spatial_asymmetry']:.1f}%",
                      delta="Good" if stats['spatial_asymmetry'] < 10 else "Needs Attention",
                      delta_color="inverse" if stats['spatial_asymmetry'] < 10 else "normal")

    with tab2:
        st.markdown('<h2 style="color:#d32f2f;">Session Complete!</h2>', unsafe_allow_html=True)
        st.markdown(create_hero_summary(stats, session_time, total_distance), unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Session Notes")
        session_notes = st.text_area("Add notes about this session:", height=150)
        if st.button("Save Session"):
            session_data = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'patient_id': patient_id,
                'duration': session_time,
                'distance': total_distance,
                'avg_speed': stats['avg_speed'],
                'avg_cadence': stats['avg_right_cadence'],
                'notes': session_notes,
                'stats': stats
            }
            st.session_state.session_history.append(session_data)
            st.success("✅ Session saved successfully!")

    with tab3:
        st.markdown('<h2 style="color:#d32f2f;">Post-Analysis</h2>', unsafe_allow_html=True)
        analysis_tabs = st.tabs(["Metric Trends", "Fatigue Analysis", "Comparison View", "Symmetry Score"])

        with analysis_tabs[0]:
            st.subheader("Metrics Over Time")
            metric_choice = st.selectbox("Select Metric to Analyze:", [
                "Right Cadence (steps/min)", "Left Cadence (steps/min)",
                "Right Step Length (meters)", "Left Step Length (meters)", "GaitSpeed Rtable (mph*10)"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['Sample Index'], y=filtered_df[metric_choice],
                                     mode='lines+markers', name=metric_choice, line=dict(width=3)))
            if st.session_state.self_pacing_enabled:
                fig.add_vline(x=len(filtered_df) // 2, line_dash="dash", line_color="red",
                              annotation_text="Self-Pacing Enabled", annotation_position="top")
            fig.update_layout(title=f"{metric_choice} Over Session",
                              xaxis_title="Time (samples)", yaxis_title=metric_choice,
                              height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        with analysis_tabs[1]:
            st.subheader("Fatigue Analysis")
            fatigue_metric = st.selectbox("Select Metric for Fatigue Analysis:", [
                "Right Cadence (steps/min)", "Left Cadence (steps/min)",
                "Right Step Length (meters)", "Left Step Length (meters)"], key="fatigue_metric")
            fig_fat = create_fatigue_analysis(filtered_df, fatigue_metric)
            st.plotly_chart(fig_fat, use_container_width=True)

        with analysis_tabs[2]:
            st.subheader("Session Comparison")
            if len(st.session_state.session_history) > 1:
                comp_df = pd.DataFrame([
                    {'Session': i + 1, 'Date': s['date'], 'Avg Speed': s['avg_speed'],
                     'Avg Cadence': s['avg_cadence'], 'Distance': s['distance']}
                    for i, s in enumerate(st.session_state.session_history)])
                st.dataframe(comp_df, use_container_width=True)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=comp_df['Session'], y=comp_df['Avg Cadence'],
                                             mode='lines+markers', name='Average Cadence', line=dict(width=3)))
                fig_cmp.update_layout(title="Cadence Progress Over Sessions",
                                      xaxis_title="Session Number", yaxis_title="Average Cadence (steps/min)",
                                      height=400)
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.info("Complete more sessions to see comparison data.")

        with analysis_tabs[3]:
            st.subheader("Symmetry Score Over Time")
            fig_sym2 = go.Figure()
            fig_sym2.add_trace(go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Temporal_Asymmetry'],
                                          mode='lines', name='Temporal Asymmetry', line=dict(color='#FF6B6B', width=2)))
            fig_sym2.add_trace(go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Spatial_Asymmetry'],
                                          mode='lines', name='Spatial Asymmetry', line=dict(color='#4ECDC4', width=2)))
            fig_sym2.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="10% Threshold")
            fig_sym2.update_layout(title="Asymmetry Percentage Over Time",
                                   xaxis_title="Time (samples)", yaxis_title="Asymmetry (%)",
                                   height=500, hovermode='x unified')
            st.plotly_chart(fig_sym2, use_container_width=True)

    with tab4:
        st.markdown('<h2 style="color:#d32f2f;">Progress Tracking</h2>', unsafe_allow_html=True)
        if st.session_state.session_history:
            st.subheader("Patient History")
            history_df = pd.DataFrame([
                {'Date': s['date'], 'Duration (min)': f"{s['duration']:.1f}",
                 'Distance (m)': f"{s['distance']:.1f}", 'Avg Speed (mph)': f"{s['avg_speed']:.2f}",
                 'Avg Cadence': f"{s['avg_cadence']:.0f}",
                 'Notes': s['notes'][:50] + '…' if len(s['notes']) > 50 else s['notes']}
                for s in st.session_state.session_history])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            st.markdown("---")
            st.subheader("Historical Trends")
            trend_metric = st.selectbox("Select Metric to Track:", ["Avg Speed", "Avg Cadence", "Distance", "Duration"])
            if len(st.session_state.session_history) > 1:
                dates = [s['date'] for s in st.session_state.session_history]
                val_map = {
                    "Avg Speed": ([s['avg_speed'] for s in st.session_state.session_history], "Speed (mph)"),
                    "Avg Cadence": ([s['avg_cadence'] for s in st.session_state.session_history], "Cadence (steps/min)"),
                    "Distance": ([s['distance'] for s in st.session_state.session_history], "Distance (meters)"),
                    "Duration": ([s['duration'] for s in st.session_state.session_history], "Duration (minutes)"),
                }
                vals, ylabel = val_map[trend_metric]
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=dates, y=vals, mode='lines+markers', name=trend_metric,
                                               line=dict(width=3, color='#667eea'), marker=dict(size=10)))
                fig_trend.update_layout(title=f"{trend_metric} Over Time",
                                        xaxis_title="Session Date", yaxis_title=ylabel, height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("---")
            st.subheader("Personal Records")
            pr1, pr2, pr3 = st.columns(3)
            pr1.metric("Fastest Speed 🏃", f"{max(s['avg_speed'] for s in st.session_state.session_history):.2f} mph")
            pr2.metric("Longest Distance 📏", f"{max(s['distance'] for s in st.session_state.session_history):.0f} m")
            pr3.metric("Longest Duration ⏱️", f"{max(s['duration'] for s in st.session_state.session_history):.1f} min")
            st.markdown("---")
            report_type = st.selectbox("Select Report Type:", ["Daily", "Weekly", "Monthly"])
            if st.button("Generate & Email PDF Report"):
                st.success(f"✅ {report_type} report sent to {st.session_state.user_email}")
        else:
            st.info("No session history yet. Complete a session to start tracking progress.")

    with tab5:
        st.markdown('<h2 style="color:#d32f2f;">Stepping Target</h2>', unsafe_allow_html=True)
        track_col1, track_col2 = st.columns([3, 1])
        with track_col1:
            target_step_length = st.slider(
                "Target Step Length (meters)", min_value=0.5, max_value=1.0,
                value=st.session_state.admin_target_step_length, step=0.05,
                help="Administrator sets the target step length for patient to match")
            st.session_state.admin_target_step_length = target_step_length

            current_step = st.slider("Current Step", min_value=0, max_value=9, value=0,
                                     help="Move slider to simulate stepping progression",
                                     key="admin_step_slider")
            fig_overhead = create_overhead_foot_tracking_visualization(
                current_step=current_step, num_steps=10,
                target_step_length=target_step_length, is_patient=False)
            st.plotly_chart(fig_overhead, use_container_width=True)

        with track_col2:
            st.subheader("Stepping Stats")
            st.metric("Current Step", current_step)
            st.metric("Total Steps", 10)
            on_target_percentage = np.random.randint(75, 95)
            st.metric("On Target", f"{on_target_percentage}%")
            stepping_score = np.random.randint(80, 100)
            st.metric("Score", stepping_score)
            st.markdown("---")
            st.subheader("Legend")
            st.markdown("🔴 Right Foot")
            st.markdown("🔵 Left Foot")
            st.markdown("Larger dot = active step")

        st.markdown("---")
        st.subheader("Stepping Target Data")
        stepping_data = generate_fake_foot_tracking_data(10)
        stepping_data['On Target'] = np.random.choice(['Yes', 'No'], size=10, p=[0.8, 0.2])
        stepping_data['Deviation (cm)'] = np.random.uniform(0, 5, size=10).round(2)
        st.dataframe(stepping_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Stepping Target Performance Over Time")
        steps_range = list(range(10))
        accuracy_scores = [np.random.randint(70, 100) for _ in range(10)]
        fig_target_perf = go.Figure()
        fig_target_perf.add_trace(go.Scatter(
            x=steps_range, y=accuracy_scores, mode='lines+markers', name='Accuracy Score',
            line=dict(color='#28a745', width=3), marker=dict(size=10)))
        fig_target_perf.add_hline(
            y=90, line_dash="dash", line_color="green",
            annotation_text="Target: 90%",
            annotation_font_size=14,
            annotation_font_color="#111111")
        fig_target_perf.update_layout(
            title=dict(text="Step Accuracy Score Over Session", font=dict(size=18, color="#222222")),
            xaxis=dict(title=dict(text="Step Number", font=dict(size=14, color="#222222")),
                       tickfont=dict(size=13, color="#333333")),
            yaxis=dict(title=dict(text="Accuracy Score (%)", font=dict(size=14, color="#222222")),
                       tickfont=dict(size=13, color="#333333"), range=[0, 100]),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_target_perf, use_container_width=True)

    st.markdown("---")
    st.markdown('<span style="color:#d32f2f; font-size:32px;">Export Options</span>', unsafe_allow_html=True)
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        if st.button("Export Statistics Summary"):
            summary_stats = pd.DataFrame({
                'Parameter': [
                    'Average Gait Speed (mph)', 'Right Cadence (steps/min)', 'Left Cadence (steps/min)',
                    'Right Step Length (m)', 'Left Step Length (m)', 'Right Stride Length (m)',
                    'Left Stride Length (m)', 'Right Swing/Stance Ratio', 'Left Swing/Stance Ratio',
                    'Temporal Asymmetry (%)', 'Spatial Asymmetry (%)'],
                'Value': [
                    f"{stats['avg_speed']:.2f}", f"{stats['avg_right_cadence']:.1f}",
                    f"{stats['avg_left_cadence']:.1f}", f"{stats['avg_right_step_length']:.3f}",
                    f"{stats['avg_left_step_length']:.3f}", f"{stats['avg_right_stride_length']:.3f}",
                    f"{stats['avg_left_stride_length']:.3f}", f"{stats['right_swing_stance_ratio']:.2f}",
                    f"{stats['left_swing_stance_ratio']:.2f}", f"{stats['temporal_asymmetry']:.1f}",
                    f"{stats['spatial_asymmetry']:.1f}"]})
            csv_stats = summary_stats.to_csv(index=False)
            st.download_button(label="Download Statistics CSV", data=csv_stats,
                               file_name=f"patient_{patient_id}_gait_stats.csv", mime="text/csv")
            st.dataframe(summary_stats, hide_index=True)

    with export_col2:
        def export_full_report_to_pdf(s, pid):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=16)
            pdf.cell(0, 10, "Patient Gait Analysis Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Patient ID: {pid}", ln=True)
            pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(0, 10, "Key Statistics", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", size=12)
            for line in [
                f"- Average Gait Speed: {s['avg_speed']:.2f} mph",
                f"- Right Cadence: {s['avg_right_cadence']:.1f} steps/min",
                f"- Left Cadence: {s['avg_left_cadence']:.1f} steps/min",
                f"- Right Step Length: {s['avg_right_step_length']:.3f} m",
                f"- Left Step Length: {s['avg_left_step_length']:.3f} m",
                f"- Temporal Asymmetry: {s['temporal_asymmetry']:.1f}%",
                f"- Spatial Asymmetry: {s['spatial_asymmetry']:.1f}%",
            ]:
                pdf.cell(0, 10, line, ln=True)
            pdf_buffer = BytesIO()
            pdf_buffer.write(pdf.output(dest='S').encode('latin1'))
            pdf_buffer.seek(0)
            return pdf_buffer

        if st.button("Export Full Report"):
            pdf_buffer = export_full_report_to_pdf(stats, patient_id)
            st.download_button(label="Download Full Report as PDF", data=pdf_buffer,
                               file_name=f"patient_{patient_id}_gait_report.pdf", mime="application/pdf")

    st.markdown('<span style="color:#d32f2f; font-size:32px;">Raw Data</span>', unsafe_allow_html=True)
    if st.checkbox("Show Raw Data Table"):
        st.dataframe(filtered_df)
        if st.button("Export Raw Data"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(label="Download Raw Data CSV", data=csv_data,
                               file_name=f"patient_{patient_id}_raw_gait_data.csv", mime="text/csv")

def main():
    initialize_session_state()
    st.markdown('<div class="main-header">Gait Analysis System</div>', unsafe_allow_html=True)
    st.sidebar.title("System Dashboard")
    
    if not st.session_state.logged_in:
        st.sidebar.subheader("Login")
        user_mode = st.sidebar.radio("Select Mode:", ["Administrator", "Patient"])
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            st.session_state.logged_in = True
            st.session_state.user_mode = user_mode
            st.session_state.user_email = email
            st.rerun()
        st.info("👈 Please login to continue")
        return

    st.sidebar.success(f"Logged in as: {st.session_state.user_mode}")
    st.sidebar.text(f"Email: {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    is_admin = st.session_state.user_mode == "Administrator"

    if is_admin:
        st.sidebar.subheader("Patient Selection")
        patient_id = st.sidebar.text_input("Patient ID (Deidentified)",
                                            value="PT-" + str(np.random.randint(1000, 9999)))
        with st.sidebar.expander("Patient Profile"):
            st.text_input("Sex", key="patient_sex")
            st.text_input("Month/Year", key="patient_dob")
            st.number_input("Height (cm)", key="patient_height")
            st.number_input("Weight (kg)", key="patient_weight")
            st.text_input("Pathology (if applicable)", key="patient_pathology")
            st.selectbox("Side Affected", ["None", "Left", "Right", "Bilateral"], key="patient_side")
    else:
        patient_id = "USER-" + str(np.random.randint(1000, 9999))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Configuration")
    self_pacing = st.sidebar.checkbox("Enable Self-Pacing", value=st.session_state.self_pacing_enabled)
    st.session_state.self_pacing_enabled = self_pacing

    if is_admin:
        max_speed = st.sidebar.number_input("Max Speed Limit (mph)", min_value=0.0, max_value=15.0,
                                             value=8.0, step=0.5)
        st.sidebar.subheader("Metric Selection")
        available_metrics = ["Cadence", "Step Length", "Stride Length",
                             "Step Time", "Gait Speed", "Swing Time", "Stance Time", "Symmetry"]
        primary_metric = st.sidebar.selectbox("Primary Metric (Large Display)", available_metrics)
        secondary_metrics = st.sidebar.multiselect("Secondary Metrics (Up to 3)", available_metrics,
                                                    max_selections=3)
        if st.sidebar.button("Save as Favorite"):
            st.session_state.favorite_metrics = [primary_metric] + secondary_metrics
            st.sidebar.success("Saved to favorites!")
        if st.session_state.favorite_metrics:
            if st.sidebar.button("Load Favorite Metrics"):
                st.sidebar.info(f"Loaded: {', '.join(st.session_state.favorite_metrics)}")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Target Zones")
        with st.sidebar.expander("Set Target Zones"):
            target_cadence_min = st.number_input("Min Cadence (steps/min)", value=100.0)
            target_cadence_max = st.number_input("Max Cadence (steps/min)", value=120.0)
            st.session_state.target_zones['cadence'] = (target_cadence_min, target_cadence_max)
            target_step_length_min = st.number_input("Min Step Length (m)", value=0.6)
            target_step_length_max = st.number_input("Max Step Length (m)", value=0.8)
            st.session_state.target_zones['step_length'] = (target_step_length_min, target_step_length_max)

        st.sidebar.subheader("Display Settings")
        admin_color_coding = st.sidebar.checkbox("Enable Color Coding", value=st.session_state.admin_color_coding)
        st.session_state.admin_color_coding = admin_color_coding

        st.sidebar.subheader("Stepping Target")
        admin_step_length = st.sidebar.number_input(
            "Patient Step Length Target (m)",
            min_value=0.3, max_value=1.2,
            value=st.session_state.admin_target_step_length, step=0.05)
        st.session_state.admin_target_step_length = admin_step_length

    else:
        target_cadence_min, target_cadence_max = 100.0, 120.0
        target_step_length_min, target_step_length_max = 0.6, 0.8
        if 'cadence' in st.session_state.target_zones:
            target_cadence_min, target_cadence_max = st.session_state.target_zones['cadence']
        if 'step_length' in st.session_state.target_zones:
            target_step_length_min, target_step_length_max = st.session_state.target_zones['step_length']

        st.sidebar.markdown("---")

    with st.sidebar.expander("Help & Information"):
        if is_admin:
            st.write("""
            **Real-time Display Options:**
            - Live Feed: Single selected metric
            - Key Metrics Dashboard: 3-5 primary metrics
            - Trend Arrows: Improvement indicators
            - Stepping Target: Overhead footstep graphic (dots)
            - Symmetry: Real-time bar graphs
            - Target Zones: Visual optimal ranges

            **Color Coding (admin-configurable):**
            - 🟢 Green: Optimal
            - 🟡 Yellow: Borderline
            - 🔴 Red: Needs correction
            - ⚫ Gray: Inactive/N/A
            """)
        else:
            st.write("""
            **During your session:**
            - Watch the large numbers on screen
            - Green = great, Yellow = keep working, Red = adjust
            - Match your steps to the dots in Stepping Target

            **After your session:**
            - See how you did in Session Summary
            - Check My Progress to track improvement
            """)

    df = None
    error_message = None
    uploaded_file = st.sidebar.file_uploader("Upload Session Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df, error_message = process_csv_file(uploaded_file)
        if error_message:
            st.error(error_message)

    if df is not None:
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = patient_id

        if is_admin:
            time_range = st.sidebar.slider("Session Time Range", 0, len(df) - 1, (0, len(df) - 1))
            filtered_df = df.iloc[time_range[0]:time_range[1] + 1].copy()
        else:
            filtered_df = df.copy()

        stats = calculate_statistics(filtered_df)
        session_time = len(filtered_df) * 0.1 / 60
        total_distance = stats['avg_speed'] * session_time * 1609.34

        if not is_admin:
            target_cadence_min = st.session_state.target_zones.get('cadence', (100.0, 120.0))[0]
            target_cadence_max = st.session_state.target_zones.get('cadence', (100.0, 120.0))[1]
            target_step_length_min = st.session_state.target_zones.get('step_length', (0.6, 0.8))[0]
            target_step_length_max = st.session_state.target_zones.get('step_length', (0.6, 0.8))[1]

        if is_admin:
            render_admin_ui(
                filtered_df, stats, session_time, total_distance,
                target_cadence_min, target_cadence_max,
                target_step_length_min, target_step_length_max,
                patient_id)
        else:
            render_patient_ui(
                filtered_df, stats, session_time, total_distance,
                target_cadence_min, target_cadence_max,
                target_step_length_min, target_step_length_max,
                color_coding_on=st.session_state.admin_color_coding,
                admin_target_step_length=st.session_state.admin_target_step_length)
    else:
        if not error_message:
            st.info("👈 Upload a CSV file to begin your session.")


if __name__ == "__main__":
    main()
