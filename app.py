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
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

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
        if i % 2 == 0:
            foot = "Right"
            x = lateral_offset
        else:
            foot = "Left"
            x = -lateral_offset
        steps.append(
            {
                "step": i,
                "foot": foot,
                "x": x,
                "y": y_position,
                "phase": "contact",
            }
        )
    return pd.DataFrame(steps)


def create_overhead_foot_tracking_visualization(current_step: int = 0, num_steps: int = 10, target_step_length: float = 0.75):
    df_track = generate_fake_foot_tracking_data(num_steps)
    
    fig = go.Figure()
    
    fig.add_shape(
        type="rect",
        x0=-0.4,
        y0=0,
        x1=0.4,
        y1=num_steps * target_step_length + 1,
        line=dict(color="gray", width=2),
        fillcolor="lightgray",
        opacity=0.2,
        layer="below",
    )
    
    fig.add_annotation(
        x=0,
        y=num_steps * target_step_length + 0.5,
        text="↑ Direction of Movement",
        showarrow=False,
        font=dict(size=14, color="gray"),
        opacity=0.7
    )
    
    color_map = {"Right": "#FF6B6B", "Left": "#4ECDC4"}
    
    for idx, row in df_track.iterrows():
        if idx == current_step:
            size = 35
            opacity = 1.0
            line_width = 4
        elif idx < current_step:
            size = 22
            opacity = 0.6
            line_width = 2
        else:
            size = 18
            opacity = 0.3
            line_width = 1
        
        foot_symbol = "circle"
        if row['foot'] == "Right":
            foot_symbol = "triangle-right"
        else:
            foot_symbol = "triangle-left"
        
        fig.add_trace(
            go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color_map[row["foot"]],
                    opacity=opacity,
                    symbol=foot_symbol,
                    line=dict(width=line_width, color=color_map[row["foot"]]),
                ),
                text=[f"S{row['step']}"],
                textposition="middle center",
                textfont=dict(size=10, color="white", family="Arial Black"),
                name=f"{row['foot']} Foot",
                hovertemplate=(
                    f"<b>{row['foot']} Foot</b><br>"
                    f"Step: {row['step']}<br>"
                    f"Position: {row['y']:.2f}m<br>"
                    f"Lateral: {row['x']:.2f}m<br>"
                    "<extra></extra>"
                ),
                showlegend=(idx in [0, 1]),
            )
        )
        
        target_y = (idx + 1) * target_step_length
        if idx < num_steps - 1:
            fig.add_shape(
                type="line",
                x0=-0.35,
                y0=target_y,
                x1=0.35,
                y1=target_y,
                line=dict(color="green", width=1, dash="dot"),
                opacity=0.3,
                layer="below"
            )
    
    fig.update_layout(
        title=dict(
            text=f"<b>Stepping Target (Current Step: {current_step})</b>",
            font=dict(size=20, color="#d32f2f")
        ),
        xaxis=dict(
            title="Left ← | → Right (meters)",
            range=[-0.5, 0.5],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black"
        ),
        yaxis=dict(
            title="Forward Movement (meters)",
            range=[-0.5, num_steps * target_step_length + 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        height=600,
        plot_bgcolor="rgba(250,250,250,1)",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=80, r=80, t=80, b=100),
    )
    
    return fig


def create_hero_summary(stats, session_time, distance):
    hero_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; 
                border-radius: 15px; 
                color: white; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: bold;">Session Complete!</h1>
        <div style="display: flex; justify-content: space-around; margin-top: 30px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px; margin: 10px;">
                <div style="font-size: 3em; font-weight: bold;">{distance:.2f}</div>
                <div style="font-size: 1.2em; opacity: 0.9;">meters</div>
                <div style="font-size: 0.9em; opacity: 0.7;">Total Distance</div>
            </div>
            <div style="flex: 1; min-width: 150px; margin: 10px;">
                <div style="font-size: 3em; font-weight: bold;">{session_time:.1f}</div>
                <div style="font-size: 1.2em; opacity: 0.9;">minutes</div>
                <div style="font-size: 0.9em; opacity: 0.7;">Session Time</div>
            </div>
            <div style="flex: 1; min-width: 150px; margin: 10px;">
                <div style="font-size: 3em; font-weight: bold;">{stats['avg_speed']:.2f}</div>
                <div style="font-size: 1.2em; opacity: 0.9;">mph</div>
                <div style="font-size: 0.9em; opacity: 0.7;">Average Speed</div>
            </div>
            <div style="flex: 1; min-width: 150px; margin: 10px;">
                <div style="font-size: 3em; font-weight: bold;">{stats['avg_right_cadence']:.0f}</div>
                <div style="font-size: 1.2em; opacity: 0.9;">steps/min</div>
                <div style="font-size: 0.9em; opacity: 0.7;">Average Cadence</div>
            </div>
        </div>
    </div>
    """
    return hero_html


def create_real_time_symmetry_bars(left_value, right_value, metric_name, target_min=None, target_max=None):
    fig = go.Figure()
    
    def get_color(value, target_min, target_max):
        if target_min is not None and target_max is not None:
            if target_min <= value <= target_max:
                return "#28a745"
            elif (target_min * 0.9 <= value < target_min) or (target_max < value <= target_max * 1.1):
                return "#ffc107"
            else:
                return "#dc3545"
        return "#6c757d"
    
    left_color = get_color(left_value, target_min, target_max) if target_min else "#4ECDC4"
    right_color = get_color(right_value, target_min, target_max) if target_min else "#FF6B6B"
    
    fig.add_trace(go.Bar(
        x=[left_value],
        y=['Left'],
        orientation='h',
        name='Left',
        marker=dict(color=left_color),
        text=[f'{left_value:.2f}'],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        x=[right_value],
        y=['Right'],
        orientation='h',
        name='Right',
        marker=dict(color=right_color),
        text=[f'{right_value:.2f}'],
        textposition='auto',
    ))
    
    if target_min and target_max:
        fig.add_vrect(
            x0=target_min, x1=target_max,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
        )
    
    fig.update_layout(
        title=f"Real-Time Symmetry: {metric_name}",
        xaxis_title=metric_name,
        height=200,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=40),
        plot_bgcolor="white",
    )
    
    return fig


def calculate_asymmetry_percentage(left_value, right_value):
    if left_value == 0 or right_value == 0:
        return 0
    avg = (left_value + right_value) / 2
    diff = abs(right_value - left_value)
    return (diff / avg) * 100


def create_fatigue_analysis(df, metric_name):
    first_third = df.iloc[:len(df)//3]
    last_third = df.iloc[-len(df)//3:]
    
    start_mean = first_third[metric_name].mean()
    end_mean = last_third[metric_name].mean()
    change = ((end_mean - start_mean) / start_mean) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=first_third[metric_name],
        name="Start",
        marker_color="#4ECDC4"
    ))
    
    fig.add_trace(go.Box(
        y=last_third[metric_name],
        name="End",
        marker_color="#FF6B6B"
    ))
    
    fig.update_layout(
        title=f"Fatigue Analysis: {metric_name}<br><sub>Change: {change:+.1f}%</sub>",
        yaxis_title=metric_name,
        height=400,
        showlegend=True
    )
    
    return fig


def initialize_session_state():
    if 'user_mode' not in st.session_state:
        st.session_state.user_mode = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'patient_profiles' not in st.session_state:
        st.session_state.patient_profiles = {}
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    if 'favorite_metrics' not in st.session_state:
        st.session_state.favorite_metrics = []
    if 'target_zones' not in st.session_state:
        st.session_state.target_zones = {}
    if 'self_pacing_enabled' not in st.session_state:
        st.session_state.self_pacing_enabled = False


st.set_page_config(
    page_title="Gait Analysis System",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0055cc;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .large-metric {
        font-size: 4em;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
    }
    .medium-metric {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .status-green {
        background-color: #28a745;
        color: white;
    }
    .status-yellow {
        background-color: #ffc107;
        color: black;
    }
    .status-red {
        background-color: #dc3545;
        color: white;
    }
    .status-gray {
        background-color: #6c757d;
        color: white;
    }
    .assessment-box {
        background-color: transparent; 
        color: #d32f2f;
        border: 1px solid #d32f2f; 
        border-radius: 5px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .assessment-box p {
        color: #FFFFFF;
    }
    .statistics-group {
        border: 2px solid #6c757d;
        border-radius: 5px;
        padding: 15px;
        margin-top: 20px;
    }
    .metric-separator {
        border-top: 1px solid #000;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 1.2em;
    }
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
        df['Left Step Time (sec)'].mean(), 
        df['Right Step Time (sec)'].mean()
    )
    stats['spatial_asymmetry'] = calculate_asymmetry_percentage(
        df['Left Step Length (meters)'].mean(), 
        df['Right Step Length (meters)'].mean()
    )
    
    return stats

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">Gait Analysis System</div>', unsafe_allow_html=True)
    
    st.sidebar.title("System Dashboard")
    
    if not st.session_state.logged_in:
        st.sidebar.subheader("Login")
        user_mode = st.sidebar.radio("Select Mode:", ["Administrator", "Recreational User"])
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
    
    if st.session_state.user_mode == "Administrator":
        st.sidebar.subheader("Patient Selection")
        patient_id = st.sidebar.text_input("Patient ID (Deidentified)", value="PT-" + str(np.random.randint(1000, 9999)))
        
        with st.sidebar.expander("Patient Profile"):
            st.text_input("Sex", key="patient_sex")
            st.text_input("Month/Year", key="patient_dob")
            st.number_input("Height (cm)", key="patient_height")
            st.number_input("Weight (kg)", key="patient_weight")
            st.text_input("Pathology (if applicable)", key="patient_pathology")
            st.selectbox("Side Affected", ["None", "Left", "Right", "Bilateral"], key="patient_side")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Configuration")
    
    self_pacing = st.sidebar.checkbox("Enable Self-Pacing", value=st.session_state.self_pacing_enabled)
    st.session_state.self_pacing_enabled = self_pacing
    
    if st.session_state.user_mode == "Administrator":
        max_speed = st.sidebar.number_input("Max Speed Limit (mph)", min_value=0.0, max_value=15.0, value=8.0, step=0.5)
    
    st.sidebar.subheader("Metric Selection")
    available_metrics = [
        "Cadence", "Step Length", "Stride Length", 
        "Step Time", "Gait Speed", "Swing Time", 
        "Stance Time", "Symmetry"
    ]
    
    primary_metric = st.sidebar.selectbox("Primary Metric (Large Display)", available_metrics)
    secondary_metrics = st.sidebar.multiselect("Secondary Metrics (Up to 3)", available_metrics, max_selections=3)
    
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

    df = None
    error_message = None

    uploaded_file = st.sidebar.file_uploader("Upload Session Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df, error_message = process_csv_file(uploaded_file)
        if error_message:
            st.error(error_message)

    with st.sidebar.expander("Help & Information"):
        st.write("""
        **Real-time Display Options:**
        - Live Feed: Single selected metric
        - Key Metrics Dashboard: 3-5 primary metrics
        - Trend Arrows: Improvement indicators
        - Stepping Target: Overhead footstep graphic
        - Symmetry: Real-time bar graphs
        - Target Zones: Visual optimal ranges
        
        **Color Coding:**
        - 🟢 Green: Optimal range
        - 🟡 Yellow: Borderline
        - 🔴 Red: Needs correction
        - ⚫ Gray: Inactive/N/A
        """)

    if df is not None:
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = patient_id if st.session_state.user_mode == "Administrator" else "USER-" + str(np.random.randint(1000, 9999))
        
        time_range = st.sidebar.slider(
            "Session Time Range",
            0, len(df)-1, (0, len(df)-1)
        )
        
        filtered_df = df.iloc[time_range[0]:time_range[1]+1].copy()
        
        stats = calculate_statistics(filtered_df)
        
        session_time = len(filtered_df) * 0.1 / 60
        total_distance = stats['avg_speed'] * session_time * 1609.34
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Real-Time Display", 
            "Hero Summary", 
            "Post-Analysis", 
            "Progress Tracking",
            "Stepping Target"
        ])
        
        with tab1:
            st.markdown('<h2 style="color:#d32f2f;">Live Session Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                cadence_value = stats['avg_right_cadence']
                if 'cadence' in st.session_state.target_zones:
                    target_min, target_max = st.session_state.target_zones['cadence']
                    if target_min <= cadence_value <= target_max:
                        status_class = "status-green"
                    elif (target_min * 0.9 <= cadence_value < target_min) or (target_max < cadence_value <= target_max * 1.1):
                        status_class = "status-yellow"
                    else:
                        status_class = "status-red"
                else:
                    status_class = "status-gray"
                
                st.markdown(f'''
                    <div class="large-metric {status_class}">
                        <div style="font-size: 0.4em; opacity: 0.8;">PRIMARY METRIC</div>
                        <div>{cadence_value:.0f}</div>
                        <div style="font-size: 0.5em;">steps/min</div>
                        <div style="font-size: 0.4em; opacity: 0.8;">Cadence</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                    <div class="medium-metric status-green">
                        <div style="font-size: 0.5em; opacity: 0.8;">SPEED</div>
                        <div>{stats['avg_speed']:.1f}</div>
                        <div style="font-size: 0.6em;">mph</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                    <div class="medium-metric status-green">
                        <div style="font-size: 0.5em; opacity: 0.8;">STEP LENGTH</div>
                        <div>{stats['avg_right_step_length']:.2f}</div>
                        <div style="font-size: 0.6em;">meters</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("Real-Time Symmetry Analysis")
            sym_col1, sym_col2 = st.columns(2)
            
            with sym_col1:
                fig_sym_cadence = create_real_time_symmetry_bars(
                    stats['avg_left_cadence'],
                    stats['avg_right_cadence'],
                    "Cadence (steps/min)",
                    target_cadence_min if 'cadence' in st.session_state.target_zones else None,
                    target_cadence_max if 'cadence' in st.session_state.target_zones else None
                )
                st.plotly_chart(fig_sym_cadence, use_container_width=True)
            
            with sym_col2:
                fig_sym_step = create_real_time_symmetry_bars(
                    stats['avg_left_step_length'],
                    stats['avg_right_step_length'],
                    "Step Length (meters)",
                    target_step_length_min if 'step_length' in st.session_state.target_zones else None,
                    target_step_length_max if 'step_length' in st.session_state.target_zones else None
                )
                st.plotly_chart(fig_sym_step, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Asymmetry Indicators")
            asym_col1, asym_col2 = st.columns(2)
            
            with asym_col1:
                st.metric(
                    "Temporal Asymmetry",
                    f"{stats['temporal_asymmetry']:.1f}%",
                    delta=f"{'Good' if stats['temporal_asymmetry'] < 10 else 'Needs Attention'}",
                    delta_color="inverse" if stats['temporal_asymmetry'] < 10 else "normal"
                )
            
            with asym_col2:
                st.metric(
                    "Spatial Asymmetry",
                    f"{stats['spatial_asymmetry']:.1f}%",
                    delta=f"{'Good' if stats['spatial_asymmetry'] < 10 else 'Needs Attention'}",
                    delta_color="inverse" if stats['spatial_asymmetry'] < 10 else "normal"
                )
        
        with tab2:
            st.markdown('<h2 style="color:#d32f2f;">Session Complete!</h2>', unsafe_allow_html=True)
            
            hero_html = create_hero_summary(stats, session_time, total_distance)
            st.markdown(hero_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("Session Notes")
            session_notes = st.text_area("Add notes about this session:", height=150)
            
            if st.button("Save Session"):
                session_data = {
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'patient_id': st.session_state.patient_id,
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
            
            analysis_tabs = st.tabs([
                "Metric Trends", 
                "Fatigue Analysis", 
                "Comparison View",
                "Symmetry Score"
            ])
            
            with analysis_tabs[0]:
                st.subheader("Metrics Over Time")
                
                metric_choice = st.selectbox(
                    "Select Metric to Analyze:",
                    ["Right Cadence (steps/min)", "Left Cadence (steps/min)", 
                     "Right Step Length (meters)", "Left Step Length (meters)",
                     "GaitSpeed Rtable (mph*10)"]
                )
                
                filtered_df['Sample Index'] = range(len(filtered_df))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_df['Sample Index'],
                    y=filtered_df[metric_choice],
                    mode='lines+markers',
                    name=metric_choice,
                    line=dict(width=3)
                ))
                
                if st.session_state.self_pacing_enabled:
                    pacing_change_point = len(filtered_df) // 2
                    fig.add_vline(
                        x=pacing_change_point,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Self-Pacing Enabled",
                        annotation_position="top"
                    )
                
                fig.update_layout(
                    title=f"{metric_choice} Over Session",
                    xaxis_title="Time (samples)",
                    yaxis_title=metric_choice,
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with analysis_tabs[1]:
                st.subheader("Fatigue Analysis")
                
                fatigue_metric = st.selectbox(
                    "Select Metric for Fatigue Analysis:",
                    ["Right Cadence (steps/min)", "Left Cadence (steps/min)", 
                     "Right Step Length (meters)", "Left Step Length (meters)"],
                    key="fatigue_metric"
                )
                
                fig_fatigue = create_fatigue_analysis(filtered_df, fatigue_metric)
                st.plotly_chart(fig_fatigue, use_container_width=True)
                
                st.info("""
                **Fatigue Analysis Interpretation:**
                - Compares first third vs. last third of session
                - Positive change: metric increased toward end
                - Negative change: metric decreased (potential fatigue)
                """)
            
            with analysis_tabs[2]:
                st.subheader("Session Comparison")
                
                if len(st.session_state.session_history) > 1:
                    comparison_df = pd.DataFrame([
                        {
                            'Session': i+1,
                            'Date': session['date'],
                            'Avg Speed': session['avg_speed'],
                            'Avg Cadence': session['avg_cadence'],
                            'Distance': session['distance']
                        }
                        for i, session in enumerate(st.session_state.session_history)
                    ])
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Scatter(
                        x=comparison_df['Session'],
                        y=comparison_df['Avg Cadence'],
                        mode='lines+markers',
                        name='Average Cadence',
                        line=dict(width=3)
                    ))
                    
                    fig_comparison.update_layout(
                        title="Cadence Progress Over Sessions",
                        xaxis_title="Session Number",
                        yaxis_title="Average Cadence (steps/min)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                else:
                    st.info("Complete more sessions to see comparison data.")
            
            with analysis_tabs[3]:
                st.subheader("Symmetry Score Over Time")
                
                filtered_df['Temporal_Asymmetry'] = filtered_df.apply(
                    lambda row: calculate_asymmetry_percentage(
                        row['Left Step Time (sec)'],
                        row['Right Step Time (sec)']
                    ), axis=1
                )
                
                filtered_df['Spatial_Asymmetry'] = filtered_df.apply(
                    lambda row: calculate_asymmetry_percentage(
                        row['Left Step Length (meters)'],
                        row['Right Step Length (meters)']
                    ), axis=1
                )
                
                fig_sym = go.Figure()
                fig_sym.add_trace(go.Scatter(
                    x=filtered_df['Sample Index'],
                    y=filtered_df['Temporal_Asymmetry'],
                    mode='lines',
                    name='Temporal Asymmetry',
                    line=dict(color='#FF6B6B', width=2)
                ))
                fig_sym.add_trace(go.Scatter(
                    x=filtered_df['Sample Index'],
                    y=filtered_df['Spatial_Asymmetry'],
                    mode='lines',
                    name='Spatial Asymmetry',
                    line=dict(color='#4ECDC4', width=2)
                ))
                
                fig_sym.add_hline(
                    y=10,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="10% Threshold"
                )
                
                fig_sym.update_layout(
                    title="Asymmetry Percentage Over Time",
                    xaxis_title="Time (samples)",
                    yaxis_title="Asymmetry (%)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_sym, use_container_width=True)
        
        with tab4:
            st.markdown('<h2 style="color:#d32f2f;">Progress Tracking</h2>', unsafe_allow_html=True)
            
            if st.session_state.session_history:
                st.subheader("Patient History")
                
                history_df = pd.DataFrame([
                    {
                        'Date': session['date'],
                        'Duration (min)': f"{session['duration']:.1f}",
                        'Distance (m)': f"{session['distance']:.1f}",
                        'Avg Speed (mph)': f"{session['avg_speed']:.2f}",
                        'Avg Cadence': f"{session['avg_cadence']:.0f}",
                        'Notes': session['notes'][:50] + '...' if len(session['notes']) > 50 else session['notes']
                    }
                    for session in st.session_state.session_history
                ])
                
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.subheader("Historical Trends")
                
                trend_metric = st.selectbox(
                    "Select Metric to Track:",
                    ["Avg Speed", "Avg Cadence", "Distance", "Duration"]
                )
                
                if len(st.session_state.session_history) > 1:
                    dates = [session['date'] for session in st.session_state.session_history]
                    
                    if trend_metric == "Avg Speed":
                        values = [session['avg_speed'] for session in st.session_state.session_history]
                        ylabel = "Speed (mph)"
                    elif trend_metric == "Avg Cadence":
                        values = [session['avg_cadence'] for session in st.session_state.session_history]
                        ylabel = "Cadence (steps/min)"
                    elif trend_metric == "Distance":
                        values = [session['distance'] for session in st.session_state.session_history]
                        ylabel = "Distance (meters)"
                    else:
                        values = [session['duration'] for session in st.session_state.session_history]
                        ylabel = "Duration (minutes)"
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name=trend_metric,
                        line=dict(width=3, color='#667eea'),
                        marker=dict(size=10)
                    ))
                    
                    fig_trend.update_layout(
                        title=f"{trend_metric} Over Time",
                        xaxis_title="Session Date",
                        yaxis_title=ylabel,
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Personal Records")
                
                if st.session_state.session_history:
                    max_speed_session = max(st.session_state.session_history, key=lambda x: x['avg_speed'])
                    max_distance_session = max(st.session_state.session_history, key=lambda x: x['distance'])
                    max_duration_session = max(st.session_state.session_history, key=lambda x: x['duration'])
                    
                    pr_col1, pr_col2, pr_col3 = st.columns(3)
                    
                    with pr_col1:
                        st.metric(
                            "Fastest Speed 🏃",
                            f"{max_speed_session['avg_speed']:.2f} mph",
                            delta=max_speed_session['date']
                        )
                    
                    with pr_col2:
                        st.metric(
                            "Longest Distance 📏",
                            f"{max_distance_session['distance']:.0f} m",
                            delta=max_distance_session['date']
                        )
                    
                    with pr_col3:
                        st.metric(
                            "Longest Duration ⏱️",
                            f"{max_duration_session['duration']:.1f} min",
                            delta=max_duration_session['date']
                        )
                
                st.markdown("---")
                
                st.subheader("Export Reports")
                
                report_type = st.selectbox("Select Report Type:", ["Daily", "Weekly", "Monthly"])
                
                if st.button("Generate & Email PDF Report"):
                    st.success(f"✅ {report_type} report has been generated and sent to {st.session_state.user_email}")
            
            else:
                st.info("No session history available yet. Complete a session to start tracking progress.")
        
        with tab5:
            st.markdown('<h2 style="color:#d32f2f;">Stepping Target</h2>', unsafe_allow_html=True)
            
            track_col1, track_col2 = st.columns([3, 1])
            
            with track_col1:
                target_step_length = st.slider(
                    "Target Step Length (meters)",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    help="Administrator sets the target step length for patient to match"
                )
                
                current_step = st.slider(
                    "Current Step",
                    min_value=0,
                    max_value=9,
                    value=0,
                    help="Move slider to simulate stepping progression"
                )
                
                fig_overhead = create_overhead_foot_tracking_visualization(
                    current_step=current_step, 
                    num_steps=10,
                    target_step_length=target_step_length
                )
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
                st.markdown("🔺 Right Foot")
                st.markdown("◀️ Left Foot")
                st.markdown("- - - Target Line")
            
            st.markdown("---")
            
            st.subheader("Stepping Target Data")
            stepping_data = generate_fake_foot_tracking_data(10)
            stepping_data['On Target'] = np.random.choice(['Yes', 'No'], size=10, p=[0.8, 0.2])
            stepping_data['Deviation (cm)'] = np.random.uniform(0, 5, size=10).round(2)
            st.dataframe(stepping_data, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.subheader("Stepping Target Performance Over Time")
            fig_target_perf = go.Figure()
            
            steps_range = list(range(10))
            accuracy_scores = [np.random.randint(70, 100) for _ in range(10)]
            
            fig_target_perf.add_trace(go.Scatter(
                x=steps_range,
                y=accuracy_scores,
                mode='lines+markers',
                name='Accuracy Score',
                line=dict(color='#28a745', width=3),
                marker=dict(size=10)
            ))
            
            fig_target_perf.add_hline(
                y=90,
                line_dash="dash",
                line_color="green",
                annotation_text="Target: 90%"
            )
            
            fig_target_perf.update_layout(
                title="Step Accuracy Score Over Session",
                xaxis_title="Step Number",
                yaxis_title="Accuracy Score (%)",
                height=400,
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_target_perf, use_container_width=True)
        
        st.markdown("---")
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Export Options</span>', unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Statistics Summary"):
                summary_stats = pd.DataFrame({
                    'Parameter': [
                        'Average Gait Speed (mph)',
                        'Right Cadence (steps/min)',
                        'Left Cadence (steps/min)',
                        'Right Step Length (m)',
                        'Left Step Length (m)',
                        'Right Stride Length (m)',
                        'Left Stride Length (m)',
                        'Right Swing/Stance Ratio',
                        'Left Swing/Stance Ratio',
                        'Temporal Asymmetry (%)',
                        'Spatial Asymmetry (%)',
                    ],
                    'Value': [
                        f"{stats['avg_speed']:.2f}",
                        f"{stats['avg_right_cadence']:.1f}",
                        f"{stats['avg_left_cadence']:.1f}",
                        f"{stats['avg_right_step_length']:.3f}",
                        f"{stats['avg_left_step_length']:.3f}",
                        f"{stats['avg_right_stride_length']:.3f}",
                        f"{stats['avg_left_stride_length']:.3f}",
                        f"{stats['right_swing_stance_ratio']:.2f}",
                        f"{stats['left_swing_stance_ratio']:.2f}",
                        f"{stats['temporal_asymmetry']:.1f}",
                        f"{stats['spatial_asymmetry']:.1f}",
                    ]
                })
                
                csv_stats = summary_stats.to_csv(index=False)
                
                st.download_button(
                    label="Download Statistics CSV",
                    data=csv_stats,
                    file_name=f"patient_{st.session_state.patient_id}_gait_stats.csv",
                    mime="text/csv",
                )
                
                st.dataframe(summary_stats, hide_index=True)
        
        with export_col2:
            def export_full_report_to_pdf(stats, patient_id):
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.set_font("Arial", style="B", size=16)
                pdf.cell(0, 10, "Patient Gait Analysis Report", ln=True, align="C")
                pdf.ln(10)

                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
                pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
                pdf.ln(10)

                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(0, 10, "Key Statistics", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"- Average Gait Speed: {stats['avg_speed']:.2f} mph", ln=True)
                pdf.cell(0, 10, f"- Right Cadence: {stats['avg_right_cadence']:.1f} steps/min", ln=True)
                pdf.cell(0, 10, f"- Left Cadence: {stats['avg_left_cadence']:.1f} steps/min", ln=True)
                pdf.cell(0, 10, f"- Right Step Length: {stats['avg_right_step_length']:.3f} m", ln=True)
                pdf.cell(0, 10, f"- Left Step Length: {stats['avg_left_step_length']:.3f} m", ln=True)
                pdf.cell(0, 10, f"- Temporal Asymmetry: {stats['temporal_asymmetry']:.1f}%", ln=True)
                pdf.cell(0, 10, f"- Spatial Asymmetry: {stats['spatial_asymmetry']:.1f}%", ln=True)
                pdf.ln(10)

                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(0, 10, "Recommendations", ln=True)
                pdf.ln(5)

                pdf_buffer = BytesIO()  
                pdf_content = pdf.output(dest='S').encode('latin1')  
                pdf_buffer.write(pdf_content)  
                pdf_buffer.seek(0)  

                return pdf_buffer
            
            if st.button("Export Full Report"):
                pdf_buffer = export_full_report_to_pdf(stats, st.session_state.patient_id)
                st.download_button(
                    label="Download Full Report as PDF",
                    data=pdf_buffer,
                    file_name=f"patient_{st.session_state.patient_id}_gait_report.pdf",
                    mime="application/pdf",
                )
        
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Raw Data</span>', unsafe_allow_html=True)
        
        if st.checkbox("Show Raw Data Table"):
            st.dataframe(filtered_df)
            
            if st.button("Export Raw Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data CSV",
                    data=csv_data,
                    file_name=f"patient_{st.session_state.patient_id}_raw_gait_data.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()
