import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import re
from zoneinfo import ZoneInfo
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# MySQL Connection Setup
import os
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pymysql


def get_cest_time() -> datetime:
    """Return current time in CEST (Europe/Oslo)."""
    return datetime.now(ZoneInfo("Europe/Oslo"))


def get_mysql_config():
    """Get MySQL configuration from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
            os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
        elif hasattr(st, 'secrets') and 'MYSQL_HOST' in st.secrets:
            for key in ['MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE']:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
    except Exception as e:
        pass  # Silently fall back to environment variables

    # Fallback to environment variables
    return {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'port': int(os.getenv('MYSQL_PORT', 3306)),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'ev_charging'),
        'charset': 'utf8mb4'
    }


def get_connection_string():
    """Get MySQL connection string"""
    # Check if DATABASE_URL is provided (Railway/PlanetScale format)
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL').replace('mysql://', 'mysql+pymysql://')

    # Otherwise build from components
    config = get_mysql_config()
    password = quote_plus(config['password'])
    return (
        f"mysql+pymysql://{config['user']}:{password}@"
        f"{config['host']}:{config['port']}/{config['database']}"
        f"?charset={config['charset']}"
    )


@st.cache_resource
def get_engine():
    """Create SQLAlchemy engine with connection pooling - cached as resource"""
    try:
        return create_engine(
            get_connection_string(),
            pool_size=10,  # Increased pool size
            max_overflow=20,  # Increased overflow
            pool_timeout=10,  # Reduced timeout
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False
        )
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        return None


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create global engine instance
engine = get_engine()
MYSQL_AVAILABLE = engine is not None


# Optimized MySQL Data Loading Functions with connection pooling
def load_from_mysql_optimized(table_name, where_clause=None, limit=None, columns=None):
    """Optimized data loading with column selection and better error handling"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        # Select only needed columns to reduce data transfer
        if columns:
            column_str = ', '.join(columns)
        else:
            column_str = '*'
            
        query = f"SELECT {column_str} FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        # Use connection pool efficiently
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Parse only datetime columns that exist
        datetime_columns = ['timestamp', 'hourly_timestamp', 'start_time', 'end_time', 'created_at', 'last_updated']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    except Exception as e:
        logger.error(f"Error loading from MySQL table {table_name}: {e}")
        return pd.DataFrame()


# Optimized data loading functions with aggressive caching
@st.cache_data(ttl=30, max_entries=5, show_spinner=False)  # Reduced TTL, limited entries
def load_stations_data():
    """Load essential charging stations data only"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required.")
        st.stop()

    try:
        # Load only essential columns
        essential_columns = ['id', 'name', 'address', 'latitude', 'longitude', 'status', 
                           'total_connectors', 'ccs_connectors', 'chademo_connectors', 'type2_connectors']
        
        df = load_from_mysql_optimized('charging_stations', columns=essential_columns)
        if df.empty:
            st.warning("⚠️ No stations data found.")
            st.stop()

        # Efficient type conversion
        numeric_cols = ['latitude', 'longitude', 'total_connectors', 'ccs_connectors',
                        'chademo_connectors', 'type2_connectors']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Error loading stations: {e}")
        st.stop()


@st.cache_data(ttl=15, max_entries=3, show_spinner=False)  # Very short TTL for real-time data
def load_utilization_data():
    """Load latest utilization data with optimized query"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required.")
        st.stop()

    try:
        # Optimized query with subquery for better performance
        query = """
        SELECT u1.* FROM utilization_data u1
        INNER JOIN (
            SELECT connector_id, MAX(timestamp) as max_timestamp
            FROM utilization_data
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            GROUP BY connector_id
        ) u2 ON u1.connector_id = u2.connector_id AND u1.timestamp = u2.max_timestamp
        LIMIT 1000
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading utilization data: {e}")
        st.stop()


@st.cache_data(ttl=120, max_entries=2, show_spinner=False)  # Longer TTL for historical data
def load_historical_utilization_data(hours=24):
    """Load historical utilization data with limit"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        # Limit data and use efficient query
        query = """
        SELECT connector_id, station_id, timestamp, is_occupied, is_available, is_out_of_order
        FROM utilization_data
        WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY timestamp DESC
        LIMIT 5000
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=(hours,))
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        return df
    except Exception as e:
        logger.error(f"Error getting historical utilization: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)  # 5 min cache for aggregated data
def load_hourly_data():
    """Load hourly aggregated data efficiently"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        query = """
        SELECT 
            hourly_timestamp,
            SUM(is_occupied) as total_occupied,
            SUM(is_available) as total_available,
            SUM(is_out_of_order) as total_out_of_order,
            COUNT(*) as total_connectors,
            AVG(CASE WHEN is_occupied = 1 THEN 1.0 ELSE 0.0 END) as avg_occupancy_rate
        FROM hourly_utilization
        WHERE hourly_timestamp >= DATE_SUB(NOW(), INTERVAL 168 HOUR)
        GROUP BY hourly_timestamp
        ORDER BY hourly_timestamp DESC
        LIMIT 168
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if 'hourly_timestamp' in df.columns:
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'], errors='coerce')
            
        return df
    except Exception as e:
        logger.error(f"Error getting hourly stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)  # 5 min cache for sessions
def load_sessions_data():
    """Load recent charging sessions efficiently"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        # Load essential session columns only
        query = """
        SELECT 
            cs.connector_id, cs.station_id, cs.start_time, cs.end_time,
            cs.energy_kwh, cs.revenue_nok, cs.duration_hours,
            st.name as station_name, st.address as station_address
        FROM charging_sessions cs
        LEFT JOIN charging_stations st ON cs.station_id = st.id
        WHERE cs.end_time >= DATE_SUB(NOW(), INTERVAL 168 HOUR)
        ORDER BY cs.end_time DESC
        LIMIT 1000
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        # Efficient datetime parsing
        for col in ['start_time', 'end_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        return df
    except Exception as e:
        logger.error(f"Error getting recent sessions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def test_mysql_connection():
    """Cached connection test"""
    if not MYSQL_AVAILABLE or engine is None:
        return False, "Engine not available"

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return True, "Connected successfully"
    except Exception as e:
        return False, str(e)


# Page configuration with performance optimizations
st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized CSS - reduced and more efficient
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .refresh-timer {
        position: fixed;
        top: 70px;
        right: 20px;
        background-color: #f8f9fa;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 999;
    }
    .station-card {
        border-radius: 8px;
        padding: 12px;
        margin: 3px;
        text-align: center;
        height: 160px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Import timezone handling
import pytz
CEST = pytz.timezone('Europe/Oslo')

# Optimized session state initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

REFRESH_INTERVAL = 60


# Optimized refresh functionality
def check_auto_refresh():
    """Efficient auto-refresh check"""
    current_time = time.time()
    
    if (st.session_state.auto_refresh_enabled and
            current_time - st.session_state.last_refresh > REFRESH_INTERVAL):
        st.session_state.last_refresh = current_time
        # Clear only real-time data caches
        load_utilization_data.clear()
        # Less frequent clearing for other data
        if current_time % 300 < 60:
            load_stations_data.clear()
            load_historical_utilization_data.clear()
            load_hourly_data.clear()
            load_sessions_data.clear()
        st.rerun()


def show_refresh_timer():
    """Lightweight refresh timer"""
    if st.session_state.auto_refresh_enabled:
        time_since_refresh = int(time.time() - st.session_state.last_refresh)
        time_to_refresh = REFRESH_INTERVAL - time_since_refresh

        if time_to_refresh > 0:
            st.markdown(
                f'<div class="refresh-timer">🔄 {time_to_refresh}s</div>',
                unsafe_allow_html=True
            )


# Revenue calculation utilities
def extract_tariff(tariff_str):
    """Extract numeric tariff from string"""
    if pd.isna(tariff_str) or tariff_str == '':
        return 0.0
    tariff_str = re.sub(r'[^\d.,]', '', str(tariff_str)).replace(',', '.')
    match = re.search(r'\d+\.?\d*', tariff_str)
    return float(match.group()) if match else 0.0


# Streamlined error handling
def handle_deployment_errors():
    """Lightweight error handling"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ **Database Connection Required**")
        st.markdown("Configure MySQL credentials in Streamlit secrets or environment variables.")
        st.stop()

    try:
        connection_status, message = test_mysql_connection()
        if not connection_status:
            st.error(f"❌ **Database Connection Failed**: {message}")
            st.stop()
    except Exception as e:
        st.error(f"❌ **Database Error**: {e}")
        st.stop()


# Optimized main dashboard
def main():
    check_auto_refresh()

    # Streamlined title
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 15px;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">⚡ EV Charging Analytics</h1>
        <p>Real-time Updates | Last refresh: {} CEST</p>
    </div>
    """.format(get_cest_time().strftime('%H:%M:%S')), unsafe_allow_html=True)

    show_refresh_timer()

    # Load data efficiently with minimal blocking
    try:
        stations_df = load_stations_data()
        utilization_df = load_utilization_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Streamlined sidebar
    with st.sidebar:
        st.markdown("### 🔌 Navigation")

        page_options = ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analytics",
                        "⚡ Real-time Monitor", "📋 Data Explorer"]

        page = st.radio(
            "Select Dashboard",
            options=page_options,
            index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0,
            key="navigation_radio"
        )

        if page != st.session_state.current_page:
            st.session_state.current_page = page

        st.markdown("---")
        st.markdown("### 🕐 Last Updated")
        st.info(f"{get_cest_time().strftime('%H:%M:%S')}")

        # Simplified controls
        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh

        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

        # Quick stats
        st.markdown("---")
        st.metric("Stations", len(stations_df))
        active_sessions = len(utilization_df[utilization_df['is_occupied'] == 1]) if not utilization_df.empty else 0
        st.metric("Active", active_sessions)

        # Connection status
        connection_status, _ = test_mysql_connection()
        if connection_status:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")

    # Load additional data only when needed
    hourly_df = pd.DataFrame()
    sessions_df = pd.DataFrame()
    historical_util_df = pd.DataFrame()

    # Conditional data loading based on current page
    if st.session_state.current_page in ["📊 Overview", "📈 Utilization Analytics", "📋 Data Explorer"]:
        hourly_df = load_hourly_data()
        sessions_df = load_sessions_data()

    if st.session_state.current_page == "📈 Utilization Analytics":
        historical_util_df = load_historical_utilization_data(24)

    # Route to appropriate page
    if st.session_state.current_page == "📊 Overview":
        show_overview(stations_df, utilization_df, hourly_df, sessions_df)
    elif st.session_state.current_page == "🗺️ Station Map":
        show_station_map(stations_df, utilization_df, sessions_df)
    elif st.session_state.current_page == "📈 Utilization Analytics":
        show_utilization_analytics(utilization_df, hourly_df, sessions_df, historical_util_df)
    elif st.session_state.current_page == "⚡ Real-time Monitor":
        show_realtime_monitor(stations_df, utilization_df, sessions_df)
    elif st.session_state.current_page == "📋 Data Explorer":
        show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df)


def show_overview(stations_df, utilization_df, hourly_df, sessions_df):
    """Optimized overview dashboard"""
    st.header("📊 Overview")

    # Pre-calculate metrics to avoid repeated computations
    total_stations = len(stations_df)
    available_stations = len(stations_df[stations_df['status'] == 'Available']) if not stations_df.empty else 0
    total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0
    occupied_connectors = len(utilization_df[utilization_df['is_occupied'] == 1]) if not utilization_df.empty else 0

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Available Stations",
            f"{available_stations}/{total_stations}",
            f"{(available_stations / total_stations * 100):.1f}%" if total_stations > 0 else "0%"
        )

    with col2:
        st.metric(
            "Active Sessions",
            occupied_connectors,
            f"{(occupied_connectors / total_connectors * 100):.1f}%" if total_connectors > 0 else "0%"
        )

    with col3:
        if not hourly_df.empty and 'avg_occupancy_rate' in hourly_df.columns:
            avg_occupancy = hourly_df['avg_occupancy_rate'].mean() * 100
        else:
            avg_occupancy = (occupied_connectors / total_connectors * 100) if total_connectors > 0 else 0
        st.metric("Avg Occupancy", f"{avg_occupancy:.1f}%")

    with col4:
        if not sessions_df.empty and 'revenue_nok' in sessions_df.columns:
            last_24h = datetime.now() - timedelta(hours=24)
            recent_sessions = sessions_df[sessions_df['end_time'] >= last_24h]
            daily_revenue_nok = recent_sessions['revenue_nok'].sum()
            daily_revenue_usd = daily_revenue_nok / 10.5
        else:
            daily_revenue_usd = 0
        st.metric("Daily Revenue", f"${daily_revenue_usd:,.0f}")

    st.markdown("---")

    # Optimized charts
    col1, col2 = st.columns(2)

    with col1:
        if not stations_df.empty:
            status_counts = stations_df['status'].value_counts()
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Station Status",
                color_discrete_map={
                    'Available': '#2ecc71',
                    'Occupied': '#f39c12',
                    'OutOfOrder': '#e74c3c'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        if not stations_df.empty:
            connector_data = {
                'CCS': stations_df['ccs_connectors'].sum(),
                'CHAdeMO': stations_df['chademo_connectors'].sum(),
                'Type 2': stations_df['type2_connectors'].sum()
            }
            fig_bar = px.bar(
                x=list(connector_data.keys()),
                y=list(connector_data.values()),
                title="Connector Types",
                color=list(connector_data.keys()),
                color_discrete_map={
                    'CCS': '#3498db',
                    'CHAdeMO': '#9b59b6',
                    'Type 2': '#1abc9c'
                }
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # Simplified hourly pattern
    if not sessions_df.empty and 'start_time' in sessions_df.columns:
        st.markdown("---")
        sessions_df_copy = sessions_df.copy()
        sessions_df_copy['hour'] = sessions_df_copy['start_time'].dt.hour
        sessions_per_hour = sessions_df_copy.groupby('hour').size()
        
        # Ensure all hours represented
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(sessions_per_hour)
        
        fig_line = px.line(
            x=all_hours.index,
            y=all_hours.values,
            title='24-Hour Usage Pattern'
        )
        fig_line.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Sessions Started",
            height=300
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Top stations table (simplified)
    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        st.markdown("---")
        st.subheader("🏆 Top Stations")
        
        station_revenue = sessions_df.groupby('station_id').agg({
            'revenue_nok': 'sum',
            'energy_kwh': 'sum',
            'connector_id': 'count'
        }).reset_index()
        station_revenue.columns = ['station_id', 'total_revenue', 'total_energy', 'session_count']

        station_performance = station_revenue.merge(
            stations_df[['id', 'name', 'address']],
            left_on='station_id',
            right_on='id',
            how='left'
        ).sort_values('total_revenue', ascending=False).head(5)

        if not station_performance.empty:
            st.dataframe(
                station_performance[['name', 'total_revenue', 'session_count']].rename(
                    columns={
                        'name': 'Station',
                        'total_revenue': 'Revenue (NOK)',
                        'session_count': 'Sessions'
                    }),
                hide_index=True,
                use_container_width=True
            )


def show_station_map(stations_df, utilization_df, sessions_df):
    """Optimized station map"""
    st.header("🗺️ Station Map")

    # Pre-calculate revenue dictionary for efficiency
    station_revenue = {}
    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        station_revenue = sessions_df.groupby('station_id')['revenue_nok'].sum().to_dict()

    # Simplified filters
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.multiselect(
            "Status Filter",
            options=['Available', 'Occupied', 'OutOfOrder'],
            default=['Available', 'Occupied', 'OutOfOrder'],
            key="map_status"
        )

    with col2:
        connector_min = st.slider(
            "Min Connectors",
            1, int(stations_df['total_connectors'].max()) if not stations_df.empty else 10,
            1,
            key="map_connectors"
        )

    # Filter data efficiently
    if not stations_df.empty:
        filtered_df = stations_df[
            (stations_df['status'].isin(status_filter)) &
            (stations_df['total_connectors'] >= connector_min)
        ]
    else:
        filtered_df = pd.DataFrame()

    # Create optimized map
    if not filtered_df.empty:
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )

        # Optimized marker creation
        for _, station in filtered_df.iterrows():
            # Get connector status efficiently
            if not utilization_df.empty:
                station_connectors = utilization_df[utilization_df['station_id'] == station['id']]
                occupied = len(station_connectors[station_connectors['is_occupied'] == 1])
                available = len(station_connectors[station_connectors['is_available'] == 1])
            else:
                occupied = 0
                available = station['total_connectors']

            # Simple color logic
            if available > 0:
                color = 'green'
                status_text = 'Available'
            else:
                color = 'red'
                status_text = 'Occupied'

            revenue = station_revenue.get(station['id'], 0)

            # Simplified popup
            popup_html = f"""
            <b>{station['name']}</b><br>
            Status: {status_text}<br>
            Available: {available}/{station['total_connectors']}<br>
            Revenue: NOK {revenue:,.0f}
            """

            folium.Marker(
                location=[station['latitude'], station['longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon='plug', prefix='fa')
            ).add_to(m)

        st_folium(m, width=None, height=500, key="station_map")

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shown", len(filtered_df))
        with col2:
            st.metric("Connectors", filtered_df['total_connectors'].sum())
        with col3:
            total_rev = sum(station_revenue.get(sid, 0) for sid in filtered_df['id'])
            st.metric("Revenue", f"NOK {total_rev:,.0f}")


def show_utilization_analytics(utilization_df, hourly_df, sessions_df, historical_util_df):
    """Optimized analytics with faster heatmap"""
    st.header("📈 Analytics")

    # Simplified time range
    time_range = st.selectbox(
        "Time Range",
        ["Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=1,
        key="analytics_range"
    )

    # Filter data
    now = datetime.now()
    if time_range == "Last 6 Hours":
        time_filter = now - timedelta(hours=6)
    elif time_range == "Last 24 Hours":
        time_filter = now - timedelta(hours=24)
    else:
        time_filter = now - timedelta(days=7)

    filtered_util = utilization_df[
        utilization_df['timestamp'] >= time_filter] if not utilization_df.empty else utilization_df
    filtered_sessions = sessions_df[
        sessions_df['end_time'] >= time_filter] if not sessions_df.empty else sessions_df

    # Streamlined tabs
    tab1, tab2 = st.tabs(["📊 Heatmap", "📈 Trends"])

    with tab1:
        # Optimized heatmap
        st.subheader("Utilization Heatmap")
        
        if not historical_util_df.empty and len(historical_util_df) > 10:
            # Simplified heatmap preparation
            heatmap_data = historical_util_df.copy()
            heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
            heatmap_data['day'] = heatmap_data['timestamp'].dt.day_name()
            
            # Faster pivot with less data processing
            pivot_data = heatmap_data.groupby(['day', 'hour'])['is_occupied'].mean().reset_index()
            
            if not pivot_data.empty and len(pivot_data) > 5:
                pivot_table = pivot_data.pivot(index='day', columns='hour', values='is_occupied')
                
                # Fill missing hours efficiently
                for hour in range(24):
                    if hour not in pivot_table.columns:
                        pivot_table[hour] = 0
                
                pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
                pivot_table = pivot_table.fillna(0)

                # Simplified colorscale
                custom_colorscale = [
                    [0.0, '#27ae60'],
                    [0.2, '#f1c40f'], 
                    [0.5, '#f39c12'],
                    [0.8, '#e74c3c'],
                    [1.0, '#8b0000']
                ]

                # Optimized heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=list(range(24)),
                    y=pivot_table.index.tolist(),
                    colorscale=custom_colorscale,
                    colorbar=dict(title='Occupancy Rate'),
                    hoverongaps=False,
                    hovertemplate='%{y}<br>Hour: %{x}<br>Rate: %{z:.1%}<extra></extra>',
                    zmin=0,
                    zmax=0.40
                ))

                fig_heatmap.update_layout(
                    title='Utilization Pattern by Hour and Day',
                    xaxis_title='Hour',
                    yaxis_title='Day',
                    height=400,
                    font=dict(size=10)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak", f"{pivot_table.values.max():.1%}")
                with col2:
                    st.metric("Average", f"{pivot_table.values.mean():.1%}")
                with col3:
                    peak_hour = pivot_table.mean(axis=0).idxmax()
                    st.metric("Peak Hour", f"{peak_hour}:00")
            else:
                st.info("Insufficient data for heatmap")
        else:
            st.info("No historical data available")

    with tab2:
        # Simplified trends
        col1, col2 = st.columns(2)

        with col1:
            if not filtered_sessions.empty and 'start_time' in filtered_sessions.columns:
                peak_hour_sessions = filtered_sessions.groupby(filtered_sessions['start_time'].dt.hour).size()
                if not peak_hour_sessions.empty:
                    peak_hour = peak_hour_sessions.idxmax()
                    peak_count = peak_hour_sessions.max()
                    st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_count} sessions")
                else:
                    st.metric("Peak Hour", "No data")
            else:
                st.metric("Peak Hour", "No data")

        with col2:
            total_completed = len(filtered_sessions) if not filtered_sessions.empty else 0
            current_active = len(filtered_util[filtered_util['is_occupied'] == 1]) if not filtered_util.empty else 0
            st.metric("Sessions", f"{total_completed} completed", f"{current_active} active")

        # Simplified trend chart
        if not filtered_sessions.empty:
            st.subheader("Session Timeline")
            
            # Simple daily aggregation
            daily_sessions = filtered_sessions.groupby(filtered_sessions['start_time'].dt.date).size()
            
            fig_trend = px.line(
                x=daily_sessions.index,
                y=daily_sessions.values,
                title='Daily Sessions'
            )
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Revenue analysis
        if not filtered_sessions.empty and 'revenue_nok' in filtered_sessions.columns:
            st.subheader("Revenue Stats")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_energy = filtered_sessions['energy_kwh'].sum()
                st.metric("Energy", f"{total_energy:,.0f} kWh")
            with col2:
                total_revenue = filtered_sessions['revenue_nok'].sum()
                st.metric("Revenue", f"NOK {total_revenue:,.0f}")
            with col3:
                avg_session = filtered_sessions['revenue_nok'].mean()
                st.metric("Avg/Session", f"NOK {avg_session:.1f}")


def show_realtime_monitor(stations_df, utilization_df, sessions_df):
    """Optimized real-time monitor"""
    st.header("⚡ Real-time Monitor")

    # Pre-calculate all metrics
    if not utilization_df.empty:
        current_available = len(utilization_df[utilization_df['is_available'] == 1])
        current_occupied = len(utilization_df[utilization_df['is_occupied'] == 1])
        current_out_of_order = len(utilization_df[utilization_df['is_out_of_order'] == 1])
        total_connectors = len(utilization_df)
    else:
        current_available = current_occupied = current_out_of_order = 0
        total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0

    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Available", f"{current_available}/{total_connectors}")
    with col2:
        st.metric("🟠 Occupied", f"{current_occupied}/{total_connectors}")
    with col3:
        st.metric("🔴 Out of Order", current_out_of_order)
    with col4:
        utilization_pct = (current_occupied / total_connectors * 100) if total_connectors > 0 else 0
        st.metric("📊 Utilization", f"{utilization_pct:.1f}%")

    # Revenue tracking
    if not sessions_df.empty:
        st.markdown("---")
        st.subheader("💰 Revenue Tracking")

        col1, col2, col3 = st.columns(3)
        now = datetime.now()

        with col1:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_sessions = sessions_df[sessions_df['start_time'] >= today_start]
            today_revenue = today_sessions['revenue_nok'].sum()
            st.metric("Today", f"NOK {today_revenue:,.0f}")

        with col2:
            last_hour = now - timedelta(hours=1)
            hour_sessions = sessions_df[sessions_df['start_time'] >= last_hour]
            hour_revenue = hour_sessions['revenue_nok'].sum()
            st.metric("Last Hour", f"NOK {hour_revenue:,.0f}")

        with col3:
            st.metric("Active Sessions", current_occupied)

    # Simplified station grid
    st.markdown("---")
    st.subheader("Station Status")

    search_term = st.text_input("🔍 Search stations", key="realtime_search")

    if search_term and not stations_df.empty:
        display_df = stations_df[
            stations_df['name'].str.contains(search_term, case=False, na=False)
        ]
    else:
        display_df = stations_df.head(20)  # Limit to first 20 for performance

    # Efficient grid display
    if not display_df.empty:
        for i in range(0, min(len(display_df), 15), 5):  # Limit to 15 stations max
            cols = st.columns(5)
            
            for j, col in enumerate(cols):
                if i + j < len(display_df):
                    station = display_df.iloc[i + j]
                    
                    # Get station status efficiently
                    if not utilization_df.empty:
                        station_util = utilization_df[utilization_df['station_id'] == station['id']]
                        occupied = len(station_util[station_util['is_occupied'] == 1])
                        total = len(station_util) if len(station_util) > 0 else station['total_connectors']
                    else:
                        occupied = 0
                        total = station['total_connectors']

                    with col:
                        # Simple status display
                        if occupied == 0:
                            status_color = "#2ecc71"
                            status_icon = "✅"
                        elif occupied < total:
                            status_color = "#f39c12"
                            status_icon = "🔌"
                        else:
                            status_color = "#e74c3c"
                            status_icon = "⚡"

                        st.markdown(f"""
                        <div style="
                            background-color: {status_color}20;
                            border: 2px solid {status_color};
                            border-radius: 8px;
                            padding: 10px;
                            text-align: center;
                            margin: 2px;
                            height: 120px;">
                            <h5 style="margin: 0; font-size: 0.8em;">{status_icon} {station['name'][:15]}...</h5>
                            <p style="margin: 5px 0; font-size: 0.7em;"><b>{occupied}/{total}</b></p>
                            <p style="margin: 0; font-size: 0.6em;">⚡ {station['total_connectors']}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Recent activity (limited for performance)
    st.markdown("---")
    st.subheader("📰 Recent Activity")

    if not sessions_df.empty:
        recent_sessions = sessions_df.sort_values('start_time', ascending=False).head(5)
        
        for _, session in recent_sessions.iterrows():
            station_name = session.get('station_name', 'Unknown Station')
            if pd.isna(station_name):
                station_match = stations_df[stations_df['id'] == session.get('station_id')]
                station_name = station_match['name'].values[0] if len(station_match) > 0 else 'Unknown'

            time_ago = (datetime.now() - session['start_time']).total_seconds() / 60
            if time_ago < 60:
                time_str = f"{int(time_ago)}m ago"
            else:
                time_str = f"{int(time_ago / 60)}h ago"

            st.markdown(
                f"• 🔌 **{station_name}** - "
                f"{session['duration_hours'] * 60:.0f}min, "
                f"{session['energy_kwh']:.1f}kWh, "
                f"NOK {session['revenue_nok']:.0f} - *{time_str}*"
            )


def show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df):
    """Streamlined data explorer"""
    st.header("📋 Data Explorer")

    # Dataset selector
    dataset = st.selectbox(
        "Dataset",
        ["Charging Stations", "Utilization Data", "Hourly Aggregations", "Charging Sessions"],
        key="explorer_dataset"
    )

    # Get dataframe
    df_map = {
        "Charging Stations": stations_df,
        "Utilization Data": utilization_df,
        "Hourly Aggregations": hourly_df,
        "Charging Sessions": sessions_df
    }
    
    df = df_map[dataset]

    if df.empty:
        st.warning(f"No data available for {dataset}")
        return

    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage().sum() / 1024 ** 2
        st.metric("Memory", f"{memory_mb:.1f} MB")

    # Simplified filters for key datasets
    filtered_df = df.copy()
    
    if dataset == "Charging Stations" and 'status' in df.columns:
        status_filter = st.multiselect("Status Filter", df['status'].unique(), key="explorer_status")
        if status_filter:
            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    
    elif dataset == "Charging Sessions" and 'revenue_nok' in df.columns:
        revenue_range = st.slider(
            "Revenue Range (NOK)",
            float(df['revenue_nok'].min()),
            float(df['revenue_nok'].max()),
            (float(df['revenue_nok'].min()), float(df['revenue_nok'].max())),
            key="explorer_revenue"
        )
        filtered_df = filtered_df[
            (filtered_df['revenue_nok'] >= revenue_range[0]) &
            (filtered_df['revenue_nok'] <= revenue_range[1])
        ]

    # Data preview with limits
    st.subheader("Data Preview")
    
    n_rows = st.slider("Rows to show", 10, 100, 50, key="explorer_rows")
    
    if not filtered_df.empty:
        # Show limited columns for better performance
        display_cols = filtered_df.columns.tolist()[:10]  # First 10 columns
        if len(filtered_df.columns) > 10:
            st.info(f"Showing first 10 of {len(filtered_df.columns)} columns")
        
        st.dataframe(
            filtered_df[display_cols].head(n_rows),
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        if st.button("📥 Download CSV", key="download_btn"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download Complete Dataset",
                csv,
                f"{dataset.lower().replace(' ', '_')}.csv",
                "text/csv",
                key="download_csv"
            )


# Streamlined footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 10px;">
        <p>⚡ EV Charging Analytics | Real-time Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    handle_deployment_errors()
    main()
    add_footer()
