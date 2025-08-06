"""
EV Charging Analytics Dashboard - Optimized for Railway MySQL
Professional dashboard with real-time data visualization and analytics
"""

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
import pytz

# MySQL Connection Setup
import os
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Import MySQL connector with fallback options
try:
    import pymysql
    pymysql.install_as_MySQLdb()  # This makes pymysql work as MySQLdb
    MYSQL_DRIVER = "pymysql"
except ImportError:
    try:
        import mysql.connector
        MYSQL_DRIVER = "mysql+connector"
    except ImportError:
        try:
            import MySQLdb
            MYSQL_DRIVER = "mysqlclient"
        except ImportError:
            st.error("❌ No MySQL driver found. Please install one of: pymysql, mysql-connector-python, or mysqlclient")
            st.info("Run: pip install pymysql")
            st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    .station-card {
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .station-card:hover {
        transform: translateY(-2px);
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .refresh-info {
        position: fixed;
        top: 70px;
        right: 20px;
        background: rgba(255,255,255,0.9);
        padding: 8px 15px;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 0.8rem;
        z-index: 999;
    }
    .kpi-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database Configuration
def get_mysql_config():
    """Get MySQL configuration from environment or Streamlit secrets"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets'):
            if 'DATABASE_URL' in st.secrets:
                return st.secrets['DATABASE_URL']
            elif all(key in st.secrets for key in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE']):
                config = {
                    'host': st.secrets['MYSQL_HOST'],
                    'port': int(st.secrets.get('MYSQL_PORT', 3306)),
                    'user': st.secrets['MYSQL_USER'],
                    'password': st.secrets['MYSQL_PASSWORD'],
                    'database': st.secrets['MYSQL_DATABASE']
                }
                password = quote_plus(config['password'])
                return f"mysql+{MYSQL_DRIVER}://{config['user']}:{password}@{config['host']}:{config['port']}/{config['database']}?charset=utf8mb4"
    except:
        pass
    
    # Fallback to environment variables
    if os.getenv('DATABASE_URL'):
        db_url = os.getenv('DATABASE_URL')
        # Replace mysql:// with the appropriate driver
        if db_url.startswith('mysql://'):
            db_url = db_url.replace('mysql://', f'mysql+{MYSQL_DRIVER}://')
        return db_url
    
    # Build from individual components
    host = os.getenv('MYSQL_HOST', 'localhost')
    port = int(os.getenv('MYSQL_PORT', 3306))
    user = os.getenv('MYSQL_USER', 'root')
    password = quote_plus(os.getenv('MYSQL_PASSWORD', ''))
    database = os.getenv('MYSQL_DATABASE', 'ev_charging_db')
    
    return f"mysql+{MYSQL_DRIVER}://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"

@st.cache_resource
def get_database_engine():
    """Create and cache database engine"""
    try:
        connection_string = get_mysql_config()
        engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.info("Please check your database credentials in Streamlit secrets or environment variables")
        st.stop()

# Initialize database engine
engine = get_database_engine()

# Timezone setup
OSLO_TZ = pytz.timezone('Europe/Oslo')

def get_oslo_time():
    """Get current time in Oslo timezone"""
    return datetime.now(OSLO_TZ)

def format_oslo_time(dt):
    """Convert UTC datetime to Oslo timezone"""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(OSLO_TZ)

# Optimized data loading functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_stations_data():
    """Load charging stations data"""
    query = """
    SELECT 
        id, name, operator, status, address, description,
        latitude, longitude, total_connectors, 
        ccs_connectors, chademo_connectors, type2_connectors, other_connectors,
        amenities, last_updated
    FROM charging_stations
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.warning("⚠️ No charging stations found in database")
            return pd.DataFrame()
        
        # Convert numeric columns
        numeric_cols = ['latitude', 'longitude', 'total_connectors', 
                       'ccs_connectors', 'chademo_connectors', 'type2_connectors', 'other_connectors']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading stations: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_utilization_data():
    """Load latest available 24 hours of utilization data"""
    try:
        # First, find the latest timestamp in the data
        latest_query = "SELECT MAX(timestamp) as latest_time FROM utilization_data"
        latest_result = pd.read_sql(latest_query, engine)
        
        if latest_result.empty or latest_result['latest_time'].iloc[0] is None:
            st.warning("No utilization data found in database")
            return pd.DataFrame()
        
        latest_time = latest_result['latest_time'].iloc[0]
        
        # Calculate 24 hours before the latest timestamp
        query = """
        SELECT 
            timestamp, hourly_timestamp, station_id, connector_id, connector_type,
            power, status, is_occupied, is_available, is_out_of_order, tariff
        FROM utilization_data
        WHERE timestamp >= DATE_SUB(%s, INTERVAL 24 HOUR)
        ORDER BY timestamp DESC
        """
        
        df = pd.read_sql(query, engine, params=[latest_time])
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
            
            # Log the data period for debugging
            data_start = df['timestamp'].min()
            data_end = df['timestamp'].max()
            st.sidebar.info(f"📊 Utilization Data Period: {data_start.strftime('%Y-%m-%d %H:%M')} to {data_end.strftime('%Y-%m-%d %H:%M')}")
            
        return df
    except Exception as e:
        st.error(f"Error loading utilization data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_hourly_data():
    """Load latest available 24 hours of hourly aggregated data"""
    try:
        # First, find the latest timestamp in hourly data
        latest_query = "SELECT MAX(hourly_timestamp) as latest_time FROM hourly_utilization"
        latest_result = pd.read_sql(latest_query, engine)
        
        if latest_result.empty or latest_result['latest_time'].iloc[0] is None:
            st.warning("No hourly utilization data found in database")
            return pd.DataFrame()
        
        latest_time = latest_result['latest_time'].iloc[0]
        
        # Get 24 hours of data from the latest available timestamp
        query = """
        SELECT 
            hourly_timestamp, station_id, is_available, is_occupied, 
            is_out_of_order, total_connectors, occupancy_rate, availability_rate
        FROM hourly_utilization
        WHERE hourly_timestamp >= DATE_SUB(%s, INTERVAL 24 HOUR)
        ORDER BY hourly_timestamp DESC
        """
        
        df = pd.read_sql(query, engine, params=[latest_time])
        if not df.empty:
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
            
            # Log the data period for debugging
            data_start = df['hourly_timestamp'].min()
            data_end = df['hourly_timestamp'].max()
            st.sidebar.info(f"⏱️ Hourly Data Period: {data_start.strftime('%Y-%m-%d %H:%M')} to {data_end.strftime('%Y-%m-%d %H:%M')}")
            
        return df
    except Exception as e:
        st.error(f"Error loading hourly data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_sessions_data():
    """Load latest available 24 hours of charging sessions"""
    try:
        # First, find the latest end_time in sessions data
        latest_query = "SELECT MAX(end_time) as latest_time FROM charging_sessions"
        latest_result = pd.read_sql(latest_query, engine)
        
        if latest_result.empty or latest_result['latest_time'].iloc[0] is None:
            st.warning("No charging sessions found in database")
            return pd.DataFrame()
        
        latest_time = latest_result['latest_time'].iloc[0]
        
        # Get 24 hours of session data from the latest available timestamp
        query = """
        SELECT 
            s.connector_id, s.station_id, s.start_time, s.end_time,
            s.duration_hours, s.energy_kwh, s.revenue_nok,
            st.name as station_name, st.address as station_address
        FROM charging_sessions s
        LEFT JOIN charging_stations st ON s.station_id = st.id
        WHERE s.end_time >= DATE_SUB(%s, INTERVAL 24 HOUR)
        ORDER BY s.end_time DESC
        """
        
        df = pd.read_sql(query, engine, params=[latest_time])
        if not df.empty:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            
            # Log the data period for debugging
            data_start = df['end_time'].min()
            data_end = df['end_time'].max()
            st.sidebar.info(f"🔌 Sessions Data Period: {data_start.strftime('%Y-%m-%d %H:%M')} to {data_end.strftime('%Y-%m-%d %H:%M')}")
            
        return df
    except Exception as e:
        st.error(f"Error loading sessions data: {e}")
        return pd.DataFrame()

def get_latest_connector_status():
    """Get latest status for each connector from the most recent available data"""
    try:
        # First, find the latest timestamp in utilization data
        latest_query = "SELECT MAX(timestamp) as latest_time FROM utilization_data"
        latest_result = pd.read_sql(latest_query, engine)
        
        if latest_result.empty or latest_result['latest_time'].iloc[0] is None:
            return pd.DataFrame()
        
        latest_time = latest_result['latest_time'].iloc[0]
        
        # Get latest status for each connector within the last 2 hours from latest data
        query = """
        SELECT u1.*
        FROM utilization_data u1
        INNER JOIN (
            SELECT connector_id, MAX(timestamp) as max_timestamp
            FROM utilization_data
            WHERE timestamp >= DATE_SUB(%s, INTERVAL 2 HOUR)
            GROUP BY connector_id
        ) u2 ON u1.connector_id = u2.connector_id AND u1.timestamp = u2.max_timestamp
        """
        
        df = pd.read_sql(query, engine, params=[latest_time])
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Log latest connector status time
            latest_status_time = df['timestamp'].max()
            st.sidebar.success(f"🔄 Latest Status: {latest_status_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        return df
    except Exception as e:
        st.error(f"Error loading latest connector status: {e}")
        return pd.DataFrame()

# Auto-refresh functionality
def setup_auto_refresh():
    """Setup auto-refresh mechanism"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Auto-refresh every 60 seconds
    if time.time() - st.session_state.last_refresh > 60:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

def show_refresh_status():
    """Show refresh status indicator"""
    time_since_refresh = int(time.time() - st.session_state.get('last_refresh', time.time()))
    next_refresh = 60 - time_since_refresh
    
    if next_refresh > 0:
        st.markdown(
            f'<div class="refresh-info">🔄 Next refresh in {next_refresh}s</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="refresh-info">🔄 Refreshing...</div>',
            unsafe_allow_html=True
        )

# Main application
def main():
    setup_auto_refresh()
    
    # Header
    st.markdown('<h1 class="main-header">⚡ EV Charging Analytics Dashboard</h1>', unsafe_allow_html=True)
    show_refresh_status()
    
    # Load all data
    with st.spinner('Loading data...'):
        stations_df = load_stations_data()
        utilization_df = load_utilization_data()
        hourly_df = load_hourly_data()
        sessions_df = load_sessions_data()
        latest_status_df = get_latest_connector_status()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Select Dashboard",
            ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analysis", 
             "⚡ Real-time Monitor", "📋 Data Explorer"]
        )
        
        st.markdown("---")
        st.markdown("### ⏱️ Data Status")
        
        # Get data availability info
        try:
            # Check latest data timestamps
            util_latest = pd.read_sql("SELECT MAX(timestamp) as latest FROM utilization_data", engine)['latest'].iloc[0]
            sessions_latest = pd.read_sql("SELECT MAX(end_time) as latest FROM charging_sessions", engine)['latest'].iloc[0]
            hourly_latest = pd.read_sql("SELECT MAX(hourly_timestamp) as latest FROM hourly_utilization", engine)['latest'].iloc[0]
            
            if util_latest:
                age_hours = (datetime.now() - util_latest.replace(tzinfo=None)).total_seconds() / 3600
                if age_hours < 2:
                    st.success(f"🟢 Live ({age_hours:.1f}h ago)")
                elif age_hours < 24:
                    st.warning(f"🟡 Recent ({age_hours:.1f}h ago)")
                else:
                    st.info(f"🔵 Historical ({age_hours/24:.1f}d ago)")
            else:
                st.error("🔴 No data available")
                
        except Exception as e:
            st.info(f"Last updated: {get_oslo_time().strftime('%H:%M:%S')}")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Stations", len(stations_df))
            occupied = len(latest_status_df[latest_status_df['is_occupied'] == 1]) if not latest_status_df.empty else 0
            st.metric("Active", occupied)
        
        with col2:
            total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0
            st.metric("Connectors", total_connectors)
            
            if not sessions_df.empty:
                daily_revenue = sessions_df['revenue_nok'].sum()
                st.metric("Revenue", f"NOK {daily_revenue:,.0f}")
        
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    # Route to appropriate dashboard
    if page == "📊 Overview":
        show_overview_dashboard(stations_df, utilization_df, hourly_df, sessions_df, latest_status_df)
    elif page == "🗺️ Station Map":
        show_map_dashboard(stations_df, latest_status_df, sessions_df)
    elif page == "📈 Utilization Analysis":
        show_utilization_dashboard(utilization_df, hourly_df, sessions_df)
    elif page == "⚡ Real-time Monitor":
        show_realtime_dashboard(stations_df, latest_status_df, sessions_df)
    else:  # Data Explorer
        show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df)

def show_overview_dashboard(stations_df, utilization_df, hourly_df, sessions_df, latest_status_df):
    """Overview dashboard with key metrics and insights"""
    
    # Key Performance Indicators
    st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_stations = len(stations_df)
    available_stations = len(stations_df[stations_df['status'] == 'Available']) if not stations_df.empty else 0
    
    total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0
    occupied_connectors = len(latest_status_df[latest_status_df['is_occupied'] == 1]) if not latest_status_df.empty else 0
    
    avg_occupancy = (occupied_connectors / total_connectors * 100) if total_connectors > 0 else 0
    
    daily_revenue = sessions_df['revenue_nok'].sum() if not sessions_df.empty else 0
    
    with col1:
        st.metric(
            "Available Stations", 
            f"{available_stations}/{total_stations}",
            f"{(available_stations/total_stations*100):.1f}%" if total_stations > 0 else "0%"
        )
    
    with col2:
        st.metric(
            "Active Sessions", 
            occupied_connectors,
            f"{avg_occupancy:.1f}% utilization"
        )
    
    with col3:
        st.metric("Avg Occupancy Rate", f"{avg_occupancy:.1f}%")
    
    with col4:
        st.metric(
            "Daily Revenue (24h)", 
            f"NOK {daily_revenue:,.0f}",
            f"${daily_revenue/10.5:,.0f} USD"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Station status distribution
        if not stations_df.empty:
            status_counts = stations_df['status'].value_counts()
            colors = {'Available': '#2ecc71', 'Occupied': '#f39c12', 'OutOfOrder': '#e74c3c', 'Maintenance': '#9b59b6'}
            
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Station Status Distribution",
                color=status_counts.index,
                color_discrete_map=colors
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Connector type distribution
        if not stations_df.empty:
            connector_data = {
                'CCS': stations_df['ccs_connectors'].sum(),
                'CHAdeMO': stations_df['chademo_connectors'].sum(),
                'Type 2': stations_df['type2_connectors'].sum(),
                'Other': stations_df['other_connectors'].sum()
            }
            
            # Filter out zero values
            connector_data = {k: v for k, v in connector_data.items() if v > 0}
            
            fig_bar = px.bar(
                x=list(connector_data.keys()),
                y=list(connector_data.values()),
                title="Connector Type Distribution",
                labels={'x': 'Connector Type', 'y': 'Count'},
                color=list(connector_data.keys()),
                color_discrete_map={
                    'CCS': '#3498db', 'CHAdeMO': '#9b59b6', 
                    'Type 2': '#1abc9c', 'Other': '#e67e22'
                }
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # 24-Hour usage pattern
    st.subheader("📈 24-Hour Usage Pattern")
    
    if not sessions_df.empty:
        # Create hourly usage pattern
        sessions_df['hour'] = sessions_df['start_time'].dt.hour
        hourly_sessions = sessions_df.groupby('hour').agg({
            'connector_id': 'count',
            'revenue_nok': 'sum',
            'energy_kwh': 'sum'
        }).reset_index()
        hourly_sessions.columns = ['hour', 'sessions', 'revenue', 'energy']
        
        # Ensure all 24 hours are represented
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_sessions = all_hours.merge(hourly_sessions, on='hour', how='left').fillna(0)
        
        fig_usage = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Sessions bar chart
        fig_usage.add_trace(
            go.Bar(
                x=hourly_sessions['hour'],
                y=hourly_sessions['sessions'],
                name='Sessions Started',
                marker_color='rgba(52, 152, 219, 0.7)',
                yaxis='y'
            ),
            secondary_y=False
        )
        
        # Revenue line chart
        fig_usage.add_trace(
            go.Scatter(
                x=hourly_sessions['hour'],
                y=hourly_sessions['revenue'],
                name='Revenue (NOK)',
                mode='lines+markers',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        fig_usage.update_xaxes(
            title_text="Hour of Day",
            tickmode='linear',
            tick0=0,
            dtick=2
        )
        fig_usage.update_yaxes(title_text="Sessions Started", secondary_y=False)
        fig_usage.update_yaxes(title_text="Revenue (NOK)", secondary_y=True)
        fig_usage.update_layout(
            title='24-Hour Usage Pattern',
            hovermode='x unified',
            height=400,
            legend=dict(orientation='h', y=-0.2)
        )
        
        st.plotly_chart(fig_usage, use_container_width=True)
    else:
        st.info("No session data available for usage pattern")
    
    # Top performing stations
    st.subheader("🏆 Top Performing Stations (Last 24h)")
    
    if not sessions_df.empty:
        station_performance = sessions_df.groupby(['station_id', 'station_name']).agg({
            'revenue_nok': 'sum',
            'energy_kwh': 'sum',
            'connector_id': 'count'
        }).reset_index()
        station_performance.columns = ['station_id', 'station_name', 'revenue', 'energy', 'sessions']
        station_performance = station_performance.sort_values('revenue', ascending=False).head(10)
        
        if not station_performance.empty:
            # Format the data for display
            display_df = station_performance[['station_name', 'revenue', 'energy', 'sessions']].copy()
            display_df['revenue'] = display_df['revenue'].round(0)
            display_df['energy'] = display_df['energy'].round(1)
            display_df.columns = ['Station Name', 'Revenue (NOK)', 'Energy (kWh)', 'Sessions']
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No station performance data available")
    else:
        st.info("No session data available for station ranking")

def show_map_dashboard(stations_df, latest_status_df, sessions_df):
    """Interactive map dashboard"""
    st.header("🗺️ Charging Station Map")
    
    if stations_df.empty:
        st.warning("No station data available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=stations_df['status'].unique().tolist(),
            default=stations_df['status'].unique().tolist()
        )
    
    with col2:
        min_connectors = st.slider(
            "Minimum Connectors",
            min_value=1,
            max_value=int(stations_df['total_connectors'].max()),
            value=1
        )
    
    with col3:
        map_style = st.selectbox(
            "Map Style",
            ['OpenStreetMap', 'CartoDB positron', 'CartoDB dark_matter']
        )
    
    # Filter data
    filtered_stations = stations_df[
        (stations_df['status'].isin(status_filter)) &
        (stations_df['total_connectors'] >= min_connectors)
    ]
    
    if filtered_stations.empty:
        st.warning("No stations match the selected filters")
        return
    
    # Calculate center
    center_lat = filtered_stations['latitude'].mean()
    center_lon = filtered_stations['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=map_style.replace(' ', '')
    )
    
    # Calculate station revenues
    station_revenues = {}
    if not sessions_df.empty:
        station_revenues = sessions_df.groupby('station_id')['revenue_nok'].sum().to_dict()
    
    # Add markers
    for _, station in filtered_stations.iterrows():
        # Get current status
        station_connectors = latest_status_df[latest_status_df['station_id'] == station['id']] if not latest_status_df.empty else pd.DataFrame()
        
        if not station_connectors.empty:
            occupied = len(station_connectors[station_connectors['is_occupied'] == 1])
            available = len(station_connectors[station_connectors['is_available'] == 1])
            out_of_order = len(station_connectors[station_connectors['is_out_of_order'] == 1])
        else:
            occupied = 0
            available = station['total_connectors']
            out_of_order = 0
        
        # Determine marker color
        if out_of_order == station['total_connectors']:
            color = 'red'
            status_text = 'Out of Order'
        elif available > 0:
            color = 'green'
            status_text = 'Available'
        elif occupied > 0:
            color = 'orange'
            status_text = 'Fully Occupied'
        else:
            color = 'gray'
            status_text = 'Unknown'
        
        revenue = station_revenues.get(station['id'], 0)
        
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">{station['name']}</h4>
            <p><b>Status:</b> <span style="color: {color};">●</span> {status_text}</p>
            <p><b>Address:</b> {station.get('address', 'N/A')}</p>
            <p><b>Connectors:</b> {station['total_connectors']} total</p>
            <div style="margin: 10px 0;">
                <div>🟢 Available: {available}</div>
                <div>🟠 Occupied: {occupied}</div>
                <div>🔴 Out of Order: {out_of_order}</div>
            </div>
            <p><b>Types:</b></p>
            <div style="font-size: 0.9em;">
                • CCS: {station.get('ccs_connectors', 0)}<br>
                • CHAdeMO: {station.get('chademo_connectors', 0)}<br>
                • Type 2: {station.get('type2_connectors', 0)}
            </div>
            <p><b>Revenue (24h):</b> NOK {revenue:,.0f}</p>
        </div>
        """
        
        folium.Marker(
            location=[station['latitude'], station['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon='plug', prefix='fa'),
            tooltip=f"{station['name']} - {status_text}"
        ).add_to(m)
    
    # Display map
    st_folium(m, width=None, height=600)
    
    # Map statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stations Shown", len(filtered_stations))
    
    with col2:
        total_connectors = filtered_stations['total_connectors'].sum()
        st.metric("Total Connectors", total_connectors)
    
    with col3:
        total_revenue = sum(station_revenues.get(sid, 0) for sid in filtered_stations['id'])
        st.metric("Total Revenue (24h)", f"NOK {total_revenue:,.0f}")
    
    with col4:
        avg_connectors = filtered_stations['total_connectors'].mean()
        st.metric("Avg Connectors/Station", f"{avg_connectors:.1f}")

def show_utilization_dashboard(utilization_df, hourly_df, sessions_df):
    """Comprehensive utilization analysis dashboard"""
    st.header("📈 Utilization Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 24h Heatmap", "📊 Trends", "⚡ Power & Revenue", "📋 Session Analysis"])
    
    with tab1:
        st.subheader("24-Hour Utilization Heatmap")
        
        if not utilization_df.empty:
            # Prepare heatmap data
            heatmap_data = utilization_df.copy()
            heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
            heatmap_data['day_name'] = heatmap_data['timestamp'].dt.day_name()
            
            # Create pivot table
            pivot_data = heatmap_data.groupby(['day_name', 'hour'])['is_occupied'].mean().reset_index()
            pivot_table = pivot_data.pivot(index='day_name', columns='hour', values='is_occupied')
            
            # Ensure all hours are present
            for hour in range(24):
                if hour not in pivot_table.columns:
                    pivot_table[hour] = 0
            
            # Order days correctly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_table = pivot_table.reindex([d for d in day_order if d in pivot_table.index])
            pivot_table = pivot_table.fillna(0)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=list(range(24)),
                y=pivot_table.index.tolist(),
                colorscale='RdYlGn_r',
                colorbar=dict(title='Occupancy Rate'),
                hoverongaps=False,
                hovertemplate='Day: %{y}<br>Hour: %{x}<br>Occupancy: %{z:.1%}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='Weekly Utilization Pattern',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week',
                height=400,
                xaxis=dict(tickmode='linear', tick0=0, dtick=2)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Peak usage insights
            col1, col2 = st.columns(2)
            
            with col1:
                if not sessions_df.empty:
                    peak_hour_sessions = sessions_df.groupby(sessions_df['start_time'].dt.hour).size()
                    peak_hour = peak_hour_sessions.idxmax()
                    peak_count = peak_hour_sessions.max()
                    st.metric("Peak Usage Hour", f"{peak_hour}:00", f"{peak_count} sessions")
                else:
                    st.metric("Peak Usage Hour", "No data", "0 sessions")
            
            with col2:
                avg_occupancy = heatmap_data['is_occupied'].mean()
                st.metric("Average Occupancy", f"{avg_occupancy:.1%}")
        
        else:
            st.info("No utilization data available for heatmap")
    
    with tab2:
        st.subheader("Utilization Trends")
        
        if not hourly_df.empty:
            # Trend analysis
            trend_data = hourly_df.groupby('hourly_timestamp').agg({
                'is_occupied': 'sum',
                'is_available': 'sum',
                'is_out_of_order': 'sum',
                'occupancy_rate': 'mean'
            }).reset_index()
            
            fig_trend = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Connector Status Over Time', 'Average Occupancy Rate'),
                vertical_spacing=0.1
            )
            
            # Status trend
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_data['hourly_timestamp'], 
                    y=trend_data['is_occupied'],
                    name='Occupied', 
                    line=dict(color='#e74c3c', width=2),
                    fill='tonexty' if 'is_available' in trend_data.columns else None
                ),
                row=1, col=1
            )
            
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_data['hourly_timestamp'], 
                    y=trend_data['is_available'],
                    name='Available', 
                    line=dict(color='#2ecc71', width=2),
                    fill='tozeroy'
                ),
                row=1, col=1
            )
            
            # Occupancy rate trend
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_data['hourly_timestamp'], 
                    y=trend_data['occupancy_rate'] * 100,
                    name='Occupancy %', 
                    line=dict(color='#3498db', width=3),
                    mode='lines+markers'
                ),
                row=2, col=1
            )
            
            fig_trend.update_yaxes(title_text="Connector Count", row=1, col=1)
            fig_trend.update_yaxes(title_text="Occupancy %", row=2, col=1)
            fig_trend.update_layout(height=600, showlegend=True)
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Trend insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_occupancy = trend_data['occupancy_rate'].iloc[-1] * 100 if not trend_data.empty else 0
                st.metric("Current Occupancy", f"{current_occupancy:.1f}%")
            
            with col2:
                peak_occupancy = trend_data['occupancy_rate'].max() * 100 if not trend_data.empty else 0
                st.metric("Peak Occupancy (24h)", f"{peak_occupancy:.1f}%")
            
            with col3:
                avg_occupancy = trend_data['occupancy_rate'].mean() * 100 if not trend_data.empty else 0
                st.metric("Average Occupancy", f"{avg_occupancy:.1f}%")
        
        else:
            st.info("No hourly trend data available")
    
    with tab3:
        st.subheader("Power & Revenue Analysis")
        
        if not sessions_df.empty:
            # Revenue analysis by hour
            sessions_df['hour'] = sessions_df['start_time'].dt.hour
            hourly_revenue = sessions_df.groupby('hour').agg({
                'revenue_nok': ['sum', 'mean'],
                'energy_kwh': ['sum', 'mean'],
                'connector_id': 'count'
            }).round(2)
            
            hourly_revenue.columns = ['total_revenue', 'avg_revenue', 'total_energy', 'avg_energy', 'sessions']
            hourly_revenue = hourly_revenue.reset_index()
            
            # Revenue pattern chart
            fig_revenue = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_revenue.add_trace(
                go.Bar(
                    x=hourly_revenue['hour'],
                    y=hourly_revenue['total_revenue'],
                    name='Total Revenue (NOK)',
                    marker_color='rgba(46, 204, 113, 0.7)'
                ),
                secondary_y=False
            )
            
            fig_revenue.add_trace(
                go.Scatter(
                    x=hourly_revenue['hour'],
                    y=hourly_revenue['sessions'],
                    name='Sessions',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )
            
            fig_revenue.update_xaxes(title_text="Hour of Day", tickmode='linear', tick0=0, dtick=2)
            fig_revenue.update_yaxes(title_text="Revenue (NOK)", secondary_y=False)
            fig_revenue.update_yaxes(title_text="Sessions Count", secondary_y=True)
            fig_revenue.update_layout(
                title="Hourly Revenue and Session Pattern",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
            
            # Power and revenue metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_energy = sessions_df['energy_kwh'].sum()
                st.metric("Total Energy (24h)", f"{total_energy:,.0f} kWh")
            
            with col2:
                total_revenue = sessions_df['revenue_nok'].sum()
                st.metric("Total Revenue (24h)", f"NOK {total_revenue:,.0f}")
            
            with col3:
                avg_session_energy = sessions_df['energy_kwh'].mean()
                st.metric("Avg Energy/Session", f"{avg_session_energy:.1f} kWh")
            
            with col4:
                avg_session_revenue = sessions_df['revenue_nok'].mean()
                st.metric("Avg Revenue/Session", f"NOK {avg_session_revenue:.0f}")
            
            # Energy vs Revenue scatter
            if len(sessions_df) > 1:
                fig_scatter = px.scatter(
                    sessions_df,
                    x='energy_kwh',
                    y='revenue_nok',
                    title='Energy vs Revenue Relationship',
                    labels={'energy_kwh': 'Energy (kWh)', 'revenue_nok': 'Revenue (NOK)'},
                    trendline='ols',
                    color_continuous_scale='viridis'
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        else:
            st.info("No session data available for power and revenue analysis")
    
    with tab4:
        st.subheader("Session Analysis")
        
        if not sessions_df.empty:
            # Session duration analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig_duration = px.histogram(
                    sessions_df,
                    x='duration_hours',
                    nbins=20,
                    title='Session Duration Distribution',
                    labels={'duration_hours': 'Duration (hours)', 'count': 'Sessions'}
                )
                fig_duration.update_traces(marker_color='#3498db')
                fig_duration.update_layout(height=400)
                st.plotly_chart(fig_duration, use_container_width=True)
            
            with col2:
                fig_revenue_dist = px.histogram(
                    sessions_df,
                    x='revenue_nok',
                    nbins=20,
                    title='Revenue Distribution',
                    labels={'revenue_nok': 'Revenue (NOK)', 'count': 'Sessions'}
                )
                fig_revenue_dist.update_traces(marker_color='#2ecc71')
                fig_revenue_dist.update_layout(height=400)
                st.plotly_chart(fig_revenue_dist, use_container_width=True)
            
            # Session metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sessions = len(sessions_df)
                st.metric("Total Sessions (24h)", total_sessions)
            
            with col2:
                avg_duration = sessions_df['duration_hours'].mean() * 60
                st.metric("Avg Duration", f"{avg_duration:.0f} min")
            
            with col3:
                median_revenue = sessions_df['revenue_nok'].median()
                st.metric("Median Revenue", f"NOK {median_revenue:.0f}")
            
            with col4:
                max_energy = sessions_df['energy_kwh'].max()
                st.metric("Max Energy", f"{max_energy:.1f} kWh")
            
            # Top connectors by revenue
            st.subheader("🏆 Top Performing Connectors")
            
            top_connectors = sessions_df.groupby('connector_id').agg({
                'revenue_nok': 'sum',
                'energy_kwh': 'sum',
                'duration_hours': ['count', 'mean']
            }).round(2)
            
            top_connectors.columns = ['total_revenue', 'total_energy', 'session_count', 'avg_duration']
            top_connectors = top_connectors.sort_values('total_revenue', ascending=False).head(10)
            top_connectors = top_connectors.reset_index()
            
            # Format for display
            display_df = top_connectors[['connector_id', 'total_revenue', 'total_energy', 'session_count']].copy()
            display_df.columns = ['Connector ID', 'Revenue (NOK)', 'Energy (kWh)', 'Sessions']
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        else:
            st.info("No session data available for analysis")

def show_realtime_dashboard(stations_df, latest_status_df, sessions_df):
    """Real-time monitoring dashboard"""
    st.header("⚡ Real-time Station Monitor")
    
    # Current status overview
    col1, col2, col3, col4 = st.columns(4)
    
    if not latest_status_df.empty:
        current_available = len(latest_status_df[latest_status_df['is_available'] == 1])
        current_occupied = len(latest_status_df[latest_status_df['is_occupied'] == 1])
        current_out_of_order = len(latest_status_df[latest_status_df['is_out_of_order'] == 1])
        total_connectors = len(latest_status_df)
    else:
        current_available = 0
        current_occupied = 0
        current_out_of_order = 0
        total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0
    
    with col1:
        st.metric("🟢 Available", f"{current_available}")
    
    with col2:
        st.metric("🟠 Occupied", f"{current_occupied}")
    
    with col3:
        st.metric("🔴 Out of Order", f"{current_out_of_order}")
    
    with col4:
        utilization = (current_occupied / total_connectors * 100) if total_connectors > 0 else 0
        st.metric("📊 Utilization", f"{utilization:.1f}%")
    
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search stations by name or ID", "")
    
    with col2:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Available", "Occupied", "Out of Order"]
        )
    
    # Filter stations
    if search_term and not stations_df.empty:
        filtered_stations = stations_df[
            stations_df['name'].str.contains(search_term, case=False, na=False) |
            stations_df['id'].str.contains(search_term, case=False, na=False)
        ]
    else:
        filtered_stations = stations_df
    
    # Real-time station grid
    if not filtered_stations.empty:
        st.subheader("📍 Station Status Grid")
        
        # Display stations in grid format
        stations_per_row = 4
        for i in range(0, len(filtered_stations), stations_per_row):
            cols = st.columns(stations_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(filtered_stations):
                    station = filtered_stations.iloc[i + j]
                    
                    # Get current status
                    station_connectors = latest_status_df[latest_status_df['station_id'] == station['id']] if not latest_status_df.empty else pd.DataFrame()
                    
                    if not station_connectors.empty:
                        occupied = len(station_connectors[station_connectors['is_occupied'] == 1])
                        available = len(station_connectors[station_connectors['is_available'] == 1])
                        out_of_order = len(station_connectors[station_connectors['is_out_of_order'] == 1])
                    else:
                        occupied = 0
                        available = station['total_connectors']
                        out_of_order = 0
                    
                    # Determine card color
                    if out_of_order == station['total_connectors']:
                        card_color = "#e74c3c"
                        icon = "🔴"
                        status_text = "Out of Order"
                    elif occupied == station['total_connectors']:
                        card_color = "#f39c12"
                        icon = "🟠"
                        status_text = "Fully Occupied"
                    elif available > 0:
                        card_color = "#2ecc71"
                        icon = "🟢"
                        status_text = "Available"
                    else:
                        card_color = "#95a5a6"
                        icon = "⚪"
                        status_text = "Unknown"
                    
                    # Skip if status filter doesn't match
                    if status_filter != "All":
                        if status_filter == "Available" and available == 0:
                            continue
                        elif status_filter == "Occupied" and occupied == 0:
                            continue
                        elif status_filter == "Out of Order" and out_of_order == 0:
                            continue
                    
                    with col:
                        st.markdown(f"""
                        <div class="station-card" style="
                            background: linear-gradient(135deg, {card_color}20 0%, {card_color}10 100%);
                            border: 2px solid {card_color};
                            min-height: 180px;
                        ">
                            <h4 style="margin: 0 0 10px 0; font-size: 0.9em; color: {card_color};">
                                {icon} {station['name'][:25]}{'...' if len(station['name']) > 25 else ''}
                            </h4>
                            <p style="margin: 5px 0; font-size: 0.85em;"><b>{status_text}</b></p>
                            <p style="margin: 5px 0; font-size: 0.8em;">
                                🟢 {available} | 🟠 {occupied} | 🔴 {out_of_order}
                            </p>
                            <p style="margin: 5px 0; font-size: 0.8em;">⚡ {station['total_connectors']} total</p>
                            <p style="margin: 5px 0; font-size: 0.7em; color: #666;">
                                📍 {str(station.get('address', 'No address'))[:30]}{'...' if len(str(station.get('address', ''))) > 30 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Recent activity feed
    st.markdown("---")
    st.subheader("📰 Recent Activity")
    
    if not sessions_df.empty:
        recent_sessions = sessions_df.sort_values('start_time', ascending=False).head(15)
        
        for _, session in recent_sessions.iterrows():
            # Calculate time ago
            time_ago = (get_oslo_time().replace(tzinfo=None) - session['start_time']).total_seconds() / 60
            
            if time_ago < 60:
                time_str = f"{int(time_ago)}m ago"
            elif time_ago < 1440:
                time_str = f"{int(time_ago / 60)}h ago"
            else:
                time_str = f"{int(time_ago / 1440)}d ago"
            
            station_name = session.get('station_name', 'Unknown Station')
            
            st.markdown(
                f"🔌 **{station_name}** - Duration: {session['duration_hours']*60:.0f}min, "
                f"Energy: {session['energy_kwh']:.1f}kWh, Revenue: NOK {session['revenue_nok']:.0f} "
                f"*({time_str})*"
            )
    else:
        st.info("No recent session activity to display")

def show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df):
    """Data explorer with filtering and export capabilities"""
    st.header("📋 Data Explorer")
    
    # Dataset selector
    dataset = st.selectbox(
        "Select Dataset",
        ["Charging Stations", "Utilization Data", "Hourly Data", "Charging Sessions"]
    )
    
    # Get appropriate dataframe
    if dataset == "Charging Stations":
        df = stations_df.copy()
        description = "All charging stations with specifications and current status"
    elif dataset == "Utilization Data":
        df = utilization_df.copy()
        description = "Real-time connector utilization data (last 24 hours)"
    elif dataset == "Hourly Data":
        df = hourly_df.copy()
        description = "Hourly aggregated utilization statistics"
    else:
        df = sessions_df.copy()
        description = "Completed charging sessions (last 24 hours)"
    
    st.markdown(f"*{description}*")
    
    if df.empty:
        st.warning(f"No data available for {dataset}")
        return
    
    # Filtering options
    st.subheader("🔍 Data Filters")
    
    col1, col2, col3 = st.columns(3)
    
    filters_applied = []
    
    # Dataset-specific filters
    if dataset == "Charging Stations":
        with col1:
            if 'status' in df.columns:
                status_values = df['status'].dropna().unique().tolist()
                selected_status = st.multiselect("Status", status_values, default=status_values)
                if len(selected_status) < len(status_values):
                    df = df[df['status'].isin(selected_status)]
                    filters_applied.append(f"Status: {', '.join(selected_status)}")
        
        with col2:
            if 'total_connectors' in df.columns:
                min_conn, max_conn = int(df['total_connectors'].min()), int(df['total_connectors'].max())
                conn_range = st.slider("Total Connectors", min_conn, max_conn, (min_conn, max_conn))
                df = df[(df['total_connectors'] >= conn_range[0]) & (df['total_connectors'] <= conn_range[1])]
                if conn_range != (min_conn, max_conn):
                    filters_applied.append(f"Connectors: {conn_range[0]}-{conn_range[1]}")
        
        with col3:
            if 'operator' in df.columns:
                operators = df['operator'].dropna().unique().tolist()
                selected_operators = st.multiselect("Operator", operators, default=operators)
                if len(selected_operators) < len(operators):
                    df = df[df['operator'].isin(selected_operators)]
                    filters_applied.append(f"Operators: {', '.join(selected_operators)}")
    
    elif dataset == "Charging Sessions":
        with col1:
            if 'revenue_nok' in df.columns:
                min_rev, max_rev = float(df['revenue_nok'].min()), float(df['revenue_nok'].max())
                rev_range = st.slider("Revenue (NOK)", min_rev, max_rev, (min_rev, max_rev))
                df = df[(df['revenue_nok'] >= rev_range[0]) & (df['revenue_nok'] <= rev_range[1])]
                if rev_range != (min_rev, max_rev):
                    filters_applied.append(f"Revenue: NOK {rev_range[0]:.0f}-{rev_range[1]:.0f}")
        
        with col2:
            if 'duration_hours' in df.columns:
                min_dur, max_dur = float(df['duration_hours'].min()), float(df['duration_hours'].max())
                dur_range = st.slider("Duration (hours)", min_dur, max_dur, (min_dur, max_dur))
                df = df[(df['duration_hours'] >= dur_range[0]) & (df['duration_hours'] <= dur_range[1])]
                if dur_range != (min_dur, max_dur):
                    filters_applied.append(f"Duration: {dur_range[0]:.1f}-{dur_range[1]:.1f}h")
        
        with col3:
            if 'energy_kwh' in df.columns:
                min_energy, max_energy = float(df['energy_kwh'].min()), float(df['energy_kwh'].max())
                energy_range = st.slider("Energy (kWh)", min_energy, max_energy, (min_energy, max_energy))
                df = df[(df['energy_kwh'] >= energy_range[0]) & (df['energy_kwh'] <= energy_range[1])]
                if energy_range != (min_energy, max_energy):
                    filters_applied.append(f"Energy: {energy_range[0]:.1f}-{energy_range[1]:.1f} kWh")
    
    # Display applied filters
    if filters_applied:
        st.info(f"**Active filters:** {' | '.join(filters_applied)}")
    
    # Data summary
    st.subheader("📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        if dataset == "Charging Sessions" and 'revenue_nok' in df.columns:
            total_revenue = df['revenue_nok'].sum()
            st.metric("Total Revenue", f"NOK {total_revenue:,.0f}")
        else:
            non_null_pct = (df.count().sum() / (len(df) * len(df.columns)) * 100)
            st.metric("Data Completeness", f"{non_null_pct:.1f}%")
    
    # Data preview
    st.subheader("📋 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_columns = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=df.columns.tolist()[:10] if len(df.columns) > 10 else df.columns.tolist()
        )
    
    with col2:
        n_rows = st.number_input("Rows to display", min_value=10, max_value=1000, value=50)
    
    with col3:
        if show_columns:
            sort_column = st.selectbox("Sort by", show_columns)
            sort_ascending = st.checkbox("Ascending", value=True)
        else:
            sort_column = None
            sort_ascending = True
    
    # Apply column selection and sorting
    if show_columns:
        display_df = df[show_columns].copy()
        
        if sort_column and sort_column in display_df.columns:
            display_df = display_df.sort_values(sort_column, ascending=sort_ascending)
        
        # Display data
        st.dataframe(
            display_df.head(n_rows),
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{dataset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary statistics
            if st.button("📊 Generate Summary"):
                summary_stats = display_df.describe(include='all').to_string()
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_stats,
                    file_name=f"{dataset.lower().replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            # Basic insights
            if dataset == "Charging Sessions" and not display_df.empty:
                total_sessions = len(display_df)
                total_revenue = display_df['revenue_nok'].sum() if 'revenue_nok' in display_df.columns else 0
                avg_duration = display_df['duration_hours'].mean() * 60 if 'duration_hours' in display_df.columns else 0
                
                insights = f"""
Data Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Sessions: {total_sessions:,}
Total Revenue: NOK {total_revenue:,.2f}
Average Duration: {avg_duration:.1f} minutes
Date Range: {df['start_time'].min()} to {df['start_time'].max()}
                """
                
                st.download_button(
                    label="📥 Download Insights",
                    data=insights,
                    file_name=f"session_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
    else:
        st.warning("Please select at least one column to display")

# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: #666; padding: 20px; font-size: 0.9em;">
            <p>⚡ <b>EV Charging Analytics Dashboard</b> | Connected to Railway MySQL</p>
            <p>🔄 Auto-refresh enabled (60s) | 📊 Last updated: {get_oslo_time().strftime('%Y-%m-%d %H:%M:%S')} CET</p>
            <p>🚀 Built with Streamlit | 📡 Real-time data from utilization_data, charging_sessions, hourly_utilization</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()
