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


def get_engine():
    """Create SQLAlchemy engine with connection pooling"""
    try:
        return create_engine(
            get_connection_string(),
            pool_size=3,
            max_overflow=5,
            pool_timeout=30,
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
try:
    engine = get_engine()
    MYSQL_AVAILABLE = engine is not None
except Exception as e:
    engine = None
    MYSQL_AVAILABLE = False


# MySQL Data Loading Functions
def load_from_mysql(table_name, where_clause=None, limit=None):
    """Load data from MySQL table"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql(query, engine)

        # Parse datetime columns
        datetime_columns = ['timestamp', 'hourly_timestamp', 'start_time', 'end_time', 'created_at', 'last_updated']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df

    except Exception as e:
        logger.error(f"Error loading from MySQL table {table_name}: {e}")
        return pd.DataFrame()


def get_latest_utilization():
    """Get the latest utilization status for each connector"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        # Try to use a view if it exists, otherwise get latest records manually
        query = """
        SELECT u1.* FROM utilization_data u1
        INNER JOIN (
            SELECT connector_id, MAX(timestamp) as max_timestamp
            FROM utilization_data
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
            GROUP BY connector_id
        ) u2 ON u1.connector_id = u2.connector_id AND u1.timestamp = u2.max_timestamp
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error getting latest utilization: {e}")
        # Fallback to recent data
        return load_from_mysql('utilization_data',
                               where_clause="timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)")


def get_historical_utilization(hours=24):
    """Get ALL historical utilization data for the specified time period"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        query = """
        SELECT * FROM utilization_data
        WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, engine, params=(hours,))
        
        # Parse datetime columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'hourly_timestamp' in df.columns:
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
            
        return df
    except Exception as e:
        logger.error(f"Error getting historical utilization: {e}")
        return pd.DataFrame()


def get_hourly_stats(hours=24):
    """Get hourly statistics for the last N hours"""
    if not MYSQL_AVAILABLE:
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
        WHERE hourly_timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        GROUP BY hourly_timestamp
        ORDER BY hourly_timestamp DESC
        """
        return pd.read_sql(query, engine, params=(hours,))
    except Exception as e:
        logger.error(f"Error getting hourly stats: {e}")
        # Fallback to utilization data aggregation
        return load_from_mysql('utilization_data',
                               where_clause=f"timestamp >= DATE_SUB(NOW(), INTERVAL {hours} HOUR)")


def get_recent_sessions(hours=24):
    """Get charging sessions from the last N hours"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        query = """
        SELECT 
            cs.*,
            st.name as station_name,
            st.address as station_address
        FROM charging_sessions cs
        LEFT JOIN charging_stations st ON cs.station_id = st.id
        WHERE cs.end_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY cs.end_time DESC
        """
        return pd.read_sql(query, engine, params=[hours])
    except Exception as e:
        logger.error(f"Error getting recent sessions: {e}")
        # Fallback to sessions table without join
        return load_from_mysql('charging_sessions',
                               where_clause=f"end_time >= DATE_SUB(NOW(), INTERVAL {hours} HOUR)")


def test_mysql_connection():
    """Test MySQL connection and return status"""
    if not MYSQL_AVAILABLE or engine is None:
        return False, "Engine not available"

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return True, "Connected successfully"
    except Exception as e:
        return False, str(e)


# Page configuration
st.set_page_config(
    page_title="EV Charging Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme as default
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .refresh-timer {
        position: fixed;
        top: 70px;
        right: 20px;
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 999;
    }
    .station-card {
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        height: 180px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Import timezone handling
import pytz

# Set CEST timezone
CEST = pytz.timezone('Europe/Oslo')

# Session state initialization with better navigation handling
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Changed refresh interval from 30 to 60 seconds
REFRESH_INTERVAL = 60


# More efficient data loading with better caching strategy
@st.cache_data(ttl=60, show_spinner=False)  # Increased TTL to 60 seconds
def load_stations_data():
    """Load charging stations data from MySQL"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = load_from_mysql('charging_stations')
        if df.empty:
            st.warning("⚠️ No stations data found in database.")
            st.stop()

        # Ensure numeric columns are properly typed
        numeric_cols = ['latitude', 'longitude', 'total_connectors', 'ccs_connectors',
                        'chademo_connectors', 'type2_connectors', 'other_connectors']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Error loading stations from database: {e}")
        st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_utilization_data():
    """Load latest utilization data from MySQL"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = get_latest_utilization()
        if df.empty:
            st.warning("⚠️ No utilization data found in database.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error loading utilization data: {e}")
        st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_historical_utilization_data(hours=24):
    """Load historical utilization data for heatmap and analytics"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = get_historical_utilization(hours)
        if df.empty:
            st.warning("⚠️ No historical utilization data found in database.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"❌ Error loading historical utilization data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)  # Cache hourly data for 5 minutes
def load_hourly_data():
    """Load hourly aggregated data from MySQL"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = get_hourly_stats(hours=168)  # Last 7 days
        if df.empty:
            st.warning("⚠️ No hourly data found in database.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error loading hourly data: {e}")
        st.stop()


@st.cache_data(ttl=300, show_spinner=False)  # Cache sessions for 5 minutes
def load_sessions_data():
    """Load recent charging sessions from MySQL"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = get_recent_sessions(hours=168)  # Last 7 days
        if df.empty:
            st.warning("⚠️ No sessions data found in database.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error loading sessions data: {e}")
        st.stop()


# Helper function to format CEST time
def format_cest_time(dt):
    """Format datetime to CEST timezone"""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(CEST)


# Better auto-refresh functionality with navigation preservation
def check_auto_refresh():
    """Check if it's time to auto-refresh (every 60 seconds) without disrupting navigation"""
    current_time = time.time()
    
    if (st.session_state.auto_refresh_enabled and
            current_time - st.session_state.last_refresh > REFRESH_INTERVAL):
        st.session_state.last_refresh = current_time
        # Only clear data cache, not navigation state
        load_stations_data.clear()
        load_utilization_data.clear()
        load_historical_utilization_data.clear()
        # Don't clear hourly and sessions cache as frequently
        if current_time % 300 < 60:  # Clear every 5 minutes
            load_hourly_data.clear()
            load_sessions_data.clear()
        st.rerun()


def show_refresh_timer():
    """Show countdown to next refresh"""
    if st.session_state.auto_refresh_enabled:
        time_since_refresh = int(time.time() - st.session_state.last_refresh)
        time_to_refresh = REFRESH_INTERVAL - time_since_refresh

        if time_to_refresh > 0:
            st.markdown(
                f'<div class="refresh-timer">🔄 Auto-refresh in {time_to_refresh}s</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="refresh-timer">🔄 Refreshing...</div>',
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


# Error handling for deployment - database required
def handle_deployment_errors():
    """Handle database connection requirements"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ **Database Connection Required**")
        st.markdown("""
        This dashboard requires a MySQL database connection. Please:

        1. **Set up MySQL database** (Railway, PlanetScale, AWS RDS, etc.)
        2. **Configure credentials** in Streamlit secrets or environment variables:
           - `MYSQL_HOST`
           - `MYSQL_USER` 
           - `MYSQL_PASSWORD`
           - `MYSQL_DATABASE`
           - OR single `DATABASE_URL`
        3. **Ensure database tables exist** with the correct schema
        4. **Run ETL pipeline** to populate data

        See deployment guide for detailed instructions.
        """)
        st.stop()

    try:
        # Test database connectivity
        connection_status, message = test_mysql_connection()
        if not connection_status:
            st.error(f"❌ **Database Connection Failed**: {message}")
            st.stop()
    except Exception as e:
        st.error(f"❌ **Database Error**: {e}")
        st.stop()


# Main dashboard with better navigation handling
def main():
    # Check for auto-refresh but preserve navigation state
    check_auto_refresh()

    # Title with updated refresh info
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h1 style="color: #1f77b4; margin-bottom: 1rem;">⚡ EV Charging Analytics Dashboard</h1>
        <p>📊 MySQL Backend | Real-time Updates every 60 seconds</p>
        <p>Data updates automatically | Last refresh: {} CEST</p>
    </div>
    """.format(get_cest_time().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

    # Show refresh timer
    show_refresh_timer()

    # Load data with better error handling and spinner control
    if not st.session_state.data_loaded:
        with st.spinner('Loading initial data...'):
            try:
                stations_df = load_stations_data()
                utilization_df = load_utilization_data()
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading initial data: {e}")
                st.stop()
    else:
        # Load data silently after initial load
        try:
            stations_df = load_stations_data()
            utilization_df = load_utilization_data()
        except Exception as e:
            st.error(f"Error refreshing data: {e}")
            st.stop()

    # Sidebar with better navigation state management
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=EV+Charging+Analytics", width=300)
        st.markdown("### 🔌 Navigation")

        # Use session state for navigation to prevent page switching issues
        page_options = ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analytics",
                        "⚡ Real-time Monitor", "📋 Data Explorer"]

        page = st.radio(
            "Select Dashboard",
            options=page_options,
            index=page_options.index(
                st.session_state.current_page) if st.session_state.current_page in page_options else 0,
            key="navigation_radio"
        )

        # Update session state when page changes
        if page != st.session_state.current_page:
            st.session_state.current_page = page

        st.markdown("---")
        st.markdown("### 🕐 Last Updated (CEST)")
        st.info(f"{get_cest_time().strftime('%Y-%m-%d %H:%M:%S')}")

        # Auto-refresh controls
        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        auto_refresh = st.checkbox("Real-time refresh (60s)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh

        if st.button("🔄 Refresh Now"):
            # Clear all caches and reset timer
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.data_loaded = False
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Stations", len(stations_df))

        active_sessions = len(utilization_df[utilization_df['is_occupied'] == 1]) if not utilization_df.empty else 0
        st.metric("Active Sessions", active_sessions)

        total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0
        st.metric("Total Connectors", total_connectors)

        # Database connection status
        st.markdown("---")
        st.markdown("### 🗄️ Database Status")
        if MYSQL_AVAILABLE and engine is not None:
            connection_status, message = test_mysql_connection()
            if connection_status:
                st.success("✅ MySQL Connected")
            else:
                st.error(f"❌ MySQL Error: {message}")
        else:
            st.error("❌ No Database Connection")

        # Data freshness indicator
        if not utilization_df.empty and 'timestamp' in utilization_df.columns:
            latest_data = utilization_df['timestamp'].max()
            if latest_data.tzinfo is None:
                latest_data = pytz.utc.localize(latest_data)
            latest_data_cest = latest_data.astimezone(CEST)
            data_age = (get_cest_time() - latest_data_cest).total_seconds() / 60

            if data_age < 5:
                st.success(f"📡 Data current")
            elif data_age < 30:
                st.warning(f"📡 Data {data_age:.1f}m old")
            else:
                st.error(f"📡 Data {data_age:.1f}m old")

    # Page routing with better state management
    # Load additional data based on page requirements
    hourly_df = pd.DataFrame()
    sessions_df = pd.DataFrame()
    historical_util_df = pd.DataFrame()

    # Load sessions data for pages that need revenue information
    if st.session_state.current_page in ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analytics",
                                         "⚡ Real-time Monitor", "📋 Data Explorer"]:
        sessions_df = load_sessions_data()

    # Load hourly data for analytics pages
    if st.session_state.current_page in ["📊 Overview", "📈 Utilization Analytics", "📋 Data Explorer"]:
        hourly_df = load_hourly_data()

    # Load historical utilization data for analytics
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
    """Show comprehensive overview dashboard"""
    st.header("📊 Overview Dashboard")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_stations = len(stations_df)
        available_stations = len(stations_df[stations_df['status'] == 'Available'])
        st.metric(
            "Available Stations",
            f"{available_stations}/{total_stations}",
            f"{(available_stations / total_stations * 100):.1f}%" if total_stations > 0 else "0%"
        )

    with col2:
        total_connectors = stations_df['total_connectors'].sum()
        occupied_connectors = len(utilization_df[utilization_df['is_occupied'] == 1]) if not utilization_df.empty else 0
        st.metric(
            "Active Sessions",
            occupied_connectors,
            f"{(occupied_connectors / total_connectors * 100):.1f}% utilization" if total_connectors > 0 else "0%"
        )

    with col3:
        if not hourly_df.empty and 'avg_occupancy_rate' in hourly_df.columns:
            avg_occupancy = hourly_df['avg_occupancy_rate'].mean() * 100
        else:
            avg_occupancy = (occupied_connectors / total_connectors * 100) if total_connectors > 0 else 0

        st.metric(
            "Avg Occupancy Rate",
            f"{avg_occupancy:.1f}%"
        )

    with col4:
        if not sessions_df.empty and 'revenue_nok' in sessions_df.columns:
            # Calculate daily revenue (last 24 hours)
            last_24h = datetime.now() - timedelta(hours=24)
            recent_sessions = sessions_df[sessions_df['end_time'] >= last_24h]
            daily_revenue_nok = recent_sessions['revenue_nok'].sum()
            daily_revenue_usd = daily_revenue_nok / 10.5  # Convert to USD
        else:
            daily_revenue_usd = 0
            daily_revenue_nok = 0

        st.metric(
            "Daily Revenue (24h)",
            f"${daily_revenue_usd:,.0f}",
            help=f"NOK {daily_revenue_nok:,.0f}"
        )

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Station status pie chart
        if not stations_df.empty:
            status_counts = stations_df['status'].value_counts()
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Station Status Distribution",
                color_discrete_map={
                    'Available': '#2ecc71',
                    'Occupied': '#f39c12',
                    'OutOfOrder': '#e74c3c'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Connector type distribution
        if not stations_df.empty:
            connector_data = {
                'CCS': stations_df['ccs_connectors'].sum(),
                'CHAdeMO': stations_df['chademo_connectors'].sum(),
                'Type 2': stations_df['type2_connectors'].sum()
            }
            fig_bar = px.bar(
                x=list(connector_data.keys()),
                y=list(connector_data.values()),
                title="Connector Type Distribution",
                labels={'x': 'Connector Type', 'y': 'Count'},
                color=list(connector_data.keys()),
                color_discrete_map={
                    'CCS': '#3498db',
                    'CHAdeMO': '#9b59b6',
                    'Type 2': '#1abc9c'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Hourly utilization pattern
    st.markdown("---")

    if not sessions_df.empty or not utilization_df.empty:
        # Create comprehensive hourly pattern chart
        fig_line = make_subplots(specs=[[{"secondary_y": True}]])

        # Add session counts if available
        if not sessions_df.empty and 'start_time' in sessions_df.columns:
            sessions_df_copy = sessions_df.copy()
            sessions_df_copy['hour'] = sessions_df_copy['start_time'].dt.hour
            sessions_per_hour = sessions_df_copy.groupby('hour').size()

            # Ensure all hours are represented
            all_hours = pd.Series(0, index=range(24))
            all_hours.update(sessions_per_hour)
            sessions_per_hour = all_hours

            fig_line.add_trace(
                go.Bar(
                    x=sessions_per_hour.index,
                    y=sessions_per_hour.values,
                    name='Sessions Started',
                    marker_color='#3498db',
                    opacity=0.7
                ),
                secondary_y=False
            )

        # Add occupancy data
        if not utilization_df.empty and 'timestamp' in utilization_df.columns:
            hourly_occupancy = utilization_df.groupby(utilization_df['timestamp'].dt.hour)[
                'is_occupied'].sum().reset_index()

            fig_line.add_trace(
                go.Scatter(
                    x=hourly_occupancy['timestamp'],
                    y=hourly_occupancy['is_occupied'],
                    name='Occupied Connectors',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )

        # Update layout
        fig_line.update_xaxes(title_text="Hour of Day", tickmode='linear', tick0=0, dtick=1)
        fig_line.update_yaxes(title_text="Sessions Started", secondary_y=False)
        fig_line.update_yaxes(title_text="Occupied Connectors", secondary_y=True)
        fig_line.update_layout(
            title='24-Hour Usage Pattern',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.2)
        )

        st.plotly_chart(fig_line, use_container_width=True)

    # Station performance table
    st.markdown("---")
    st.subheader("🏆 Top Performing Stations")

    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        # Aggregate revenue by station
        station_revenue = sessions_df.groupby('station_id').agg({
            'revenue_nok': 'sum',
            'energy_kwh': 'sum',
            'connector_id': 'count'
        }).reset_index()
        station_revenue.columns = ['station_id', 'total_revenue', 'total_energy', 'session_count']

        # Merge with station info
        station_performance = station_revenue.merge(
            stations_df[['id', 'name', 'address']],
            left_on='station_id',
            right_on='id',
            how='left'
        )

        # Sort by revenue and get top 10
        station_performance = station_performance.sort_values('total_revenue', ascending=False).head(10)

        if not station_performance.empty:
            st.dataframe(
                station_performance[['name', 'address', 'total_revenue', 'total_energy', 'session_count']].rename(
                    columns={
                        'name': 'Station Name',
                        'address': 'Address',
                        'total_revenue': 'Revenue (NOK)',
                        'total_energy': 'Energy (kWh)',
                        'session_count': 'Sessions'
                    }),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No session data available for station performance ranking")


def show_station_map(stations_df, utilization_df, sessions_df):
    """Show interactive station map with comprehensive data"""
    st.header("🗺️ Charging Station Map")

    # Calculate station revenue if sessions data available
    station_revenue = {}
    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        station_revenue = sessions_df.groupby('station_id')['revenue_nok'].sum().to_dict()

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=['Available', 'Occupied', 'OutOfOrder'],
            default=['Available', 'Occupied', 'OutOfOrder'],
            key="map_status_filter"
        )

    with col2:
        connector_filter = st.slider(
            "Minimum Connectors",
            min_value=1,
            max_value=int(stations_df['total_connectors'].max()) if not stations_df.empty else 10,
            value=1,
            key="map_connector_filter"
        )

    with col3:
        map_style = st.selectbox(
            "Map Style",
            ['OpenStreetMap', 'CartoDB dark_matter'],
            key="map_style_selector"
        )

    # Filter data
    filtered_df = stations_df[
        (stations_df['status'].isin(status_filter)) &
        (stations_df['total_connectors'] >= connector_filter)
        ] if not stations_df.empty else pd.DataFrame()

    # Create map
    if len(filtered_df) > 0:
        # Calculate center
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()

        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=map_style.replace(' ', '')
        )

        # Add markers
        for _, station in filtered_df.iterrows():
            # Get connector status from utilization data
            if not utilization_df.empty:
                station_connectors = utilization_df[utilization_df['station_id'] == station['id']]
                occupied_count = len(station_connectors[station_connectors['is_occupied'] == 1])
                available_count = len(station_connectors[station_connectors['is_available'] == 1])
                out_of_order_count = len(station_connectors[station_connectors['is_out_of_order'] == 1])
            else:
                occupied_count = 0
                available_count = station['total_connectors']
                out_of_order_count = 0

            # Determine marker color
            if out_of_order_count == station['total_connectors']:
                color = 'red'
                status_text = 'Out of Order'
            elif available_count > 0:
                color = 'green'
                status_text = 'Available'
            elif occupied_count > 0:
                color = 'orange'
                status_text = 'Fully Occupied'
            else:
                color = 'gray'
                status_text = 'Unknown'

            # Get revenue for this station
            revenue = station_revenue.get(station['id'], 0)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>{station['name']}</h4>
                <p><b>Status:</b> {status_text}</p>
                <p><b>Address:</b> {station.get('address', 'N/A')}</p>
                <p><b>Connectors:</b> {station['total_connectors']}</p>
                <p><b>Connector Status:</b><br>
                   - Available: {available_count}<br>
                   - Occupied: {occupied_count}<br>
                   - Out of Order: {out_of_order_count}</p>
                <p><b>Types:</b><br>
                   - CCS: {station.get('ccs_connectors', 0)}<br>
                   - CHAdeMO: {station.get('chademo_connectors', 0)}<br>
                   - Type 2: {station.get('type2_connectors', 0)}</p>
                <p><b>Total Revenue:</b> NOK {revenue:,.0f}</p>
            </div>
            """

            folium.Marker(
                location=[station['latitude'], station['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon='plug', prefix='fa'),
                tooltip=f"{station['name']} - {status_text} - NOK {revenue:,.0f}"
            ).add_to(m)

        # Display map
        st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"], key="station_map")

        # Statistics below map
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stations Shown", len(filtered_df))
        with col2:
            st.metric("Total Connectors", filtered_df['total_connectors'].sum())
        with col3:
            total_revenue = sum(station_revenue.get(sid, 0) for sid in filtered_df['id'])
            st.metric("Total Revenue", f"NOK {total_revenue:,.0f}")
        with col4:
            if not utilization_df.empty:
                filtered_oo = len(utilization_df[
                                      (utilization_df['station_id'].isin(filtered_df['id'])) &
                                      (utilization_df['is_out_of_order'] == 1)
                                      ])
            else:
                filtered_oo = 0
            st.metric("Out of Order Connectors", filtered_oo)
    else:
        st.warning("No stations match the selected filters")


def show_utilization_analytics(utilization_df, hourly_df, sessions_df, historical_util_df):
    """Show comprehensive utilization analytics with fixed heatmap"""
    st.header("📈 Utilization Analytics")

    # Time range selector
    time_range = st.select_slider(
        "Select Time Range",
        options=["Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 7 Days", "All Data"],
        value="Last 24 Hours",
        key="analytics_time_range"
    )

    # Filter data based on time range
    now = datetime.now()
    if time_range == "Last 6 Hours":
        time_filter = now - timedelta(hours=6)
    elif time_range == "Last 12 Hours":
        time_filter = now - timedelta(hours=12)
    elif time_range == "Last 24 Hours":
        time_filter = now - timedelta(hours=24)
    elif time_range == "Last 7 Days":
        time_filter = now - timedelta(days=7)
    else:
        time_filter = utilization_df['timestamp'].min() if not utilization_df.empty else now - timedelta(days=1)

    # Filter datasets
    filtered_util = utilization_df[
        utilization_df['timestamp'] >= time_filter] if not utilization_df.empty else utilization_df
    filtered_hourly = hourly_df[hourly_df['hourly_timestamp'] >= time_filter] if not hourly_df.empty else hourly_df
    filtered_sessions = sessions_df[sessions_df['end_time'] >= time_filter] if not sessions_df.empty else sessions_df

    # Create tabs for analytics including heatmap
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Hourly Heatmap", "📈 Trends", "⚡ Power & Revenue", "💰 Session Analysis"])

    with tab1:
        # Hourly utilization heatmap
        st.subheader("Last 24 Hours Utilization Heatmap")
        
        if not historical_util_df.empty:
            # Prepare data for heatmap
            heatmap_data = historical_util_df.copy()
            heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
            heatmap_data['day'] = heatmap_data['timestamp'].dt.day_name()
            
            # Calculate occupancy rate for each day-hour combination
            pivot_data = heatmap_data.groupby(['day', 'hour'])['is_occupied'].mean().reset_index()
            
            if not pivot_data.empty:
                pivot_table = pivot_data.pivot(index='day', columns='hour', values='is_occupied')
                
                # Ensure all hours 0-23 exist
                for hour in range(24):
                    if hour not in pivot_table.columns:
                        pivot_table[hour] = 0
                
                pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
                
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                existing_days = [d for d in day_order if d in pivot_table.index]
                if existing_days:
                    pivot_table = pivot_table.reindex(existing_days)
                pivot_table = pivot_table.fillna(0)

                # Create custom colorscale optimized for low occupancy rates
                custom_colorscale = [
                    [0.0, '#27ae60'],     # 0% - Dark green (fully available)
                    [0.05, '#2ecc71'],    # ~2% - Green 
                    [0.15, '#f1c40f'],    # ~6% - Yellow (low usage)
                    [0.35, '#f39c12'],    # ~14% - Orange (approaching average)
                    [0.5, '#e67e22'],     # ~20% - Dark orange (above average)
                    [0.7, '#e74c3c'],     # ~28% - Red (high usage)
                    [0.85, '#c0392b'],    # ~34% - Dark red (very high)
                    [1.0, '#8b0000']      # 40%+ - Very dark red (extremely high)
                ]

                # Create heatmap with adjusted color scale for better sensitivity
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=list(range(24)),
                    y=pivot_table.index.tolist(),
                    colorscale=custom_colorscale,
                    colorbar=dict(
                        title='Occupancy Rate',
                        tickmode='array',
                        tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                        ticktext=['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%+'],
                        titleside='right'
                    ),
                    hoverongaps=False,
                    hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Occupancy: %{z:.1%}<extra></extra>',
                    zmin=0,
                    zmax=0.40  # Cap at 40% to make colors shift more rapidly for typical usage
                ))

                fig_heatmap.update_layout(
                    title='Last 24 Hours Utilization Pattern by Hour and Day<br><sub>Colors optimized for typical 15% average occupancy</sub>',
                    xaxis_title='Hour of Day',
                    yaxis_title='Day of Week',
                    height=500,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                    font=dict(size=12)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Add interpretation guide
                st.info("""
                **Color Guide for Occupancy Rates:**
                - 🟢 **Green (0-5%)**: Very low usage, many available connectors
                - 🟡 **Yellow (5-15%)**: Normal usage, around average occupancy
                - 🟠 **Orange (15-25%)**: Above average usage, moderate demand
                - 🔴 **Red (25%+)**: High usage, limited availability
                """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    peak_occupancy = pivot_table.values.max()
                    st.metric("Peak Occupancy", f"{peak_occupancy:.1%}")
                with col2:
                    avg_occupancy = pivot_table.values.mean()
                    st.metric("Average Occupancy", f"{avg_occupancy:.1%}")
                with col3:
                    # Find peak hour
                    hourly_avg = pivot_table.mean(axis=0)
                    peak_hour = hourly_avg.idxmax()
                    st.metric("Peak Hour", f"{peak_hour}:00", f"{hourly_avg[peak_hour]:.1%}")
                    
            else:
                st.warning("No hourly pattern data available")
        else:
            st.warning("No historical utilization data available")

    with tab2:
        # Peak hours analysis
        col1, col2 = st.columns(2)

        with col1:
            if not filtered_sessions.empty and 'start_time' in filtered_sessions.columns:
                peak_hour_sessions = filtered_sessions.groupby(filtered_sessions['start_time'].dt.hour).size()
                if not peak_hour_sessions.empty:
                    peak_hour = peak_hour_sessions.idxmax()
                    peak_count = peak_hour_sessions.max()
                    st.metric("Peak Hour (Sessions)", f"{peak_hour}:00", f"{peak_count} sessions started")
                else:
                    st.metric("Peak Hour (Sessions)", "No data", "0 sessions")
            else:
                st.metric("Peak Hour (Sessions)", "No data", "0 sessions")

        with col2:
            if not filtered_sessions.empty:
                total_completed = len(filtered_sessions)
                current_active = len(filtered_util[filtered_util['is_occupied'] == 1]) if not filtered_util.empty else 0
                st.metric("Sessions", f"{total_completed} completed", f"{current_active} active now")
            else:
                current_active = len(filtered_util[filtered_util['is_occupied'] == 1]) if not filtered_util.empty else 0
                st.metric("Sessions", "0 completed", f"{current_active} active now")

        # Utilization trends
        st.subheader("Utilization Trends")

        if not filtered_hourly.empty:
            # Multi-metric trend chart
            trend_data = filtered_hourly.groupby('hourly_timestamp').agg({
                'total_occupied': 'sum',
                'total_available': 'sum',
                'avg_occupancy_rate': 'mean'
            }).reset_index()

            fig_trend = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Connector Status Over Time', 'Average Occupancy Rate'),
                vertical_spacing=0.1
            )

            # Add traces
            fig_trend.add_trace(
                go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['total_occupied'],
                           name='Occupied', line=dict(color='#e74c3c')),
                row=1, col=1
            )
            fig_trend.add_trace(
                go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['total_available'],
                           name='Available', line=dict(color='#2ecc71')),
                row=1, col=1
            )
            fig_trend.add_trace(
                go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['avg_occupancy_rate'] * 100,
                           name='Occupancy %', line=dict(color='#3498db', width=3)),
                row=2, col=1
            )

            fig_trend.update_yaxes(title_text="Count", row=1, col=1)
            fig_trend.update_yaxes(title_text="Percentage", row=2, col=1)
            fig_trend.update_layout(height=600, showlegend=True)

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No hourly trend data available")

    with tab3:
        # Power and Revenue Analysis
        st.subheader("Power Consumption & Revenue Analysis")

        if not filtered_sessions.empty:
            # Ensure we have connector type data
            if 'connector_type' not in filtered_sessions.columns:
                # Try to get connector type from utilization data
                if not filtered_util.empty and 'connector_type' in filtered_util.columns:
                    connector_types = filtered_util.groupby('connector_id')['connector_type'].first()
                    filtered_sessions = filtered_sessions.merge(
                        connector_types, left_on='connector_id', right_index=True, how='left'
                    )
                else:
                    # Create mock connector types based on connector_id patterns
                    filtered_sessions['connector_type'] = 'Type2'  # Default fallback

            # Group by connector type
            if 'connector_type' in filtered_sessions.columns:
                type_stats = filtered_sessions.groupby('connector_type').agg({
                    'energy_kwh': ['sum', 'mean'],
                    'revenue_nok': ['sum', 'mean'],
                    'duration_hours': 'mean',
                    'connector_id': 'count'
                }).round(2)

                # Flatten column names
                type_stats.columns = ['total_energy', 'avg_energy', 'total_revenue', 'avg_revenue', 'avg_duration',
                                      'session_count']
                type_stats.reset_index(inplace=True)

                col1, col2 = st.columns(2)

                with col1:
                    # Revenue by connector type
                    fig_revenue_type = px.pie(
                        type_stats,
                        values='total_revenue',
                        names='connector_type',
                        title='Revenue Distribution by Connector Type',
                        color_discrete_map={'CCS': '#3498db', 'CHAdeMO': '#9b59b6', 'Type2': '#1abc9c'}
                    )
                    st.plotly_chart(fig_revenue_type, use_container_width=True)

                with col2:
                    # Energy delivered by connector type
                    fig_energy_type = px.bar(
                        type_stats,
                        x='connector_type',
                        y='total_energy',
                        title='Energy Delivered by Connector Type (kWh)',
                        color='connector_type',
                        color_discrete_map={'CCS': '#3498db', 'CHAdeMO': '#9b59b6', 'Type2': '#1abc9c'}
                    )
                    st.plotly_chart(fig_energy_type, use_container_width=True)

            # Session statistics
            st.markdown("---")
            st.subheader("Session Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_energy = filtered_sessions['energy_kwh'].sum()
                st.metric("Total Energy Delivered", f"{total_energy:,.0f} kWh")

            with col2:
                total_revenue = filtered_sessions['revenue_nok'].sum()
                st.metric("Total Revenue", f"NOK {total_revenue:,.0f}", f"${total_revenue / 10.5:,.0f} USD")

            with col3:
                avg_session_revenue = filtered_sessions['revenue_nok'].mean()
                st.metric("Avg Revenue/Session", f"NOK {avg_session_revenue:.1f}")

            with col4:
                avg_duration = filtered_sessions['duration_hours'].mean() * 60
                st.metric("Avg Session Duration", f"{avg_duration:.0f} min")

        else:
            st.info("No session data available for the selected time range")

    with tab4:
        # Session Analysis
        st.subheader("Detailed Session Analysis")

        if not filtered_sessions.empty:
            # Session duration and revenue distributions
            col1, col2 = st.columns(2)

            with col1:
                # Duration histogram
                fig_duration = px.histogram(
                    filtered_sessions,
                    x='duration_hours',
                    nbins=20,
                    title='Session Duration Distribution',
                    labels={'duration_hours': 'Duration (hours)', 'count': 'Number of Sessions'}
                )
                fig_duration.update_traces(marker_color='#3498db')
                st.plotly_chart(fig_duration, use_container_width=True)

            with col2:
                # Revenue per session histogram
                fig_revenue_dist = px.histogram(
                    filtered_sessions,
                    x='revenue_nok',
                    nbins=20,
                    title='Revenue per Session Distribution',
                    labels={'revenue_nok': 'Revenue (NOK)', 'count': 'Number of Sessions'}
                )
                fig_revenue_dist.update_traces(marker_color='#2ecc71')
                st.plotly_chart(fig_revenue_dist, use_container_width=True)

        else:
            st.info("No session data available for analysis")


def show_realtime_monitor(stations_df, utilization_df, sessions_df):
    """Show comprehensive real-time monitoring dashboard"""
    st.header("⚡ Real-time Station Monitor")

    # Current status overview
    col1, col2, col3, col4 = st.columns(4)

    if not utilization_df.empty:
        current_available = len(utilization_df[utilization_df['is_available'] == 1])
        current_occupied = len(utilization_df[utilization_df['is_occupied'] == 1])
        current_out_of_order = len(utilization_df[utilization_df['is_out_of_order'] == 1])
        total_connectors = len(utilization_df)
    else:
        current_available = 0
        current_occupied = 0
        current_out_of_order = 0
        total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0

    with col1:
        st.metric("🟢 Available Connectors", f"{current_available}/{total_connectors}")
    with col2:
        st.metric("🟠 Occupied Connectors", f"{current_occupied}/{total_connectors}")
    with col3:
        st.metric("🔴 Out of Order", current_out_of_order)
    with col4:
        current_utilization = (current_occupied / total_connectors * 100) if total_connectors > 0 else 0
        st.metric("📊 Current Utilization", f"{current_utilization:.1f}%")

    st.markdown("---")

    # Real-time revenue tracking
    if not sessions_df.empty:
        st.subheader("💰 Real-time Revenue Tracking")

        col1, col2, col3 = st.columns(3)

        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with col1:
            today_sessions = sessions_df[sessions_df['start_time'] >= today_start]
            today_revenue = today_sessions['revenue_nok'].sum()
            st.metric("Today's Revenue", f"NOK {today_revenue:,.0f}", f"${today_revenue / 10.5:,.0f} USD")

        with col2:
            last_hour = now - timedelta(hours=1)
            last_hour_sessions = sessions_df[sessions_df['start_time'] >= last_hour]
            last_hour_revenue = last_hour_sessions['revenue_nok'].sum()
            st.metric("Last Hour Revenue", f"NOK {last_hour_revenue:,.0f}")

        with col3:
            st.metric("Active Sessions", current_occupied)

    st.markdown("---")

    # Real-time station grid
    st.subheader("Station Status Grid")

    # Search functionality
    search_term = st.text_input("🔍 Search stations by name or address", key="realtime_search")

    if search_term and not stations_df.empty:
        display_df = stations_df[
            stations_df['name'].str.contains(search_term, case=False, na=False) |
            stations_df['address'].str.contains(search_term, case=False, na=False)
            ]
    else:
        display_df = stations_df

    # Display grid
    if not display_df.empty:
        stations_per_row = 5
        for i in range(0, len(display_df), stations_per_row):
            cols = st.columns(stations_per_row)

            for j, col in enumerate(cols):
                if i + j < len(display_df):
                    station = display_df.iloc[i + j]

                    # Get current status for this station
                    if not utilization_df.empty:
                        station_util = utilization_df[utilization_df['station_id'] == station['id']]
                        occupied = len(station_util[station_util['is_occupied'] == 1])
                        total = len(station_util) if len(station_util) > 0 else station['total_connectors']
                    else:
                        occupied = 0
                        total = station['total_connectors']

                    with col:
                        # Determine card color based on occupancy
                        if occupied == 0:
                            card_color = "#2ecc71"
                            icon = "✅"
                        elif occupied < total:
                            card_color = "#f39c12"
                            icon = "🔌"
                        else:
                            card_color = "#e74c3c"
                            icon = "⚡"

                        # Create card
                        st.markdown(f"""
                        <div class="station-card" style="
                            background-color: {card_color}20;
                            border: 2px solid {card_color};
                        ">
                            <h4 style="margin: 0; font-size: 0.9em;">{icon} {station['name'][:20]}...</h4>
                            <p style="margin: 5px 0; font-size: 0.8em;"><b>{occupied}/{total} occupied</b></p>
                            <p style="margin: 5px 0; font-size: 0.8em;">⚡ {station['total_connectors']} connectors</p>
                            <p style="margin: 5px 0; font-size: 0.7em;">📍 {str(station.get('address', 'No address'))[:25]}...</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Recent activity feed
    st.markdown("---")
    st.subheader("📰 Recent Session Activity")

    if not sessions_df.empty:
        # Get recent sessions
        recent_sessions = sessions_df.sort_values('start_time', ascending=False).head(10)

        for _, session in recent_sessions.iterrows():
            # Get station name
            if 'station_name' in session and pd.notna(session['station_name']):
                station_name = session['station_name']
            elif 'station_id' in session and pd.notna(session['station_id']):
                station_match = stations_df[stations_df['id'] == session['station_id']]
                station_name = station_match['name'].values[0] if len(station_match) > 0 else 'Unknown Station'
            else:
                station_name = 'Unknown Station'

            time_ago = (datetime.now() - session['start_time']).total_seconds() / 60

            if time_ago < 60:
                time_str = f"{int(time_ago)} minutes ago"
            elif time_ago < 1440:
                time_str = f"{int(time_ago / 60)} hours ago"
            else:
                time_str = f"{int(time_ago / 1440)} days ago"

            st.markdown(
                f"• 🔌 Session at **{station_name}** - "
                f"Duration: {session['duration_hours'] * 60:.0f} min, "
                f"Energy: {session['energy_kwh']:.1f} kWh, "
                f"Revenue: NOK {session['revenue_nok']:.0f} - "
                f"*{time_str}*"
            )
    else:
        st.info("No recent session activity to display")


def show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df):
    """Show comprehensive data explorer interface"""
    st.header("📋 Data Explorer")

    # Dataset selector
    dataset = st.selectbox(
        "Select Dataset",
        ["Charging Stations", "Utilization Data", "Hourly Aggregations", "Charging Sessions"],
        key="data_explorer_dataset"
    )

    # Get the appropriate dataframe and description
    if dataset == "Charging Stations":
        df = stations_df
        description = "Complete list of all charging stations with their specifications and current status."
    elif dataset == "Utilization Data":
        df = utilization_df
        description = "Detailed connector-level utilization records showing real-time usage patterns."
    elif dataset == "Hourly Aggregations":
        df = hourly_df
        description = "Hourly aggregated data showing utilization trends over time."
    else:
        df = sessions_df
        description = "Completed charging sessions with duration, energy consumed, and revenue generated."

    st.markdown(f"*{description}*")

    if df.empty:
        st.warning(f"No data available for {dataset}")
        return

    # Data filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)

    filters = {}

    # Dynamic filter creation based on dataset
    if dataset == "Charging Stations":
        with col1:
            if 'status' in df.columns:
                status_filter = st.multiselect("Status", df['status'].unique(), key="explorer_station_status")
                if status_filter:
                    filters['status'] = status_filter

        with col2:
            if 'operator' in df.columns:
                operator_filter = st.multiselect("Operator", df['operator'].unique(), key="explorer_station_operator")
                if operator_filter:
                    filters['operator'] = operator_filter

        with col3:
            if 'total_connectors' in df.columns:
                connector_range = st.slider(
                    "Total Connectors",
                    int(df['total_connectors'].min()),
                    int(df['total_connectors'].max()),
                    (int(df['total_connectors'].min()), int(df['total_connectors'].max())),
                    key="explorer_station_connectors"
                )
                filters['total_connectors'] = connector_range

    elif dataset == "Utilization Data":
        with col1:
            if 'status' in df.columns:
                status_filter = st.multiselect("Status", df['status'].unique(), key="explorer_util_status")
                if status_filter:
                    filters['status'] = status_filter

        with col2:
            if 'connector_type' in df.columns:
                connector_type_filter = st.multiselect("Connector Type", df['connector_type'].unique(),
                                                       key="explorer_util_type")
                if connector_type_filter:
                    filters['connector_type'] = connector_type_filter

        with col3:
            occupied_filter = st.radio("Occupancy", ["All", "Occupied", "Available"], key="explorer_util_occupancy")
            if occupied_filter == "Occupied":
                filters['is_occupied'] = 1
            elif occupied_filter == "Available":
                filters['is_occupied'] = 0

    elif dataset == "Charging Sessions":
        with col1:
            if 'revenue_nok' in df.columns:
                revenue_range = st.slider(
                    "Revenue Range (NOK)",
                    float(df['revenue_nok'].min()),
                    float(df['revenue_nok'].max()),
                    (float(df['revenue_nok'].min()), float(df['revenue_nok'].max())),
                    key="explorer_sessions_revenue"
                )
                filters['revenue_nok'] = revenue_range

        with col2:
            if 'duration_hours' in df.columns:
                duration_range = st.slider(
                    "Duration (hours)",
                    float(df['duration_hours'].min()),
                    float(df['duration_hours'].max()),
                    (float(df['duration_hours'].min()), float(df['duration_hours'].max())),
                    key="explorer_sessions_duration"
                )
                filters['duration_hours'] = duration_range

    # Apply filters
    filtered_df = df.copy()

    for col, value in filters.items():
        if col in ['total_connectors', 'revenue_nok', 'duration_hours']:
            filtered_df = filtered_df[
                (filtered_df[col] >= value[0]) &
                (filtered_df[col] <= value[1])
                ]
        elif isinstance(value, list):
            filtered_df = filtered_df[filtered_df[col].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[col] == value]

    # Display statistics
    st.subheader("📊 Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col3:
        st.metric("Columns", len(filtered_df.columns))
    with col4:
        if dataset == "Charging Sessions" and not filtered_df.empty and 'revenue_nok' in filtered_df.columns:
            total_revenue = filtered_df['revenue_nok'].sum()
            st.metric("Total Revenue", f"NOK {total_revenue:,.0f}")
        elif 'timestamp' in filtered_df.columns:
            if not filtered_df.empty:
                date_range = f"{filtered_df['timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['timestamp'].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
        else:
            memory_usage = filtered_df.memory_usage().sum() / 1024 ** 2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

    # Data preview
    st.subheader("📋 Data Preview")

    # Column selector
    if st.checkbox("Select specific columns", key="explorer_column_selector"):
        selected_columns = st.multiselect("Choose columns", filtered_df.columns.tolist(),
                                          default=filtered_df.columns.tolist(), key="explorer_columns")
        if selected_columns:
            display_df = filtered_df[selected_columns]
        else:
            display_df = filtered_df
    else:
        display_df = filtered_df

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.number_input("Number of rows to display", min_value=10, max_value=1000, value=100, step=10,
                                 key="explorer_rows")
    with col2:
        if not display_df.empty:
            sort_by = st.selectbox("Sort by", display_df.columns.tolist(), key="explorer_sort")
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="explorer_order")
        else:
            sort_by = None
            sort_order = "Ascending"

    # Sort and display
    if not display_df.empty and sort_by:
        display_df = display_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

        st.dataframe(
            display_df.head(n_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data to display")

    # Download options
    if not display_df.empty:
        st.subheader("💾 Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{dataset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )

        with col2:
            # Create Excel download
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    display_df.to_excel(writer, sheet_name='Data', index=False)

                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"{dataset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            except ImportError:
                st.info("Install xlsxwriter for Excel export: pip install xlsxwriter")

        with col3:
            # Summary statistics
            if st.button("📊 Generate Summary Report", key="explorer_summary"):
                summary = display_df.describe(include='all').to_string()
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{dataset.lower().replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_summary"
                )


# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ EV Charging Analytics Dashboard | MySQL Backend | Auto-refresh Enabled</p>
        <p>Data updates automatically every 60 seconds</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Handle deployment errors gracefully - database required
    handle_deployment_errors()

    # Run main dashboard
    main()
    add_footer()
