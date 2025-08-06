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
    try:
        if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
            os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
        elif hasattr(st, 'secrets') and 'MYSQL_HOST' in st.secrets:
            for key in ['MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE']:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
    except Exception as e:
        pass

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
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL').replace('mysql://', 'mysql+pymysql://')

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
        logging.error(f"Failed to create database engine: {e}")
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


def load_from_mysql(table_name, where_clause=None, limit=None, order_by=None):
    """Load data from MySQL table with flexible filtering"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"
            
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql(query, engine)

        datetime_columns = ['timestamp', 'hourly_timestamp', 'start_time', 'end_time', 'created_at', 'last_updated']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df

    except Exception as e:
        logger.error(f"Error loading from MySQL table {table_name}: {e}")
        return pd.DataFrame()


def get_data_date_range(table_name, date_column='timestamp'):
    """Get the available date range for a table"""
    if not MYSQL_AVAILABLE or engine is None:
        return None, None
        
    try:
        query = f"SELECT MIN({date_column}) as min_date, MAX({date_column}) as max_date FROM {table_name}"
        result = pd.read_sql(query, engine)
        if not result.empty:
            return pd.to_datetime(result['min_date'].iloc[0]), pd.to_datetime(result['max_date'].iloc[0])
    except Exception as e:
        logger.error(f"Error getting date range for {table_name}: {e}")
    
    return None, None


def get_utilization_data(start_date=None, end_date=None, latest_only=False):
    """Get utilization data for specified date range or latest data"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        if latest_only:
            query = """
            SELECT u1.* FROM utilization_data u1
            INNER JOIN (
                SELECT connector_id, MAX(timestamp) as max_timestamp
                FROM utilization_data
                GROUP BY connector_id
            ) u2 ON u1.connector_id = u2.connector_id AND u1.timestamp = u2.max_timestamp
            ORDER BY u1.timestamp DESC
            """
            return pd.read_sql(query, engine)
        else:
            where_conditions = []
            params = []
            
            if start_date:
                where_conditions.append("timestamp >= %s")
                params.append(start_date)
            if end_date:
                where_conditions.append("timestamp <= %s")
                params.append(end_date)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
            SELECT * FROM utilization_data
            WHERE {where_clause}
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql(query, engine, params=params)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'hourly_timestamp' in df.columns:
                df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])

            return df
            
    except Exception as e:
        logger.error(f"Error getting utilization data: {e}")
        return pd.DataFrame()


def get_hourly_stats(start_date=None, end_date=None):
    """Get hourly statistics for specified date range"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        where_conditions = []
        params = []
        
        if start_date:
            where_conditions.append("hourly_timestamp >= %s")
            params.append(start_date)
        if end_date:
            where_conditions.append("hourly_timestamp <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query = f"""
        SELECT 
            hourly_timestamp,
            SUM(is_occupied) as total_occupied,
            SUM(is_available) as total_available,
            SUM(is_out_of_order) as total_out_of_order,
            COUNT(*) as total_connectors,
            AVG(CASE WHEN is_occupied = 1 THEN 1.0 ELSE 0.0 END) as avg_occupancy_rate
        FROM hourly_utilization
        WHERE {where_clause}
        GROUP BY hourly_timestamp
        ORDER BY hourly_timestamp DESC
        """
        
        df = pd.read_sql(query, engine, params=params)
        if 'hourly_timestamp' in df.columns:
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
        return df
        
    except Exception as e:
        logger.error(f"Error getting hourly stats: {e}")
        return pd.DataFrame()


def get_sessions_data(start_date=None, end_date=None):
    """Get charging sessions for specified date range"""
    if not MYSQL_AVAILABLE:
        return pd.DataFrame()

    try:
        where_conditions = []
        params = []
        
        if start_date:
            where_conditions.append("start_time >= %s")
            params.append(start_date)
        if end_date:
            where_conditions.append("end_time <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query = f"""
        SELECT 
            cs.*,
            st.name as station_name,
            st.address as station_address
        FROM charging_sessions cs
        LEFT JOIN charging_stations st ON cs.station_id = st.id
        WHERE {where_clause}
        ORDER BY cs.start_time DESC
        """
        
        df = pd.read_sql(query, engine, params=params)
        
        for col in ['start_time', 'end_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting sessions data: {e}")
        return pd.DataFrame()


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

# CSS Styles
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
    .date-range-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Timezone handling
import pytz
CEST = pytz.timezone('Europe/Oslo')

# Session state initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'date_range_mode' not in st.session_state:
    st.session_state.date_range_mode = "Recent Data"

REFRESH_INTERVAL = 60


# Cache functions
@st.cache_data(ttl=300, show_spinner=False)
def load_stations_data():
    """Load charging stations data from MySQL"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required.")
        st.stop()

    try:
        df = load_from_mysql('charging_stations')
        if df.empty:
            st.warning("⚠️ No stations data found.")
            st.stop()

        numeric_cols = ['latitude', 'longitude', 'total_connectors', 'ccs_connectors',
                        'chademo_connectors', 'type2_connectors', 'other_connectors']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Error loading stations: {e}")
        st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def load_utilization_data_cached(start_date=None, end_date=None, latest_only=False):
    """Load utilization data with caching"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        df = get_utilization_data(start_date, end_date, latest_only)
        return df
    except Exception as e:
        st.error(f"❌ Error loading utilization data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_sessions_data_cached(start_date=None, end_date=None):
    """Load sessions data with caching"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        df = get_sessions_data(start_date, end_date)
        return df
    except Exception as e:
        st.error(f"❌ Error loading sessions data: {e}")
        return pd.DataFrame()


def create_date_range_selector():
    """Create date range selector in sidebar"""
    st.sidebar.markdown("### 📅 Data Range Selection")
    
    data_mode = st.sidebar.radio(
        "Data Mode",
        ["Recent Data", "Historical Analysis"],
        index=0 if st.session_state.date_range_mode == "Recent Data" else 1,
        key="data_mode_selector"
    )
    
    st.session_state.date_range_mode = data_mode
    
    if data_mode == "Recent Data":
        time_options = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6), 
            "Last 24 Hours": timedelta(hours=24),
            "Last 3 Days": timedelta(days=3),
            "Last Week": timedelta(days=7)
        }
        
        selected_range = st.sidebar.selectbox(
            "Time Range",
            list(time_options.keys()),
            index=2,
            key="recent_time_range"
        )
        
        end_date = datetime.now()
        start_date = end_date - time_options[selected_range]
        
        st.sidebar.markdown(f"""
        <div class="date-range-info">
            <strong>Selected Range:</strong><br>
            📅 {start_date.strftime('%Y-%m-%d %H:%M')} to<br>
            📅 {end_date.strftime('%Y-%m-%d %H:%M')}
        </div>
        """, unsafe_allow_html=True)
        
        return start_date, end_date, False
        
    else:
        # Historical analysis with date picker
        util_min, util_max = get_data_date_range('utilization_data', 'timestamp')
        sessions_min, sessions_max = get_data_date_range('charging_sessions', 'start_time')
        
        if util_min and sessions_min:
            overall_min = min(util_min, sessions_min)
            overall_max = max(util_max or util_min, sessions_max or sessions_min)
        elif util_min:
            overall_min, overall_max = util_min, util_max
        elif sessions_min:
            overall_min, overall_max = sessions_min, sessions_max
        else:
            overall_min = datetime.now() - timedelta(days=30)
            overall_max = datetime.now()
        
        st.sidebar.markdown(f"""
        <div class="date-range-info">
            <strong>Available Data Range:</strong><br>
            📅 {overall_min.strftime('%Y-%m-%d')} to<br>
            📅 {overall_max.strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=overall_min.date(),
                min_value=overall_min.date(),
                max_value=overall_max.date(),
                key="historical_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=min(overall_max.date(), overall_min.date() + timedelta(days=7)),
                min_value=overall_min.date(),
                max_value=overall_max.date(),
                key="historical_end_date"
            )
        
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        if start_date >= end_date:
            st.sidebar.error("❌ Start date must be before end date")
            start_date = end_date - timedelta(days=1)
        
        date_diff = (end_date - start_date).days
        st.sidebar.markdown(f"""
        <div class="date-range-info">
            <strong>Selected Range:</strong><br>
            📅 {start_date.strftime('%Y-%m-%d')} to<br>
            📅 {end_date.strftime('%Y-%m-%d')}<br>
            📊 {date_diff} days of data
        </div>
        """, unsafe_allow_html=True)
        
        return start_date, end_date, False


def check_auto_refresh():
    """Check if it's time to auto-refresh (only for recent data mode)"""
    if st.session_state.date_range_mode != "Recent Data":
        return
        
    current_time = time.time()

    if (st.session_state.auto_refresh_enabled and
            current_time - st.session_state.last_refresh > REFRESH_INTERVAL):
        st.session_state.last_refresh = current_time
        load_utilization_data_cached.clear()
        load_sessions_data_cached.clear()
        st.rerun()


def show_refresh_timer():
    """Show countdown to next refresh (only for recent data mode)"""
    if (st.session_state.auto_refresh_enabled and 
        st.session_state.date_range_mode == "Recent Data"):
        time_since_refresh = int(time.time() - st.session_state.last_refresh)
        time_to_refresh = REFRESH_INTERVAL - time_since_refresh

        if time_to_refresh > 0:
            st.markdown(
                f'<div style="position: fixed; top: 70px; right: 20px; background: #f0f2f6; padding: 5px 10px; border-radius: 5px; font-size: 0.8rem; z-index: 999;">🔄 Auto-refresh in {time_to_refresh}s</div>',
                unsafe_allow_html=True
            )


def handle_deployment_errors():
    """Handle database connection requirements"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ **Database Connection Required**")
        st.markdown("""
        This dashboard requires a MySQL database connection.
        """)
        st.stop()

    try:
        connection_status, message = test_mysql_connection()
        if not connection_status:
            st.error(f"❌ **Database Connection Failed**: {message}")
            st.stop()
    except Exception as e:
        st.error(f"❌ **Database Error**: {e}")
        st.stop()


def show_overview(stations_df, utilization_df, sessions_df, start_date, end_date):
    """Show overview dashboard"""
    st.header("📊 Overview Dashboard")
    
    date_range_days = max((end_date - start_date).days, 1)
    st.markdown(f"""
    <div class="date-range-info">
        <strong>Analysis Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} 
        ({date_range_days} days)
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_stations = len(stations_df)
        st.metric("Total Stations", total_stations)

    with col2:
        if not utilization_df.empty:
            occupied = len(utilization_df[utilization_df['is_occupied'] == 1])
            st.metric("Active Sessions", occupied)
        else:
            st.metric("Active Sessions", "No data")

    with col3:
        if not utilization_df.empty:
            avg_occupancy = utilization_df['is_occupied'].mean() * 100
            st.metric("Avg Occupancy", f"{avg_occupancy:.1f}%")
        else:
            st.metric("Avg Occupancy", "No data")

    with col4:
        if not sessions_df.empty and 'revenue_nok' in sessions_df.columns:
            total_revenue = sessions_df['revenue_nok'].sum()
            daily_avg = total_revenue / date_range_days
            st.metric("Total Revenue", f"NOK {total_revenue:,.0f}", f"NOK {daily_avg:,.0f}/day")
        else:
            st.metric("Total Revenue", "No data")

    # Charts
    if not sessions_df.empty:
        st.subheader("Session Activity")
        daily_sessions = sessions_df.groupby(sessions_df['start_time'].dt.date).size()
        fig = px.line(x=daily_sessions.index, y=daily_sessions.values, title="Daily Sessions")
        st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(stations_df, utilization_df, sessions_df, start_date, end_date):
    """Show data explorer"""
    st.header("📋 Data Explorer")
    
    dataset = st.selectbox(
        "Select Dataset",
        ["Charging Stations", "Utilization Data", "Charging Sessions"],
        key="data_explorer_dataset"
    )

    if dataset == "Charging Stations":
        df = stations_df
    elif dataset == "Utilization Data":
        df = utilization_df
    else:
        df = sessions_df

    if df.empty:
        st.warning("No data available")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)


def main():
    """Main dashboard function"""
    check_auto_refresh()

    mode_indicator = "📊 Real-time" if st.session_state.date_range_mode == "Recent Data" else "📈 Historical"
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h1 style="color: #1f77b4;">⚡ EV Charging Analytics Dashboard</h1>
        <p>{mode_indicator} | MySQL Backend</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.date_range_mode == "Recent Data":
        show_refresh_timer()

    # Load initial data
    if not st.session_state.data_loaded:
        with st.spinner('Loading data...'):
            try:
                stations_df = load_stations_data()
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()
    else:
        stations_df = load_stations_data()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=EV+Analytics", width=300)
        
        start_date, end_date, latest_only = create_date_range_selector()
        
        st.markdown("### 🔌 Navigation")
        page_options = ["📊 Overview", "📋 Data Explorer"]
        
        page = st.radio(
            "Select Dashboard",
            options=page_options,
            index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0,
            key="navigation_radio"
        )

        if page != st.session_state.current_page:
            st.session_state.current_page = page

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        if st.session_state.date_range_mode == "Recent Data":
            auto_refresh = st.checkbox("Auto-refresh (60s)", value=st.session_state.auto_refresh_enabled)
            st.session_state.auto_refresh_enabled = auto_refresh
        else:
            st.info("Auto-refresh disabled for historical mode")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

        # Database status
        st.markdown("---")
        st.markdown("### 🗄️ Database")
        if MYSQL_AVAILABLE:
            connection_status, message = test_mysql_connection()
            if connection_status:
                st.success("✅ Connected")
            else:
                st.error("❌ Error")
        else:
            st.error("❌ No Connection")

    # Load data based on selection
    utilization_df = load_utilization_data_cached(start_date, end_date, latest_only)
    sessions_df = load_sessions_data_cached(start_date, end_date)

    # Route to pages
    if st.session_state.current_page == "📊 Overview":
        show_overview(stations_df, utilization_df, sessions_df, start_date, end_date)
    elif st.session_state.current_page == "📋 Data Explorer":
        show_data_explorer(stations_df, utilization_df, sessions_df, start_date, end_date)


def add_footer():
    """Add footer"""
    st.markdown("---")
    mode_text = "Real-time" if st.session_state.date_range_mode == "Recent Data" else "Historical"
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ EV Charging Analytics - {mode_text} Mode</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    handle_deployment_errors()
    main()
    add_footer()
