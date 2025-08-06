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


# MySQL Data Loading Functions - Enhanced for Historical Data
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

        # Parse datetime columns
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
            # Get the latest utilization status for each connector
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
            # Get historical data within date range
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
            
            # Parse datetime columns
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
        # Fallback to utilization data aggregation
        util_data = get_utilization_data(start_date, end_date)
        if not util_data.empty:
            # Create hourly aggregation from utilization data
            util_data['hour'] = util_data['timestamp'].dt.floor('H')
            hourly = util_data.groupby('hour').agg({
                'is_occupied': 'sum',
                'is_available': 'sum', 
                'is_out_of_order': 'sum',
                'connector_id': 'count'
            }).reset_index()
            hourly.columns = ['hourly_timestamp', 'total_occupied', 'total_available', 'total_out_of_order', 'total_connectors']
            hourly['avg_occupancy_rate'] = hourly['total_occupied'] / hourly['total_connectors']
            return hourly
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
        
        # Parse datetime columns
        for col in ['start_time', 'end_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting sessions data: {e}")
        # Fallback to sessions table without join
        where_conditions = []
        params = []
        
        if start_date:
            where_conditions.append("start_time >= %s")
            params.append(start_date)
        if end_date:
            where_conditions.append("end_time <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        return load_from_mysql('charging_sessions', where_clause=where_clause)


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
    .date-range-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
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
if 'date_range_mode' not in st.session_state:
    st.session_state.date_range_mode = "Recent Data"

# Changed refresh interval from 30 to 60 seconds
REFRESH_INTERVAL = 60


# Enhanced data loading with date range support
@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def load_utilization_data_cached(start_date=None, end_date=None, latest_only=False):
    """Load utilization data with caching"""
    if not MYSQL_AVAILABLE or engine is None:
        st.error("❌ Database connection required. Please configure MySQL credentials.")
        st.stop()

    try:
        df = get_utilization_data(start_date, end_date, latest_only)
        if df.empty:
            if latest_only:
                st.warning("⚠️ No current utilization data found in database.")
            else:
                st.warning("⚠️ No utilization data found for the selected date range.")
        return df
    except Exception as e:
        st.error(f"❌ Error loading utilization data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_hourly_data_cached(start_date=None, end_date=None):
    """Load hourly aggregated data with caching"""
    if not MYSQL_AVAILABLE or engine is None:
        return pd.DataFrame()

    try:
        df = get_hourly_stats(start_date, end_date)
        return df
    except Exception as e:
        st.error(f"❌ Error loading hourly data: {e}")
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


# Date range component
def create_date_range_selector():
    """Create date range selector in sidebar"""
    st.sidebar.markdown("### 📅 Data Range Selection")
    
    # Mode selector
    data_mode = st.sidebar.radio(
        "Data Mode",
        ["Recent Data", "Historical Analysis"],
        index=0 if st.session_state.date_range_mode == "Recent Data" else 1,
        key="data_mode_selector"
    )
    
    st.session_state.date_range_mode = data_mode
    
    if data_mode == "Recent Data":
        # Quick time range for recent data
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
            index=2,  # Default to "Last 24 Hours"
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
        
        return start_date, end_date, False  # latest_only = False for time ranges
        
    else:
        # Historical analysis with custom date picker
        
        # Get available date ranges from database
        util_min, util_max = get_data_date_range('utilization_data', 'timestamp')
        sessions_min, sessions_max = get_data_date_range('charging_sessions', 'start_time')
        
        # Determine overall date range
        if util_min and sessions_min:
            overall_min = min(util_min, sessions_min)
            overall_max = max(util_max or util_min, sessions_max or sessions_min)
        elif util_min:
            overall_min, overall_max = util_min, util_max
        elif sessions_min:
            overall_min, overall_max = sessions_min, sessions_max
        else:
            # Fallback if no data
            overall_min = datetime.now() - timedelta(days=30)
            overall_max = datetime.now()
        
        # Show available data range
        st.sidebar.markdown(f"""
        <div class="date-range-info">
            <strong>Available Data Range:</strong><br>
            📅 {overall_min.strftime('%Y-%m-%d')} to<br>
            📅 {overall_max.strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
        
        # Date pickers
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
        if not display_df.empty:
            sort_by = st.selectbox("Sort by", display_df.columns.tolist(), key="explorer_sort")
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="explorer_order")
        else:
            sort_by = None
            sort_order = "Ascending"

    # Sort and display
    if not display_df.empty and sort_by:
        display_df = display_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

        # Show data with enhanced formatting
        st.dataframe(
            display_df.head(n_rows),
            use_container_width=True,
            hide_index=True
        )
        
        # Show summary stats for numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            with st.expander("📊 Summary Statistics"):
                summary_stats = display_df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
    else:
        st.info("No data to display")

    # Enhanced download options
    if not display_df.empty:
        st.markdown("---")
        st.subheader("💾 Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{dataset.lower().replace(' ', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_csv"
            )

        with col2:
            # Enhanced Excel download with multiple sheets
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Main data
                    display_df.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Summary statistics if numeric data exists
                    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        summary = display_df[numeric_cols].describe()
                        summary.to_excel(writer, sheet_name='Summary_Stats')
                    
                    # Metadata
                    metadata = pd.DataFrame({
                        'Dataset': [dataset],
                        'Period_Start': [start_date.strftime('%Y-%m-%d')],
                        'Period_End': [end_date.strftime('%Y-%m-%d')],
                        'Total_Records': [len(df)],
                        'Filtered_Records': [len(filtered_df)],
                        'Export_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)

                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"{dataset.lower().replace(' ', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            except ImportError:
                st.info("Install xlsxwriter for Excel export: pip install xlsxwriter")

        with col3:
            # Generate enhanced summary report
            if st.button("📊 Generate Analysis Report", key="explorer_summary"):
                report_lines = [
                    f"EV Charging Data Analysis Report",
                    f"="*50,
                    f"Dataset: {dataset}",
                    f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    f"Period Duration: {date_range_days} days",
                    f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"",
                    f"DATA SUMMARY:",
                    f"-"*20,
                    f"Total Records Available: {len(df):,}",
                    f"Records After Filtering: {len(filtered_df):,}",
                    f"Filter Coverage: {(len(filtered_df)/len(df)*100):.1f}%",
                    f"Total Columns: {len(display_df.columns)}",
                    f""
                ]
                
                # Add dataset-specific metrics
                if dataset == "Charging Sessions" and not filtered_df.empty:
                    if 'revenue_nok' in filtered_df.columns:
                        total_revenue = filtered_df['revenue_nok'].sum()
                        avg_daily_revenue = total_revenue / max(date_range_days, 1)
                        report_lines.extend([
                            f"REVENUE ANALYSIS:",
                            f"-"*20,
                            f"Total Revenue: NOK {total_revenue:,.0f}",
                            f"Average Daily Revenue: NOK {avg_daily_revenue:,.0f}",
                            f"Average Revenue per Session: NOK {filtered_df['revenue_nok'].mean():.1f}",
                            f""
                        ])
                    
                    if 'energy_kwh' in filtered_df.columns:
                        total_energy = filtered_df['energy_kwh'].sum()
                        avg_daily_energy = total_energy / max(date_range_days, 1)
                        report_lines.extend([
                            f"ENERGY ANALYSIS:",
                            f"-"*20,
                            f"Total Energy Delivered: {total_energy:,.1f} kWh",
                            f"Average Daily Energy: {avg_daily_energy:,.1f} kWh",
                            f"Average Energy per Session: {filtered_df['energy_kwh'].mean():.1f} kWh",
                            f""
                        ])
                
                elif dataset == "Utilization Data" and not filtered_df.empty:
                    if 'is_occupied' in filtered_df.columns:
                        occupancy_rate = filtered_df['is_occupied'].mean() * 100
                        unique_connectors = filtered_df['connector_id'].nunique()
                        report_lines.extend([
                            f"UTILIZATION ANALYSIS:",
                            f"-"*20,
                            f"Average Occupancy Rate: {occupancy_rate:.1f}%",
                            f"Unique Connectors: {unique_connectors:,}",
                            f"Total Utilization Records: {len(filtered_df):,}",
                            f""
                        ])
                
                # Add column information
                report_lines.extend([
                    f"COLUMN INFORMATION:",
                    f"-"*20
                ])
                
                for col in display_df.columns:
                    dtype = str(display_df[col].dtype)
                    non_null = display_df[col].count()
                    null_pct = ((len(display_df) - non_null) / len(display_df) * 100) if len(display_df) > 0 else 0
                    report_lines.append(f"{col}: {dtype} ({non_null:,} non-null, {null_pct:.1f}% missing)")
                
                # Add summary statistics for numeric columns
                numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    report_lines.extend([
                        f"",
                        f"SUMMARY STATISTICS:",
                        f"-"*20
                    ])
                    summary = display_df[numeric_cols].describe()
                    report_lines.append(summary.to_string())
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_text,
                    file_name=f"{dataset.lower().replace(' ', '_')}_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="download_summary"
                )

        # Quick visualization option
        st.markdown("---")
        st.subheader("📈 Quick Visualization")
        
        if not display_df.empty:
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = display_df.select_dtypes(include=['datetime64']).columns.tolist()
            
            if numeric_cols:
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    chart_type = st.selectbox("Chart Type", ["Histogram", "Line Plot", "Scatter Plot", "Box Plot"], key="viz_chart_type")
                
                with viz_col2:
                    if chart_type in ["Histogram", "Box Plot"]:
                        selected_col = st.selectbox("Select Column", numeric_cols, key="viz_column")
                    else:
                        selected_col = st.selectbox("Y-axis Column", numeric_cols, key="viz_y_column")
                
                if chart_type == "Histogram":
                    fig = px.histogram(display_df.sample(min(1000, len(display_df))), x=selected_col, 
                                     title=f"{selected_col} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_type == "Box Plot":
                    fig = px.box(display_df.sample(min(1000, len(display_df))), y=selected_col, 
                               title=f"{selected_col} Box Plot")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_type == "Line Plot" and date_cols:
                    date_col = st.selectbox("X-axis (Date)", date_cols, key="viz_date_column")
                    # Aggregate by date for line plot
                    line_data = display_df.groupby(display_df[date_col].dt.date)[selected_col].mean().reset_index()
                    fig = px.line(line_data, x=date_col, y=selected_col, 
                                title=f"{selected_col} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis Column", [col for col in numeric_cols if col != selected_col], key="viz_x_column")
                    sample_data = display_df.sample(min(1000, len(display_df)))
                    fig = px.scatter(sample_data, x=x_col, y=selected_col,
                                   title=f"{selected_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for visualization")


# Footer with enhanced information
def add_footer():
    st.markdown("---")
    mode_text = "Real-time Dashboard" if st.session_state.date_range_mode == "Recent Data" else "Historical Analysis Dashboard"
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ EV Charging Analytics Dashboard - {mode_text}</p>
        <p>🗄️ MySQL Backend | 📊 Enhanced Historical Data Support | 🔄 Flexible Time Ranges</p>
        <p>Auto-refresh: {'Enabled' if st.session_state.auto_refresh_enabled else 'Disabled'} | 
           Data Mode: {st.session_state.date_range_mode}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Handle deployment errors gracefully - database required
    handle_deployment_errors()

    # Run main dashboard
    main()
    add_footer():
            end_date = st.date_input(
                "End Date", 
                value=min(overall_max.date(), overall_min.date() + timedelta(days=7)),
                min_value=overall_min.date(),
                max_value=overall_max.date(),
                key="historical_end_date"
            )
        
        # Convert to datetime
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        # Validate date range
        if start_date >= end_date:
            st.sidebar.error("❌ Start date must be before end date")
            start_date = end_date - timedelta(days=1)
        
        # Show selected range info
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


# Helper function to format CEST time
def format_cest_time(dt):
    """Format datetime to CEST timezone"""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(CEST)


# Better auto-refresh functionality - only for recent data mode
def check_auto_refresh():
    """Check if it's time to auto-refresh (only for recent data mode)"""
    # Only auto-refresh in recent data mode
    if st.session_state.date_range_mode != "Recent Data":
        return
        
    current_time = time.time()

    if (st.session_state.auto_refresh_enabled and
            current_time - st.session_state.last_refresh > REFRESH_INTERVAL):
        st.session_state.last_refresh = current_time
        # Clear caches
        load_utilization_data_cached.clear()
        load_hourly_data_cached.clear()
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


# Main dashboard with enhanced historical data support
def main():
    # Check for auto-refresh but preserve navigation state
    check_auto_refresh()

    # Title with updated info
    mode_indicator = "📊 Real-time" if st.session_state.date_range_mode == "Recent Data" else "📈 Historical Analysis"
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h1 style="color: #1f77b4; margin-bottom: 1rem;">⚡ EV Charging Analytics Dashboard</h1>
        <p>{mode_indicator} | MySQL Backend | Enhanced Historical Data Support</p>
        <p>Current time: {get_cest_time().strftime('%Y-%m-%d %H:%M:%S')} CEST</p>
    </div>
    """, unsafe_allow_html=True)

    # Show refresh timer only for recent data
    if st.session_state.date_range_mode == "Recent Data":
        show_refresh_timer()

    # Load data based on date range selection
    if not st.session_state.data_loaded:
        with st.spinner('Loading initial data...'):
            try:
                stations_df = load_stations_data()
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading initial data: {e}")
                st.stop()
    else:
        try:
            stations_df = load_stations_data()
        except Exception as e:
            st.error(f"Error refreshing data: {e}")
            st.stop()

    # Sidebar with enhanced date range controls
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=EV+Charging+Analytics", width=300)
        
        # Date range selector
        start_date, end_date, latest_only = create_date_range_selector()
        
        st.markdown("### 🔌 Navigation")

        # Navigation
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

        # Auto-refresh controls (only for recent data)
        st.markdown("### ⚙️ Settings")
        
        if st.session_state.date_range_mode == "Recent Data":
            auto_refresh = st.checkbox("Real-time refresh (60s)", value=st.session_state.auto_refresh_enabled)
            st.session_state.auto_refresh_enabled = auto_refresh
        else:
            st.info("Auto-refresh disabled for historical analysis")
            st.session_state.auto_refresh_enabled = False

        if st.button("🔄 Refresh Data"):
            # Clear all caches and reset timer
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.data_loaded = False
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Stations", len(stations_df))

        # Load utilization data for quick stats
        utilization_df = load_utilization_data_cached(start_date, end_date, latest_only)
        
        if not utilization_df.empty:
            active_sessions = len(utilization_df[utilization_df['is_occupied'] == 1])
            st.metric("Active Sessions", active_sessions)
        else:
            st.metric("Active Sessions", "No data")

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
            
            if st.session_state.date_range_mode == "Recent Data":
                data_age = (get_cest_time() - latest_data_cest).total_seconds() / 60
                if data_age < 5:
                    st.success(f"📡 Data current")
                elif data_age < 30:
                    st.warning(f"📡 Data {data_age:.1f}m old")
                else:
                    st.error(f"📡 Data {data_age:.1f}m old")
            else:
                st.info(f"📡 Latest data: {latest_data_cest.strftime('%Y-%m-%d %H:%M')}")

    # Load data based on selected date range and page requirements
    utilization_df = load_utilization_data_cached(start_date, end_date, latest_only)
    hourly_df = pd.DataFrame()
    sessions_df = pd.DataFrame()

    # Load additional data based on page requirements
    if st.session_state.current_page in ["📊 Overview", "📈 Utilization Analytics", "📋 Data Explorer"]:
        hourly_df = load_hourly_data_cached(start_date, end_date)

    if st.session_state.current_page in ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analytics", 
                                         "⚡ Real-time Monitor", "📋 Data Explorer"]:
        sessions_df = load_sessions_data_cached(start_date, end_date)

    # Route to appropriate page with date range context
    if st.session_state.current_page == "📊 Overview":
        show_overview(stations_df, utilization_df, hourly_df, sessions_df, start_date, end_date)
    elif st.session_state.current_page == "🗺️ Station Map":
        show_station_map(stations_df, utilization_df, sessions_df, start_date, end_date)
    elif st.session_state.current_page == "📈 Utilization Analytics":
        show_utilization_analytics(utilization_df, hourly_df, sessions_df, start_date, end_date)
    elif st.session_state.current_page == "⚡ Real-time Monitor":
        show_realtime_monitor(stations_df, utilization_df, sessions_df, start_date, end_date)
    elif st.session_state.current_page == "📋 Data Explorer":
        show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df, start_date, end_date)


def show_overview(stations_df, utilization_df, hourly_df, sessions_df, start_date, end_date):
    """Show comprehensive overview dashboard with date range context"""
    st.header(f"📊 Overview Dashboard")
    
    # Show date range info
    date_range_days = (end_date - start_date).days
    st.markdown(f"""
    <div class="date-range-info">
        <strong>Analysis Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} 
        ({date_range_days} days)
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row - adjusted for historical context
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_stations = len(stations_df)
        if st.session_state.date_range_mode == "Recent Data":
            available_stations = len(stations_df[stations_df['status'] == 'Available'])
            st.metric(
                "Available Stations",
                f"{available_stations}/{total_stations}",
                f"{(available_stations / total_stations * 100):.1f}%" if total_stations > 0 else "0%"
            )
        else:
            st.metric("Total Stations", total_stations)

    with col2:
        total_connectors = stations_df['total_connectors'].sum()
        if not utilization_df.empty:
            if st.session_state.date_range_mode == "Recent Data":
                occupied_connectors = len(utilization_df[utilization_df['is_occupied'] == 1])
                st.metric(
                    "Active Sessions",
                    occupied_connectors,
                    f"{(occupied_connectors / total_connectors * 100):.1f}% utilization" if total_connectors > 0 else "0%"
                )
            else:
                # For historical data, show unique connectors that were active
                unique_active_connectors = utilization_df[utilization_df['is_occupied'] == 1]['connector_id'].nunique()
                st.metric("Active Connectors Used", unique_active_connectors)
        else:
            st.metric("Active Sessions", "No data")

    with col3:
        if not utilization_df.empty:
            avg_occupancy = utilization_df['is_occupied'].mean() * 100
            st.metric("Avg Occupancy Rate", f"{avg_occupancy:.1f}%")
        else:
            st.metric("Avg Occupancy Rate", "No data")

    with col4:
        if not sessions_df.empty and 'revenue_nok' in sessions_df.columns:
            total_revenue_nok = sessions_df['revenue_nok'].sum()
            total_revenue_usd = total_revenue_nok / 10.5
            if date_range_days > 0:
                daily_avg_revenue_usd = total_revenue_usd / date_range_days
                st.metric(
                    f"Total Revenue ({date_range_days}d)",
                    f"${total_revenue_usd:,.0f}",
                    f"${daily_avg_revenue_usd:,.0f}/day avg"
                )
            else:
                st.metric("Total Revenue", f"${total_revenue_usd:,.0f}")
        else:
            st.metric("Total Revenue", "No data")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Station status or utilization distribution
        if not utilization_df.empty and st.session_state.date_range_mode == "Historical Analysis":
            # For historical data, show utilization distribution
            util_counts = {
                'Occupied': len(utilization_df[utilization_df['is_occupied'] == 1]),
                'Available': len(utilization_df[utilization_df['is_available'] == 1]),
                'Out of Order': len(utilization_df[utilization_df['is_out_of_order'] == 1])
            }
            fig_pie = px.pie(
                values=list(util_counts.values()),
                names=list(util_counts.keys()),
                title="Historical Utilization Distribution",
                color_discrete_map={
                    'Available': '#2ecc71',
                    'Occupied': '#f39c12',
                    'Out of Order': '#e74c3c'
                }
            )
        elif not stations_df.empty:
            # For recent data, show station status
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
        else:
            fig_pie = None
            
        if fig_pie:
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

    # Time-based analysis
    st.markdown("---")

    if not sessions_df.empty or not utilization_df.empty:
        # Create comprehensive time-based analysis
        fig_time = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Session Activity Over Time',
                'Hourly Usage Pattern'
            ),
            vertical_spacing=0.12
        )

        # Sessions over time
        if not sessions_df.empty and 'start_time' in sessions_df.columns:
            sessions_daily = sessions_df.groupby(sessions_df['start_time'].dt.date).size().reset_index()
            sessions_daily.columns = ['date', 'session_count']
            
            fig_time.add_trace(
                go.Scatter(
                    x=sessions_daily['date'],
                    y=sessions_daily['session_count'],
                    name='Daily Sessions',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )

        # Hourly pattern analysis
        hourly_pattern = None
        if not sessions_df.empty and 'start_time' in sessions_df.columns:
            sessions_df_copy = sessions_df.copy()
            sessions_df_copy['hour'] = sessions_df_copy['start_time'].dt.hour
            hourly_pattern = sessions_df_copy.groupby('hour').size()
        elif not utilization_df.empty and 'timestamp' in utilization_df.columns:
            util_df_copy = utilization_df.copy()
            util_df_copy['hour'] = util_df_copy['timestamp'].dt.hour
            hourly_pattern = util_df_copy.groupby('hour')['is_occupied'].sum()

        if hourly_pattern is not None:
            # Ensure all hours are represented
            all_hours = pd.Series(0, index=range(24))
            all_hours.update(hourly_pattern)
            hourly_pattern = all_hours

            fig_time.add_trace(
                go.Bar(
                    x=hourly_pattern.index,
                    y=hourly_pattern.values,
                    name='Hourly Activity',
                    marker_color='#e74c3c',
                    opacity=0.7
                ),
                row=2, col=1
            )

        # Update layout
        fig_time.update_xaxes(title_text="Date", row=1, col=1)
        fig_time.update_xaxes(title_text="Hour of Day", row=2, col=1, tickmode='linear', tick0=0, dtick=1)
        fig_time.update_yaxes(title_text="Sessions", row=1, col=1)
        fig_time.update_yaxes(title_text="Activity Count", row=2, col=1)
        fig_time.update_layout(
            title=f'Usage Analysis: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig_time, use_container_width=True)

    # Station performance table
    st.markdown("---")
    st.subheader("🏆 Top Performing Stations")

    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        # Aggregate performance by station
        station_performance = sessions_df.groupby('station_id').agg({
            'revenue_nok': 'sum',
            'energy_kwh': 'sum',
            'connector_id': 'count'
        }).reset_index()
        station_performance.columns = ['station_id', 'total_revenue', 'total_energy', 'session_count']

        # Merge with station info
        station_performance = station_performance.merge(
            stations_df[['id', 'name', 'address']],
            left_on='station_id',
            right_on='id',
            how='left'
        )

        # Sort by revenue and get top 10
        station_performance = station_performance.sort_values('total_revenue', ascending=False).head(10)

        if not station_performance.empty:
            # Add performance metrics
            station_performance['daily_avg_revenue'] = station_performance['total_revenue'] / max(date_range_days, 1)
            station_performance['avg_energy_per_session'] = station_performance['total_energy'] / station_performance['session_count']
            
            st.dataframe(
                station_performance[['name', 'address', 'total_revenue', 'daily_avg_revenue', 'total_energy', 'session_count']].rename(
                    columns={
                        'name': 'Station Name',
                        'address': 'Address',
                        'total_revenue': 'Total Revenue (NOK)',
                        'daily_avg_revenue': 'Daily Avg Revenue (NOK)',
                        'total_energy': 'Total Energy (kWh)',
                        'session_count': 'Sessions'
                    }
                ),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No session data available for station performance ranking")


def show_station_map(stations_df, utilization_df, sessions_df, start_date, end_date):
    """Show interactive station map with date range context"""
    st.header("🗺️ Charging Station Map")
    
    # Show date range info
    date_range_days = (end_date - start_date).days
    st.markdown(f"""
    <div class="date-range-info">
        <strong>Data Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} 
        ({date_range_days} days)
    </div>
    """, unsafe_allow_html=True)

    # Calculate station metrics for the date range
    station_metrics = {}
    if not sessions_df.empty and 'station_id' in sessions_df.columns:
        station_revenue = sessions_df.groupby('station_id').agg({
            'revenue_nok': 'sum',
            'connector_id': 'count',
            'energy_kwh': 'sum'
        }).to_dict('index')
        station_metrics = station_revenue

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.date_range_mode == "Recent Data":
            status_options = ['Available', 'Occupied', 'OutOfOrder']
            default_status = ['Available', 'Occupied', 'OutOfOrder']
        else:
            status_options = ['Available', 'Occupied', 'OutOfOrder', 'All Historical Data']
            default_status = ['All Historical Data']
            
        status_filter = st.multiselect(
            "Filter by Status",
            options=status_options,
            default=default_status,
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
    if 'All Historical Data' in status_filter:
        filtered_df = stations_df[stations_df['total_connectors'] >= connector_filter]
    else:
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
            # Get current status or historical summary
            if st.session_state.date_range_mode == "Recent Data" and not utilization_df.empty:
                station_connectors = utilization_df[utilization_df['station_id'] == station['id']]
                occupied_count = len(station_connectors[station_connectors['is_occupied'] == 1])
                available_count = len(station_connectors[station_connectors['is_available'] == 1])
                out_of_order_count = len(station_connectors[station_connectors['is_out_of_order'] == 1])
                
                # Determine marker color for real-time
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
            else:
                # For historical data, color by activity level
                station_data = station_metrics.get(station['id'], {})
                total_sessions = station_data.get('connector_id', 0)
                
                if total_sessions > 50:
                    color = 'red'  # High activity
                    status_text = f'High Activity ({total_sessions} sessions)'
                elif total_sessions > 20:
                    color = 'orange'  # Medium activity
                    status_text = f'Medium Activity ({total_sessions} sessions)'
                elif total_sessions > 0:
                    color = 'green'  # Low activity
                    status_text = f'Low Activity ({total_sessions} sessions)'
                else:
                    color = 'gray'  # No activity
                    status_text = 'No Activity'
                
                occupied_count = available_count = out_of_order_count = "N/A"

            # Get metrics for this station
            station_data = station_metrics.get(station['id'], {})
            revenue = station_data.get('revenue_nok', 0)
            total_energy = station_data.get('energy_kwh', 0)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>{station['name']}</h4>
                <p><b>Status:</b> {status_text}</p>
                <p><b>Address:</b> {station.get('address', 'N/A')}</p>
                <p><b>Connectors:</b> {station['total_connectors']}</p>
            """
            
            if st.session_state.date_range_mode == "Recent Data":
                popup_html += f"""
                <p><b>Current Status:</b><br>
                   - Available: {available_count}<br>
                   - Occupied: {occupied_count}<br>
                   - Out of Order: {out_of_order_count}</p>
                """
            
            popup_html += f"""
                <p><b>Types:</b><br>
                   - CCS: {station.get('ccs_connectors', 0)}<br>
                   - CHAdeMO: {station.get('chademo_connectors', 0)}<br>
                   - Type 2: {station.get('type2_connectors', 0)}</p>
                <p><b>Period Revenue:</b> NOK {revenue:,.0f}</p>
                <p><b>Period Energy:</b> {total_energy:,.1f} kWh</p>
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
            total_revenue = sum(station_metrics.get(sid, {}).get('revenue_nok', 0) for sid in filtered_df['id'])
            st.metric("Period Revenue", f"NOK {total_revenue:,.0f}")
        with col4:
            total_sessions = sum(station_metrics.get(sid, {}).get('connector_id', 0) for sid in filtered_df['id'])
            st.metric("Total Sessions", total_sessions)
    else:
        st.warning("No stations match the selected filters")


def show_utilization_analytics(utilization_df, hourly_df, sessions_df, start_date, end_date):
    """Show comprehensive utilization analytics with enhanced historical support"""
    st.header("📈 Utilization Analytics")
    
    # Show date range info
    date_range_days = (end_date - start_date).days
    st.markdown(f"""
    <div class="date-range-info">
        <strong>Analysis Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} 
        ({date_range_days} days)
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Usage Heatmap", "📈 Trends", "⚡ Power & Revenue", "💰 Session Analysis"])

    with tab1:
        # Enhanced heatmap for historical data
        st.subheader("Utilization Heatmap")

        if not utilization_df.empty:
            # Prepare data for heatmap
            heatmap_data = utilization_df.copy()
            heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
            heatmap_data['day'] = heatmap_data['timestamp'].dt.day_name()
            heatmap_data['date'] = heatmap_data['timestamp'].dt.date

            # Create different heatmap views based on date range
            if date_range_days <= 7:
                # Day-hour heatmap for short periods
                pivot_data = heatmap_data.groupby(['day', 'hour'])['is_occupied'].mean().reset_index()
                pivot_table = pivot_data.pivot(index='day', columns='hour', values='is_occupied')
                
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                existing_days = [d for d in day_order if d in pivot_table.index]
                if existing_days:
                    pivot_table = pivot_table.reindex(existing_days)
                
                y_title = "Day of Week"
                title_suffix = "by Day and Hour"
                
            elif date_range_days <= 31:
                # Date-hour heatmap for medium periods
                pivot_data = heatmap_data.groupby(['date', 'hour'])['is_occupied'].mean().reset_index()
                pivot_table = pivot_data.pivot(index='date', columns='hour', values='is_occupied')
                y_title = "Date"
                title_suffix = "by Date and Hour"
                
            else:
                # Monthly-hour heatmap for long periods
                heatmap_data['month'] = heatmap_data['timestamp'].dt.to_period('M').astype(str)
                pivot_data = heatmap_data.groupby(['month', 'hour'])['is_occupied'].mean().reset_index()
                pivot_table = pivot_data.pivot(index='month', columns='hour', values='is_occupied')
                y_title = "Month"
                title_suffix = "by Month and Hour"

            if not pivot_table.empty:
                # Ensure all hours 0-23 exist
                for hour in range(24):
                    if hour not in pivot_table.columns:
                        pivot_table[hour] = 0

                pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
                pivot_table = pivot_table.fillna(0)

                # Create heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=list(range(24)),
                    y=pivot_table.index.tolist(),
                    colorscale=[
                        [0.0, '#27ae60'],   # 0% - Green
                        [0.1, '#2ecc71'],   # 10% - Light green
                        [0.25, '#f1c40f'],  # 25% - Yellow
                        [0.5, '#f39c12'],   # 50% - Orange
                        [0.75, '#e74c3c'],  # 75% - Red
                        [1.0, '#8b0000']    # 100% - Dark red
                    ],
                    colorbar=dict(title='Occupancy Rate'),
                    hoverongaps=False,
                    hovertemplate=f'{y_title}: %{{y}}<br>Hour: %{{x}}:00<br>Occupancy: %{{z:.1%}}<extra></extra>',
                ))

                fig_heatmap.update_layout(
                    title=f'Utilization Pattern {title_suffix}<br><sub>Period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}</sub>',
                    xaxis_title='Hour of Day',
                    yaxis_title=y_title,
                    height=500,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

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
                st.warning("No data available for heatmap")
        else:
            st.warning("No utilization data available")

    with tab2:
        # Enhanced trends for historical data
        st.subheader("Historical Trends")

        # Time-based metrics
        col1, col2 = st.columns(2)

        with col1:
            if not sessions_df.empty:
                total_sessions = len(sessions_df)
                if date_range_days > 0:
                    avg_sessions_per_day = total_sessions / date_range_days
                    st.metric("Total Sessions", total_sessions, f"{avg_sessions_per_day:.1f}/day avg")
                else:
                    st.metric("Total Sessions", total_sessions)
            else:
                st.metric("Total Sessions", "No data")

        with col2:
            if not utilization_df.empty:
                unique_connectors_used = utilization_df[utilization_df['is_occupied'] == 1]['connector_id'].nunique()
                total_connectors = utilization_df['connector_id'].nunique()
                utilization_rate = (unique_connectors_used / total_connectors * 100) if total_connectors > 0 else 0
                st.metric("Connector Utilization", f"{unique_connectors_used}/{total_connectors}", f"{utilization_rate:.1f}%")
            else:
                st.metric("Connector Utilization", "No data")

        # Trends over time
        if not sessions_df.empty:
            # Daily session trends
            sessions_daily = sessions_df.groupby(sessions_df['start_time'].dt.date).agg({
                'connector_id': 'count',
                'revenue_nok': 'sum',
                'energy_kwh': 'sum'
            }).reset_index()
            sessions_daily.columns = ['date', 'sessions', 'revenue', 'energy']

            fig_trends = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Daily Sessions', 'Daily Revenue (NOK)', 'Daily Energy (kWh)'),
                vertical_spacing=0.08
            )

            # Sessions trend
            fig_trends.add_trace(
                go.Scatter(x=sessions_daily['date'], y=sessions_daily['sessions'],
                           name='Sessions', line=dict(color='#3498db'), mode='lines+markers'),
                row=1, col=1
            )

            # Revenue trend
            fig_trends.add_trace(
                go.Scatter(x=sessions_daily['date'], y=sessions_daily['revenue'],
                           name='Revenue', line=dict(color='#2ecc71'), mode='lines+markers'),
                row=2, col=1
            )

            # Energy trend
            fig_trends.add_trace(
                go.Scatter(x=sessions_daily['date'], y=sessions_daily['energy'],
                           name='Energy', line=dict(color='#e74c3c'), mode='lines+markers'),
                row=3, col=1
            )

            fig_trends.update_layout(height=800, showlegend=False, title_text="Daily Trends Analysis")
            st.plotly_chart(fig_trends, use_container_width=True)

        elif not hourly_df.empty:
            # Use hourly data if sessions not available
            fig_hourly_trend = px.line(
                hourly_df.sort_values('hourly_timestamp'),
                x='hourly_timestamp',
                y=['total_occupied', 'total_available'],
                title='Hourly Utilization Trends',
                labels={'value': 'Count', 'hourly_timestamp': 'Time'}
            )
            st.plotly_chart(fig_hourly_trend, use_container_width=True)
        else:
            st.info("No trend data available for the selected period")

    with tab3:
        # Power and Revenue Analysis
        st.subheader("Power Consumption & Revenue Analysis")

        if not sessions_df.empty:
            # Revenue and energy metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_energy = sessions_df['energy_kwh'].sum()
                avg_daily_energy = total_energy / max(date_range_days, 1)
                st.metric("Total Energy", f"{total_energy:,.0f} kWh", f"{avg_daily_energy:.0f} kWh/day")

            with col2:
                total_revenue = sessions_df['revenue_nok'].sum()
                avg_daily_revenue = total_revenue / max(date_range_days, 1)
                st.metric("Total Revenue", f"NOK {total_revenue:,.0f}", f"NOK {avg_daily_revenue:.0f}/day")

            with col3:
                avg_session_revenue = sessions_df['revenue_nok'].mean()
                st.metric("Avg Revenue/Session", f"NOK {avg_session_revenue:.1f}")

            with col4:
                avg_session_energy = sessions_df['energy_kwh'].mean()
                st.metric("Avg Energy/Session", f"{avg_session_energy:.1f} kWh")

            # Connector type analysis
            if 'connector_type' in sessions_df.columns:
                connector_analysis = sessions_df.groupby('connector_type').agg({
                    'energy_kwh': ['sum', 'mean'],
                    'revenue_nok': ['sum', 'mean'],
                    'duration_hours': 'mean',
                    'connector_id': 'count'
                }).round(2)

                connector_analysis.columns = ['total_energy', 'avg_energy', 'total_revenue', 'avg_revenue', 'avg_duration', 'session_count']
                connector_analysis = connector_analysis.reset_index()

                col1, col2 = st.columns(2)

                with col1:
                    # Revenue by connector type
                    fig_revenue_type = px.pie(
                        connector_analysis,
                        values='total_revenue',
                        names='connector_type',
                        title='Revenue Distribution by Connector Type',
                        color_discrete_map={'CCS': '#3498db', 'CHAdeMO': '#9b59b6', 'Type2': '#1abc9c'}
                    )
                    st.plotly_chart(fig_revenue_type, use_container_width=True)

                with col2:
                    # Energy by connector type
                    fig_energy_type = px.bar(
                        connector_analysis,
                        x='connector_type',
                        y='total_energy',
                        title='Energy Delivered by Connector Type (kWh)',
                        color='connector_type',
                        color_discrete_map={'CCS': '#3498db', 'CHAdeMO': '#9b59b6', 'Type2': '#1abc9c'}
                    )
                    st.plotly_chart(fig_energy_type, use_container_width=True)

                # Detailed connector performance table
                st.markdown("---")
                st.subheader("Connector Type Performance")
                st.dataframe(
                    connector_analysis.rename(columns={
                        'connector_type': 'Connector Type',
                        'total_energy': 'Total Energy (kWh)',
                        'avg_energy': 'Avg Energy/Session (kWh)',
                        'total_revenue': 'Total Revenue (NOK)',
                        'avg_revenue': 'Avg Revenue/Session (NOK)',
                        'avg_duration': 'Avg Duration (hours)',
                        'session_count': 'Session Count'
                    }),
                    hide_index=True,
                    use_container_width=True
                )

        else:
            st.info("No session data available for power and revenue analysis")

    with tab4:
        # Enhanced Session Analysis
        st.subheader("Detailed Session Analysis")

        if not sessions_df.empty:
            # Session distribution analysis
            col1, col2 = st.columns(2)

            with col1:
                # Duration distribution
                fig_duration = px.histogram(
                    sessions_df,
                    x='duration_hours',
                    nbins=min(30, len(sessions_df) // 10),
                    title='Session Duration Distribution',
                    labels={'duration_hours': 'Duration (hours)', 'count': 'Number of Sessions'},
                    marginal='box'  # Add box plot
                )
                fig_duration.update_traces(marker_color='#3498db')
                st.plotly_chart(fig_duration, use_container_width=True)

            with col2:
                # Revenue distribution
                fig_revenue_dist = px.histogram(
                    sessions_df,
                    x='revenue_nok',
                    nbins=min(30, len(sessions_df) // 10),
                    title='Revenue per Session Distribution',
                    labels={'revenue_nok': 'Revenue (NOK)', 'count': 'Number of Sessions'},
                    marginal='box'  # Add box plot
                )
                fig_revenue_dist.update_traces(marker_color='#2ecc71')
                st.plotly_chart(fig_revenue_dist, use_container_width=True)

            # Energy vs Duration scatter plot
            st.markdown("---")
            if 'energy_kwh' in sessions_df.columns and 'duration_hours' in sessions_df.columns:
                fig_scatter = px.scatter(
                    sessions_df.sample(min(1000, len(sessions_df))),  # Sample for performance
                    x='duration_hours',
                    y='energy_kwh',
                    color='revenue_nok',
                    title='Energy vs Duration Analysis',
                    labels={
                        'duration_hours': 'Session Duration (hours)',
                        'energy_kwh': 'Energy Consumed (kWh)',
                        'revenue_nok': 'Revenue (NOK)'
                    },
                    hover_data=['revenue_nok']
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Session statistics summary
            st.markdown("---")
            st.subheader("Session Statistics Summary")
            
            stats_data = {
                'Metric': ['Total Sessions', 'Total Duration', 'Total Energy', 'Total Revenue', 
                          'Avg Session Duration', 'Avg Energy/Session', 'Avg Revenue/Session',
                          'Min Duration', 'Max Duration', 'Median Duration'],
                'Value': [
                    f"{len(sessions_df):,}",
                    f"{sessions_df['duration_hours'].sum():,.1f} hours",
                    f"{sessions_df['energy_kwh'].sum():,.1f} kWh",
                    f"NOK {sessions_df['revenue_nok'].sum():,.0f}",
                    f"{sessions_df['duration_hours'].mean():.1f} hours",
                    f"{sessions_df['energy_kwh'].mean():.1f} kWh",
                    f"NOK {sessions_df['revenue_nok'].mean():.1f}",
                    f"{sessions_df['duration_hours'].min():.1f} hours",
                    f"{sessions_df['duration_hours'].max():.1f} hours",
                    f"{sessions_df['duration_hours'].median():.1f} hours"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        else:
            st.info("No session data available for detailed analysis")


def show_realtime_monitor(stations_df, utilization_df, sessions_df, start_date, end_date):
    """Show monitoring dashboard adapted for historical data"""
    st.header("⚡ Station Monitor")
    
    # Adapt title based on data mode
    if st.session_state.date_range_mode == "Recent Data":
        st.markdown("*Real-time monitoring and recent activity*")
    else:
        date_range_days = (end_date - start_date).days
        st.markdown(f"*Historical monitoring for {date_range_days} days: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}*")

    # Current/Historical status overview
    col1, col2, col3, col4 = st.columns(4)

    if not utilization_df.empty:
        if st.session_state.date_range_mode == "Recent Data":
            # Real-time metrics
            current_available = len(utilization_df[utilization_df['is_available'] == 1])
            current_occupied = len(utilization_df[utilization_df['is_occupied'] == 1])
            current_out_of_order = len(utilization_df[utilization_df['is_out_of_order'] == 1])
            total_connectors = len(utilization_df)
        else:
            # Historical metrics - unique connectors that were used
            current_occupied = utilization_df[utilization_df['is_occupied'] == 1]['connector_id'].nunique()
            current_available = utilization_df[utilization_df['is_available'] == 1]['connector_id'].nunique()
            current_out_of_order = utilization_df[utilization_df['is_out_of_order'] == 1]['connector_id'].nunique()
            total_connectors = utilization_df['connector_id'].nunique()
    else:
        current_available = current_occupied = current_out_of_order = 0
        total_connectors = stations_df['total_connectors'].sum() if not stations_df.empty else 0

    with col1:
        label = "Available Connectors" if st.session_state.date_range_mode == "Recent Data" else "Connectors Used (Available)"
        st.metric(f"🟢 {label}", f"{current_available}/{total_connectors}")
    with col2:
        label = "Occupied Connectors" if st.session_state.date_range_mode == "Recent Data" else "Connectors Used (Occupied)"
        st.metric(f"🟠 {label}", f"{current_occupied}/{total_connectors}")
    with col3:
        label = "Out of Order" if st.session_state.date_range_mode == "Recent Data" else "Connectors Used (OoO)"
        st.metric(f"🔴 {label}", current_out_of_order)
    with col4:
        current_utilization = (current_occupied / total_connectors * 100) if total_connectors > 0 else 0
        label = "Current Utilization" if st.session_state.date_range_mode == "Recent Data" else "Avg Utilization"
        st.metric(f"📊 {label}", f"{current_utilization:.1f}%")

    st.markdown("---")

    # Revenue tracking
    if not sessions_df.empty:
        st.subheader("💰 Revenue Analysis")

        col1, col2, col3 = st.columns(3)
        
        date_range_days = max((end_date - start_date).days, 1)

        with col1:
            total_revenue = sessions_df['revenue_nok'].sum()
            avg_daily_revenue = total_revenue / date_range_days
            st.metric("Period Revenue", f"NOK {total_revenue:,.0f}", f"NOK {avg_daily_revenue:,.0f}/day avg")

        with col2:
            if st.session_state.date_range_mode == "Recent Data":
                # Last hour revenue
                last_hour = datetime.now() - timedelta(hours=1)
                recent_sessions = sessions_df[sessions_df['start_time'] >= last_hour]
                recent_revenue = recent_sessions['revenue_nok'].sum()
                st.metric("Last Hour Revenue", f"NOK {recent_revenue:,.0f}")
            else:
                # Peak day revenue
                daily_revenue = sessions_df.groupby(sessions_df['start_time'].dt.date)['revenue_nok'].sum()
                if not daily_revenue.empty:
                    peak_day_revenue = daily_revenue.max()
                    st.metric("Peak Day Revenue", f"NOK {peak_day_revenue:,.0f}")

        with col3:
            active_sessions = current_occupied if st.session_state.date_range_mode == "Recent Data" else len(sessions_df)
            label = "Active Sessions" if st.session_state.date_range_mode == "Recent Data" else "Total Sessions"
            st.metric(label, active_sessions)

    st.markdown("---")

    # Station grid with search
    st.subheader("Station Status Grid")

    # Search functionality
    search_term = st.text_input("🔍 Search stations by name or address", key="monitor_search")

    if search_term and not stations_df.empty:
        display_df = stations_df[
            stations_df['name'].str.contains(search_term, case=False, na=False) |
            stations_df['address'].str.contains(search_term, case=False, na=False)
        ]
    else:
        display_df = stations_df

    # Calculate station metrics for the period
    station_metrics = {}
    if not sessions_df.empty:
        station_metrics = sessions_df.groupby('station_id').agg({
            'revenue_nok': 'sum',
            'connector_id': 'count',
            'energy_kwh': 'sum'
        }).to_dict('index')

    # Display station grid
    if not display_df.empty:
        stations_per_row = 5
        for i in range(0, len(display_df), stations_per_row):
            cols = st.columns(stations_per_row)

            for j, col in enumerate(cols):
                if i + j < len(display_df):
                    station = display_df.iloc[i + j]

                    # Get metrics for this station
                    station_data = station_metrics.get(station['id'], {})
                    revenue = station_data.get('revenue_nok', 0)
                    sessions = station_data.get('connector_id', 0)

                    # Determine card appearance
                    if st.session_state.date_range_mode == "Recent Data":
                        # Real-time status
                        if not utilization_df.empty:
                            station_util = utilization_df[utilization_df['station_id'] == station['id']]
                            occupied = len(station_util[station_util['is_occupied'] == 1])
                            total = len(station_util) if len(station_util) > 0 else station['total_connectors']
                        else:
                            occupied = 0
                            total = station['total_connectors']

                        # Color based on occupancy
                        if occupied == 0:
                            card_color = "#2ecc71"
                            icon = "✅"
                            status_text = f"{occupied}/{total} occupied"
                        elif occupied < total:
                            card_color = "#f39c12"
                            icon = "🔌"
                            status_text = f"{occupied}/{total} occupied"
                        else:
                            card_color = "#e74c3c"
                            icon = "⚡"
                            status_text = f"{occupied}/{total} occupied"
                    else:
                        # Historical activity level
                        if sessions > 50:
                            card_color = "#e74c3c"  # High activity
                            icon = "🔥"
                            status_text = f"{sessions} sessions"
                        elif sessions > 20:
                            card_color = "#f39c12"  # Medium activity
                            icon = "⚡"
                            status_text = f"{sessions} sessions"
                        elif sessions > 0:
                            card_color = "#2ecc71"  # Low activity
                            icon = "🔌"
                            status_text = f"{sessions} sessions"
                        else:
                            card_color = "#95a5a6"  # No activity
                            icon = "⭕"
                            status_text = "No sessions"

                    with col:
                        st.markdown(f"""
                        <div class="station-card" style="
                            background-color: {card_color}20;
                            border: 2px solid {card_color};
                        ">
                            <h4 style="margin: 0; font-size: 0.9em;">{icon} {station['name'][:20]}...</h4>
                            <p style="margin: 5px 0; font-size: 0.8em;"><b>{status_text}</b></p>
                            <p style="margin: 5px 0; font-size: 0.8em;">⚡ {station['total_connectors']} connectors</p>
                            <p style="margin: 5px 0; font-size: 0.8em;">💰 NOK {revenue:,.0f}</p>
                            <p style="margin: 5px 0; font-size: 0.7em;">📍 {str(station.get('address', 'No address'))[:25]}...</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Activity feed
    st.markdown("---")
    if st.session_state.date_range_mode == "Recent Data":
        st.subheader("📰 Recent Session Activity")
        feed_title = "Recent sessions (most recent first)"
    else:
        st.subheader("📊 Session Activity Summary")
        feed_title = f"Session activity during analysis period ({(end_date - start_date).days} days)"

    if not sessions_df.empty:
        st.markdown(f"*{feed_title}*")
        
        # Show recent/relevant sessions
        display_sessions = sessions_df.sort_values('start_time', ascending=False).head(15)

        for _, session in display_sessions.iterrows():
            # Get station name
            if 'station_name' in session and pd.notna(session['station_name']):
                station_name = session['station_name']
            elif 'station_id' in session and pd.notna(session['station_id']):
                station_match = stations_df[stations_df['id'] == session['station_id']]
                station_name = station_match['name'].values[0] if len(station_match) > 0 else 'Unknown Station'
            else:
                station_name = 'Unknown Station'

            # Format time
            if st.session_state.date_range_mode == "Recent Data":
                time_ago = (datetime.now() - session['start_time']).total_seconds() / 60
                if time_ago < 60:
                    time_str = f"{int(time_ago)} minutes ago"
                elif time_ago < 1440:
                    time_str = f"{int(time_ago / 60)} hours ago"
                else:
                    time_str = f"{int(time_ago / 1440)} days ago"
            else:
                time_str = session['start_time'].strftime('%Y-%m-%d %H:%M')

            st.markdown(
                f"• 🔌 Session at **{station_name}** - "
                f"Duration: {session['duration_hours'] * 60:.0f} min, "
                f"Energy: {session['energy_kwh']:.1f} kWh, "
                f"Revenue: NOK {session['revenue_nok']:.0f} - "
                f"*{time_str}*"
            )
    else:
        st.info("No session activity to display for the selected period")


def show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df, start_date, end_date):
    """Show enhanced data explorer with date range context"""
    st.header("📋 Data Explorer")
    
    # Show date range info
    date_range_days = (end_date - start_date).days
    st.markdown(f"""
    <div class="date-range-info">
        <strong>Data Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} 
        ({date_range_days} days)
    </div>
    """, unsafe_allow_html=True)

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
        description = f"Connector-level utilization records for the selected period ({len(utilization_df)} records)."
    elif dataset == "Hourly Aggregations":
        df = hourly_df
        description = f"Hourly aggregated utilization data for the selected period ({len(hourly_df)} hours)."
    else:
        df = sessions_df
        description = f"Charging sessions within the selected period ({len(sessions_df)} sessions)."

    st.markdown(f"*{description}*")

    if df.empty:
        st.warning(f"No data available for {dataset} in the selected time period")
        return

    # Enhanced filters based on dataset
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)

    filters = {}

    # Dynamic filter creation
    if dataset == "Charging Stations":
        with col1:
            if 'status' in df.columns:
                status_filter = st.multiselect("Status", df['status'].unique(), key="explorer_station_status")
                if status_filter:
                    filters['status'] = status_filter

        with col2:
            if 'operator' in df.columns and df['operator'].nunique() > 1:
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
                connector_type_filter = st.multiselect("Connector Type", df['connector_type'].unique(), key="explorer_util_type")
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

        with col3:
            if 'connector_type' in df.columns:
                connector_type_filter = st.multiselect("Connector Type", df['connector_type'].unique(), key="explorer_session_type")
                if connector_type_filter:
                    filters['connector_type'] = connector_type_filter

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

    # Enhanced statistics
    st.subheader("📊 Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col3:
        filter_percentage = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Filter Coverage", f"{filter_percentage:.1f}%")
    with col4:
        if dataset == "Charging Sessions" and not filtered_df.empty and 'revenue_nok' in filtered_df.columns:
            total_revenue = filtered_df['revenue_nok'].sum()
            st.metric("Filtered Revenue", f"NOK {total_revenue:,.0f}")
        elif dataset == "Utilization Data" and not filtered_df.empty:
            occupancy_rate = filtered_df['is_occupied'].mean() * 100
            st.metric("Avg Occupancy", f"{occupancy_rate:.1f}%")
        else:
            memory_usage = filtered_df.memory_usage(deep=True).sum() / 1024 ** 2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

    # Data quality indicators
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("🔍 Data Quality Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_data = filtered_df.isnull().sum().sum()
            total_cells = filtered_df.shape[0] * filtered_df.shape[1]
            missing_percentage = (missing_data / total_cells * 100) if total_cells > 0 else 0
            st.metric("Missing Values", f"{missing_data:,}", f"{missing_percentage:.2f}%")
        
        with col2:
            duplicate_rows = filtered_df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            st.metric("Duplicate Rows", f"{duplicate_rows:,}", f"{duplicate_percentage:.2f}%")
        
        with col3:
            if 'timestamp' in filtered_df.columns or 'start_time' in filtered_df.columns:
                date_col = 'timestamp' if 'timestamp' in filtered_df.columns else 'start_time'
                if not filtered_df[date_col].empty:
                    date_range_actual = (filtered_df[date_col].max() - filtered_df[date_col].min()).days
                    st.metric("Data Span", f"{date_range_actual} days")

    # Enhanced data preview
    st.markdown("---")
    st.subheader("📋 Data Preview")

    # Column selector with smart defaults
    if st.checkbox("Select specific columns", key="explorer_column_selector"):
        # Suggest important columns based on dataset
        if dataset == "Charging Sessions":
            default_cols = ['start_time', 'end_time', 'station_name', 'connector_type', 'energy_kwh', 'revenue_nok', 'duration_hours']
        elif dataset == "Utilization Data":
            default_cols = ['timestamp', 'station_id', 'connector_id', 'connector_type', 'is_occupied', 'is_available', 'is_out_of_order']
        elif dataset == "Charging Stations":
            default_cols = ['name', 'address', 'status', 'total_connectors', 'ccs_connectors', 'latitude', 'longitude']
        else:
            default_cols = filtered_df.columns.tolist()[:10]  # First 10 columns
            
        # Filter to only existing columns
        default_cols = [col for col in default_cols if col in filtered_df.columns]
        
        selected_columns = st.multiselect(
            "Choose columns", 
            filtered_df.columns.tolist(),
            default=default_cols,
            key="explorer_columns"
        )
        if selected_columns:
            display_df = filtered_df[selected_columns]
        else:
            display_df = filtered_df
    else:
        display_df = filtered_df

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.number_input("Number of rows to display", min_value=10, max_value=1000, value=100, step=10, key="explorer_rows")
    with col2
