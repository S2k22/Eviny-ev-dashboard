"""
EV Charging Stations Analytics Dashboard
A comprehensive Streamlit dashboard for real-time monitoring and analysis
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
import os

# Page configuration
st.set_page_config(
    page_title="EV Charging Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)


# Load data functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stations_data():
    """Load charging stations data"""
    try:
        df = pd.read_csv('data/charging_stations.csv')
        # Ensure numeric columns are properly typed
        numeric_cols = ['latitude', 'longitude', 'total_connectors', 'ccs_connectors',
                        'chademo_connectors', 'type2_connectors']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.warning("Station data file not found. Creating sample data...")
        return create_sample_stations_data()
    except Exception as e:
        st.error(f"Error loading stations data: {e}")
        return create_sample_stations_data()


@st.cache_data(ttl=300)
def load_utilization_data():
    """Load utilization data"""
    try:
        df = pd.read_csv('data/utilization_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
        return df
    except FileNotFoundError:
        return create_sample_utilization_data()


@st.cache_data(ttl=300)
def load_hourly_data():
    """Load hourly aggregated data"""
    try:
        df = pd.read_csv('data/hourly_utilization.csv')
        df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
        return df
    except FileNotFoundError:
        return create_sample_hourly_data()


@st.cache_data(ttl=300)
def load_sessions_data():
    """Load charging sessions data"""
    try:
        df = pd.read_csv('data/sessions.csv')
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        return df
    except FileNotFoundError:
        st.warning("Sessions data file not found. No session data available.")
        return pd.DataFrame(columns=['connector_id', 'start_time', 'end_time',
                                     'duration_hours', 'energy_kwh', 'revenue_nok'])


# Sample data generators (for demo purposes)
def create_sample_stations_data():
    """Create sample stations data for demo"""
    np.random.seed(42)
    n_stations = 100

    # Norwegian cities and their approximate coordinates
    cities = [
        ("Oslo", 59.9139, 10.7522),
        ("Bergen", 60.3913, 5.3221),
        ("Trondheim", 63.4305, 10.3951),
        ("Stavanger", 58.9700, 5.7331),
        ("Tromsø", 69.6492, 18.9553)
    ]

    data = []
    for i in range(n_stations):
        city, lat, lon = cities[i % len(cities)]
        station = {
            'id': f'STATION_{i:03d}',
            'name': f'{city} Charging Station {i // len(cities) + 1}',
            'operator': 'Eviny',
            'status': np.random.choice(['Available', 'Occupied', 'OutOfOrder'], p=[0.7, 0.25, 0.05]),
            'latitude': lat + np.random.normal(0, 0.1),
            'longitude': lon + np.random.normal(0, 0.1),
            'total_connectors': np.random.randint(2, 8),
            'ccs_connectors': np.random.randint(0, 4),
            'chademo_connectors': np.random.randint(0, 3),
            'type2_connectors': np.random.randint(1, 5),
            'address': f'{city} Street {np.random.randint(1, 100)}'
        }
        data.append(station)

    return pd.DataFrame(data)


def create_sample_utilization_data():
    """Create sample utilization data for demo"""
    stations_df = load_stations_data()
    data = []

    # Generate 7 days of data
    base_time = datetime.now() - timedelta(days=7)

    for day in range(7):
        for hour in range(24):
            timestamp = base_time + timedelta(days=day, hours=hour)
            hourly_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)

            for _, station in stations_df.iterrows():
                for conn_idx in range(int(station['total_connectors'])):
                    # Determine connector type
                    if conn_idx < station['ccs_connectors']:
                        connector_type = 'CCS'
                        power = np.random.choice([50, 150, 350])
                    elif conn_idx < station['ccs_connectors'] + station['chademo_connectors']:
                        connector_type = 'CHAdeMO'
                        power = 50
                    else:
                        connector_type = 'Type2'
                        power = np.random.choice([11, 22])

                    # Create utilization pattern
                    hour_of_day = timestamp.hour
                    day_of_week = timestamp.weekday()

                    if day_of_week < 5:  # Weekday
                        if 7 <= hour_of_day <= 19:
                            occupied_prob = 0.5
                        else:
                            occupied_prob = 0.2
                    else:  # Weekend
                        if 10 <= hour_of_day <= 18:
                            occupied_prob = 0.3
                        else:
                            occupied_prob = 0.1

                    is_occupied = np.random.random() < occupied_prob

                    # Realistic tariff
                    if connector_type == 'CCS' and power >= 150:
                        tariff = f"NOK {np.random.uniform(3.5, 4.5):.2f} per kWh"
                    elif connector_type == 'CHAdeMO':
                        tariff = f"NOK {np.random.uniform(3.0, 4.0):.2f} per kWh"
                    else:
                        tariff = f"NOK {np.random.uniform(2.5, 3.5):.2f} per kWh"

                    record = {
                        'timestamp': timestamp,
                        'hourly_timestamp': hourly_timestamp,
                        'station_id': station['id'],
                        'connector_id': f"{station['id']}_CONN_{conn_idx}",
                        'connector_type': connector_type,
                        'status': 'Occupied' if is_occupied else 'Available',
                        'is_occupied': int(is_occupied),
                        'is_available': int(not is_occupied),
                        'is_out_of_order': 0,
                        'power': power,
                        'tariff': tariff
                    }
                    data.append(record)

    return pd.DataFrame(data)


def create_sample_hourly_data():
    """Create sample hourly aggregated data"""
    util_df = load_utilization_data()

    hourly_agg = util_df.groupby(['hourly_timestamp', 'station_id']).agg({
        'is_available': 'sum',
        'is_occupied': 'sum',
        'is_out_of_order': 'sum',
        'connector_id': 'count'
    }).reset_index()

    hourly_agg.rename(columns={'connector_id': 'total_connectors'}, inplace=True)
    hourly_agg['occupancy_rate'] = hourly_agg['is_occupied'] / hourly_agg['total_connectors']
    hourly_agg['availability_rate'] = hourly_agg['is_available'] / hourly_agg['total_connectors']

    return hourly_agg


# Main dashboard
def main():
    # Title
    st.markdown('<h1 class="main-header">⚡ EV Charging Stations Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    stations_df = load_stations_data()
    utilization_df = load_utilization_data()
    hourly_df = load_hourly_data()
    sessions_df = load_sessions_data()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=EV+Charging+Analytics", width=300)
        st.markdown("### 🔌 Navigation")

        page = st.radio(
            "Select Dashboard",
            ["📊 Overview", "🗺️ Station Map", "📈 Utilization Analytics", "⚡ Real-time Monitor", "📋 Data Explorer"]
        )

        st.markdown("---")
        st.markdown("### 🕐 Last Updated")
        st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Stations", len(stations_df))
        st.metric("Active Charging", utilization_df[utilization_df['is_occupied'] == 1]['station_id'].nunique())
        st.metric("Total Connectors", stations_df['total_connectors'].sum())

    # Page routing
    if page == "📊 Overview":
        show_overview(stations_df, utilization_df, hourly_df, sessions_df)
    elif page == "🗺️ Station Map":
        show_station_map(stations_df, utilization_df)
    elif page == "📈 Utilization Analytics":
        show_utilization_analytics(utilization_df, hourly_df, sessions_df)
    elif page == "⚡ Real-time Monitor":
        show_realtime_monitor(stations_df, utilization_df, sessions_df)
    elif page == "📋 Data Explorer":
        show_data_explorer(stations_df, utilization_df, hourly_df, sessions_df)


def show_overview(stations_df, utilization_df, hourly_df, sessions_df):
    """Show overview dashboard"""
    st.header("📊 Overview Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_stations = len(stations_df)
        available_stations = len(stations_df[stations_df['status'] == 'Available'])
        st.metric(
            "Available Stations",
            f"{available_stations}/{total_stations}",
            f"{(available_stations / total_stations * 100):.1f}%"
        )

    with col2:
        total_connectors = stations_df['total_connectors'].sum()
        latest_util = utilization_df.sort_values('timestamp').groupby(['station_id', 'connector_id']).last()
        occupied_connectors = len(latest_util[latest_util['is_occupied'] == 1])

        # Count occupied connectors as active sessions
        active_sessions = occupied_connectors

        st.metric(
            "Active Sessions",
            f"{active_sessions}",
            f"{(occupied_connectors / total_connectors * 100):.1f}% utilization",
            help="Each occupied connector is counted as an active session"
        )

    with col3:
        avg_occupancy = hourly_df['occupancy_rate'].mean() * 100 if 'occupancy_rate' in hourly_df.columns else 0
        st.metric(
            "Avg Occupancy Rate",
            f"{avg_occupancy:.1f}%",
            f"{avg_occupancy - 20:.1f}%" if avg_occupancy > 20 else f"{avg_occupancy - 20:.1f}%"
        )

    with col4:
        # Use actual revenue from sessions data
        if not sessions_df.empty:
            # Calculate daily revenue (last 24 hours)
            last_24h = datetime.now() - timedelta(hours=24)
            recent_sessions = sessions_df[sessions_df['end_time'] >= last_24h]
            daily_revenue_nok = recent_sessions['revenue_nok'].sum()
            daily_revenue_usd = daily_revenue_nok / 10.5  # Convert to USD

            # Calculate revenue change
            prev_24h_start = last_24h - timedelta(hours=24)
            prev_sessions = sessions_df[
                (sessions_df['end_time'] >= prev_24h_start) &
                (sessions_df['end_time'] < last_24h)
                ]
            prev_revenue = prev_sessions['revenue_nok'].sum()

            if prev_revenue > 0:
                revenue_change = ((daily_revenue_nok - prev_revenue) / prev_revenue) * 100
            else:
                revenue_change = 0
        else:
            daily_revenue_usd = 0
            daily_revenue_nok = 0
            revenue_change = 0

        st.metric(
            "Daily Revenue (24h)",
            f"${daily_revenue_usd:,.0f}",
            f"{revenue_change:+.1f}%",
            help=f"NOK {daily_revenue_nok:,.0f}"
        )

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Station status pie chart
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

    # Charts row 2
    st.markdown("---")

    # Hourly utilization pattern with actual session counts
    if not sessions_df.empty:
        # Count sessions by hour
        sessions_df['hour'] = sessions_df['start_time'].dt.hour
        sessions_per_hour = sessions_df.groupby('hour').size()

        # Ensure all hours are represented
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(sessions_per_hour)
        sessions_per_hour = all_hours
    else:
        sessions_per_hour = pd.Series(0, index=range(24))

    # Create figure with secondary y-axis
    fig_line = make_subplots(specs=[[{"secondary_y": True}]])

    # Add session counts (bar chart)
    fig_line.add_trace(
        go.Bar(
            x=sessions_per_hour.index,
            y=sessions_per_hour.values,
            name='Total Sessions',
            marker_color='#3498db',
            opacity=0.7
        ),
        secondary_y=False
    )

    # Add occupancy data (line chart)
    hourly_occupancy = utilization_df.groupby(utilization_df['hourly_timestamp'].dt.hour)[
        'is_occupied'].sum().reset_index()
    fig_line.add_trace(
        go.Scatter(
            x=hourly_occupancy['hourly_timestamp'],
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
        title='24-Hour Pattern: Sessions Started vs Connector Occupancy',
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.2)
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # Station performance table with actual revenue
    st.markdown("---")
    st.subheader("🏆 Top Performing Stations (by Revenue)")

    if not sessions_df.empty:
        # Check if station_id column exists, if not extract from connector_id
        if 'station_id' not in sessions_df.columns:
            sessions_df['station_id'] = sessions_df['connector_id'].str.extract(r'(STATION_\d+)')

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
            right_on='id'
        )

        # Sort by revenue and get top 10
        station_performance = station_performance.sort_values('total_revenue', ascending=False).head(10)

        st.dataframe(
            station_performance[['name', 'address', 'total_revenue', 'total_energy', 'session_count']].rename(columns={
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


def show_station_map(stations_df, utilization_df):
    """Show interactive station map with revenue data"""
    st.header("🗺️ Charging Station Map")

    # Load sessions data for revenue info
    sessions_df = load_sessions_data()

    # Calculate station revenue if sessions data available
    if not sessions_df.empty:
        # Check if station_id exists, if not extract from connector_id
        if 'station_id' not in sessions_df.columns:
            sessions_df['station_id'] = sessions_df['connector_id'].str.extract(r'(STATION_\d+)')
        station_revenue = sessions_df.groupby('station_id')['revenue_nok'].sum().to_dict()
    else:
        station_revenue = {}

    # Get out of order stations from utilization data
    latest_util = utilization_df.sort_values('timestamp').groupby(['station_id', 'connector_id']).last()

    # Count out of order connectors by station
    out_of_order_by_station = {}
    total_connectors_by_station = {}

    for station_id in latest_util.index.get_level_values('station_id').unique():
        station_connectors = latest_util[latest_util.index.get_level_values('station_id') == station_id]
        total_connectors_by_station[station_id] = len(station_connectors)
        out_of_order_count = len(station_connectors[station_connectors['status'] == 'OutOfOrder'])
        if out_of_order_count > 0:
            out_of_order_by_station[station_id] = out_of_order_count

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=['Available', 'Occupied', 'OutOfOrder'],
            default=['Available', 'Occupied', 'OutOfOrder']
        )

    with col2:
        connector_filter = st.slider(
            "Minimum Connectors",
            min_value=1,
            max_value=int(stations_df['total_connectors'].max()),
            value=1
        )

    with col3:
        map_style = st.selectbox(
            "Map Style",
            ['OpenStreetMap', 'Stamen Terrain', 'CartoDB dark_matter']
        )

    # Filter data
    filtered_df = stations_df[
        (stations_df['status'].isin(status_filter)) &
        (stations_df['total_connectors'] >= connector_filter)
        ]

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
            # Get detailed connector status from utilization data
            station_connectors = latest_util[latest_util.index.get_level_values('station_id') == station['id']]
            occupied_count = len(station_connectors[station_connectors['is_occupied'] == 1])
            available_count = len(station_connectors[station_connectors['is_available'] == 1])
            out_of_order_count = out_of_order_by_station.get(station['id'], 0)
            total_station_connectors = total_connectors_by_station.get(station['id'], station['total_connectors'])

            # Determine station status and color
            # Only show as OutOfOrder if ALL connectors are out of order
            if out_of_order_count > 0 and out_of_order_count == total_station_connectors:
                color = 'red'
                status_text = 'Out of Order'
            elif occupied_count == total_station_connectors - out_of_order_count and total_station_connectors > 0:
                color = 'orange'
                status_text = 'Fully Occupied'
            elif available_count > 0:
                color = 'green'
                status_text = 'Available'
            else:
                color = 'gray'
                status_text = 'Unknown'

            # Get revenue for this station
            revenue = station_revenue.get(station['id'], 0)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>{station['name']}</h4>
                <p><b>Status:</b> {status_text}</p>
                <p><b>Address:</b> {station['address']}</p>
                <p><b>Connectors:</b> {station['total_connectors']}</p>
                <p><b>Connector Status:</b><br>
                   - Available: {available_count}<br>
                   - Occupied: {occupied_count}<br>
                   - Out of Order: {out_of_order_count}</p>
                <p><b>Types:</b><br>
                   - CCS: {station['ccs_connectors']}<br>
                   - CHAdeMO: {station['chademo_connectors']}<br>
                   - Type 2: {station['type2_connectors']}</p>
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
        st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])

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
            # Count out of order connectors in filtered stations
            filtered_oo = sum(out_of_order_by_station.get(sid, 0) for sid in filtered_df['id'])
            st.metric("Out of Order Connectors", filtered_oo)
    else:
        st.warning("No stations match the selected filters")


def show_utilization_analytics(utilization_df, hourly_df, sessions_df):
    """Show detailed utilization analytics with actual session data"""
    st.header("📈 Utilization Analytics")

    # Time range selector
    time_range = st.select_slider(
        "Select Time Range",
        options=["Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 7 Days", "All Data"],
        value="Last 24 Hours"
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
        time_filter = utilization_df['timestamp'].min()

    filtered_util = utilization_df[utilization_df['timestamp'] >= time_filter]
    filtered_hourly = hourly_df[hourly_df['hourly_timestamp'] >= time_filter]
    filtered_sessions = sessions_df[sessions_df['end_time'] >= time_filter] if not sessions_df.empty else sessions_df

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Hourly Patterns", "📈 Trends", "⚡ Power & Revenue", "💰 Session Analysis"])

    with tab1:
        # Hourly utilization heatmap
        st.subheader("Hourly Utilization Heatmap")

        # Get latest utilization status
        latest_util = filtered_util.sort_values('timestamp').groupby(['station_id', 'connector_id']).last()

        # Prepare data for heatmap
        heatmap_data = filtered_util.copy()
        heatmap_data['hour'] = heatmap_data['hourly_timestamp'].dt.hour
        heatmap_data['day'] = heatmap_data['hourly_timestamp'].dt.day_name()

        # Create pivot table
        pivot_table = heatmap_data.groupby(['day', 'hour'])['is_occupied'].mean().reset_index()
        pivot_table = pivot_table.pivot(index='day', columns='hour', values='is_occupied')

        # Ensure all hours are present
        for hour in range(24):
            if hour not in pivot_table.columns:
                pivot_table[hour] = 0

        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

        # Define day order
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
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Occupancy: %{z:.2%}<extra></extra>'
        ))

        fig_heatmap.update_layout(
            title='Weekly Utilization Pattern by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400,
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Peak hours analysis with actual sessions
        col1, col2 = st.columns(2)

        with col1:
            if not filtered_sessions.empty:
                peak_hour_sessions = filtered_sessions.groupby(filtered_sessions['start_time'].dt.hour).size()
                peak_hour = peak_hour_sessions.idxmax()
                peak_count = peak_hour_sessions.max()
                st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_count} sessions started")
            else:
                st.metric("Peak Hour", "No data", "0 sessions")

        with col2:
            if not filtered_sessions.empty:
                # Count currently occupied connectors as active sessions
                current_active = len(latest_util[latest_util['is_occupied'] == 1])
                total_completed = len(filtered_sessions)
                st.metric("Sessions", f"{total_completed} completed", f"{current_active} active now")
            else:
                current_active = len(latest_util[latest_util['is_occupied'] == 1])
                st.metric("Sessions", "0 completed", f"{current_active} active now")

    with tab2:
        # Utilization trends
        st.subheader("Utilization Trends")

        # Multi-metric trend chart
        trend_data = filtered_hourly.groupby('hourly_timestamp').agg({
            'is_occupied': 'sum',
            'is_available': 'sum',
            'occupancy_rate': 'mean'
        }).reset_index()

        fig_trend = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Connector Status Over Time', 'Average Occupancy Rate'),
            vertical_spacing=0.1
        )

        # Add traces
        fig_trend.add_trace(
            go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['is_occupied'],
                       name='Occupied', line=dict(color='#e74c3c')),
            row=1, col=1
        )
        fig_trend.add_trace(
            go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['is_available'],
                       name='Available', line=dict(color='#2ecc71')),
            row=1, col=1
        )
        fig_trend.add_trace(
            go.Scatter(x=trend_data['hourly_timestamp'], y=trend_data['occupancy_rate'] * 100,
                       name='Occupancy %', line=dict(color='#3498db', width=3)),
            row=2, col=1
        )

        fig_trend.update_yaxes(title_text="Count", row=1, col=1)
        fig_trend.update_yaxes(title_text="Percentage", row=2, col=1)
        fig_trend.update_layout(height=600, showlegend=True)

        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        # Power and Revenue Analysis using actual session data
        st.subheader("Power Consumption & Revenue Analysis")

        if not filtered_sessions.empty:
            # Check if station_id exists, otherwise extract from connector_id
            if 'station_id' not in filtered_sessions.columns:
                filtered_sessions['station_id'] = filtered_sessions['connector_id'].str.extract(r'(STATION_\d+)')

            # Extract connector type from connector_id
            filtered_sessions['connector_type'] = filtered_sessions['connector_id'].str.extract(
                r'_(CCS|CHAdeMO|Type2)_')

            # If connector type not in ID, merge with utilization data
            if filtered_sessions['connector_type'].isna().all():
                connector_types = filtered_util.groupby('connector_id')['connector_type'].first()
                filtered_sessions = filtered_sessions.merge(
                    connector_types,
                    left_on='connector_id',
                    right_index=True,
                    how='left',
                    suffixes=('', '_util')
                )
                if 'connector_type_util' in filtered_sessions.columns:
                    filtered_sessions['connector_type'] = filtered_sessions['connector_type_util']
                    filtered_sessions = filtered_sessions.drop('connector_type_util', axis=1)

            # Group by connector type.drop('connector_type_util', axis=1, inplace=True)

            # Group by connector type
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
                fig_revenue_type.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>Revenue: NOK %{value:,.0f}<br>Sessions: %{customdata}<extra></extra>',
                    customdata=type_stats['session_count']
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
                fig_energy_type.update_traces(
                    text=type_stats['total_energy'].round(0),
                    textposition='auto'
                )
                st.plotly_chart(fig_energy_type, use_container_width=True)

            # Detailed metrics
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

            # Additional power metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_power_per_session = filtered_sessions['energy_kwh'].sum() / filtered_sessions[
                    'duration_hours'].sum() if filtered_sessions['duration_hours'].sum() > 0 else 0
                st.metric("Avg Power Delivered", f"{avg_power_per_session:.1f} kW",
                          help="Average power delivered per session")

            with col2:
                avg_energy_per_session = filtered_sessions['energy_kwh'].mean()
                st.metric("Avg Energy/Session", f"{avg_energy_per_session:.1f} kWh")

            with col3:
                total_sessions = len(filtered_sessions)
                st.metric("Total Sessions", f"{total_sessions:,}")

            with col4:
                if total_energy > 0:
                    avg_tariff = total_revenue / total_energy
                    st.metric("Effective Tariff", f"NOK {avg_tariff:.2f}/kWh")
                else:
                    st.metric("Effective Tariff", "N/A")

            # Hourly revenue pattern
            st.markdown("---")
            st.subheader("Revenue Pattern by Hour")

            # Group sessions by hour
            filtered_sessions['hour'] = filtered_sessions['start_time'].dt.hour
            hourly_stats = filtered_sessions.groupby('hour').agg({
                'revenue_nok': 'sum',
                'energy_kwh': 'sum',
                'connector_id': 'count'
            }).reset_index()
            hourly_stats.columns = ['hour', 'revenue', 'energy', 'sessions']

            # Create multi-axis chart
            fig_hourly = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )

            # Add revenue bars
            fig_hourly.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['revenue'],
                    name='Revenue (NOK)',
                    marker_color='#2ecc71',
                    opacity=0.7,
                    text=hourly_stats['revenue'].round(0),
                    textposition='auto'
                ),
                secondary_y=False
            )

            # Add session count line
            fig_hourly.add_trace(
                go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['sessions'],
                    name='Sessions Started',
                    mode='lines+markers',
                    marker=dict(size=8, color='#e74c3c'),
                    line=dict(width=3, color='#e74c3c')
                ),
                secondary_y=True
            )

            fig_hourly.update_xaxes(title_text="Hour of Day", tickmode='linear', tick0=0, dtick=1)
            fig_hourly.update_yaxes(title_text="Revenue (NOK)", secondary_y=False)
            fig_hourly.update_yaxes(title_text="Sessions Started", secondary_y=True)
            fig_hourly.update_layout(
                title="Hourly Revenue and Session Count",
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.2)
            )

            st.plotly_chart(fig_hourly, use_container_width=True)

        else:
            st.info("No session data available for the selected time range")

    with tab4:
        # Session Analysis
        st.subheader("Detailed Session Analysis")

        if not filtered_sessions.empty:
            # Session duration distribution
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

            # Time series of sessions
            st.markdown("---")
            st.subheader("Session Timeline")

            # Create daily aggregates
            filtered_sessions['date'] = filtered_sessions['start_time'].dt.date
            daily_stats = filtered_sessions.groupby('date').agg({
                'connector_id': 'count',
                'revenue_nok': 'sum',
                'energy_kwh': 'sum',
                'duration_hours': 'mean'
            }).reset_index()
            daily_stats.columns = ['date', 'sessions', 'revenue', 'energy', 'avg_duration']

            # Create time series chart
            fig_timeline = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Sessions and Revenue', 'Daily Energy Delivered'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )

            # Sessions and revenue
            fig_timeline.add_trace(
                go.Bar(
                    x=daily_stats['date'],
                    y=daily_stats['sessions'],
                    name='Sessions',
                    marker_color='#3498db',
                    opacity=0.7
                ),
                row=1, col=1, secondary_y=False
            )

            fig_timeline.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['revenue'],
                    name='Revenue (NOK)',
                    mode='lines+markers',
                    line=dict(color='#2ecc71', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1, secondary_y=True
            )

            # Energy
            fig_timeline.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['energy'],
                    name='Energy (kWh)',
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )

            fig_timeline.update_yaxes(title_text="Sessions", row=1, col=1, secondary_y=False)
            fig_timeline.update_yaxes(title_text="Revenue (NOK)", row=1, col=1, secondary_y=True)
            fig_timeline.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
            fig_timeline.update_xaxes(title_text="Date", row=2, col=1)
            fig_timeline.update_layout(height=700, showlegend=True)

            st.plotly_chart(fig_timeline, use_container_width=True)

            # Top performing connectors
            st.markdown("---")
            st.subheader("🏆 Top Performing Connectors")

            top_connectors = filtered_sessions.groupby('connector_id').agg({
                'revenue_nok': 'sum',
                'energy_kwh': 'sum',
                'duration_hours': ['count', 'mean']
            }).round(2)

            top_connectors.columns = ['total_revenue', 'total_energy', 'session_count', 'avg_duration']
            top_connectors = top_connectors.sort_values('total_revenue', ascending=False).head(10)
            top_connectors.reset_index(inplace=True)

            # Check if station_id exists in sessions, otherwise extract from connector_id
            if 'station_id' in filtered_sessions.columns:
                # Create a mapping of connector_id to station_id
                connector_station_map = filtered_sessions.groupby('connector_id')['station_id'].first().to_dict()
                top_connectors['station'] = top_connectors['connector_id'].map(connector_station_map)
            else:
                # Extract station from connector_id for display
                top_connectors['station'] = top_connectors['connector_id'].str.extract(r'(STATION_\d+)')

            st.dataframe(
                top_connectors[['connector_id', 'station', 'total_revenue', 'total_energy', 'session_count']].rename(
                    columns={
                        'connector_id': 'Connector ID',
                        'station': 'Station',
                        'total_revenue': 'Revenue (NOK)',
                        'total_energy': 'Energy (kWh)',
                        'session_count': 'Sessions'
                    }),
                hide_index=True,
                use_container_width=True
            )

        else:
            st.info("No session data available for analysis")


def show_realtime_monitor(stations_df, utilization_df, sessions_df):
    """Show real-time monitoring dashboard"""
    st.header("⚡ Real-time Station Monitor")

    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)")

    if auto_refresh:
        st.empty()
        import time
        time.sleep(30)
        st.rerun()

    # Current status overview
    col1, col2, col3, col4 = st.columns(4)

    latest_util = utilization_df.sort_values('timestamp').groupby(['station_id', 'connector_id']).last()
    current_available = len(latest_util[latest_util['is_available'] == 1])
    current_occupied = len(latest_util[latest_util['is_occupied'] == 1])
    current_out_of_order = len(latest_util[latest_util['is_out_of_order'] == 1])
    total_connectors = len(latest_util)

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
            # Count occupied connectors as active sessions
            occupied_connectors = len(latest_util[latest_util['is_occupied'] == 1])
            st.metric("Active Sessions", occupied_connectors,
                      help="Each occupied connector represents an active charging session")

    st.markdown("---")

    # Real-time station grid
    st.subheader("Station Status Grid")

    # Search functionality
    search_term = st.text_input("🔍 Search stations by name or address")

    if search_term:
        display_df = stations_df[
            stations_df['name'].str.contains(search_term, case=False) |
            stations_df['address'].str.contains(search_term, case=False)
            ]
    else:
        display_df = stations_df

    # Display grid
    stations_per_row = 5
    for i in range(0, len(display_df), stations_per_row):
        cols = st.columns(stations_per_row)

        for j, col in enumerate(cols):
            if i + j < len(display_df):
                station = display_df.iloc[i + j]

                # Get current status for this station
                station_util = latest_util[latest_util.index.get_level_values('station_id') == station['id']]
                occupied = len(station_util[station_util['is_occupied'] == 1])
                total = len(station_util)

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
                    <div style="
                        background-color: {card_color}20;
                        border: 2px solid {card_color};
                        border-radius: 10px;
                        padding: 10px;
                        margin: 5px;
                        text-align: center;
                        height: 150px;
                    ">
                        <h4 style="margin: 0; font-size: 0.9em;">{icon} {station['name'][:20]}...</h4>
                        <p style="margin: 5px 0; font-size: 0.8em;"><b>{occupied}/{total} occupied</b></p>
                        <p style="margin: 5px 0; font-size: 0.8em;">⚡ {station['total_connectors']} connectors</p>
                        <p style="margin: 5px 0; font-size: 0.7em;">📍 {str(station['address'])[:25] if pd.notna(station['address']) else 'No address'}...</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Recent activity feed
    st.markdown("---")
    st.subheader("📰 Recent Session Activity")

    if not sessions_df.empty:
        # Get recent sessions
        recent_sessions = sessions_df.sort_values('start_time', ascending=False).head(20)

        for _, session in recent_sessions.iterrows():
            # Get station from session data or extract from connector_id
            if 'station_id' in session and pd.notna(session['station_id']):
                station_id = session['station_id']
            else:
                station_id = session['connector_id'].split('_CONN_')[0]

            station_name = stations_df[stations_df['id'] == station_id]['name'].values[0] if len(
                stations_df[stations_df['id'] == station_id]) > 0 else 'Unknown Station'

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
    """Show data explorer interface"""
    st.header("📋 Data Explorer")

    # Dataset selector
    dataset = st.selectbox(
        "Select Dataset",
        ["Charging Stations", "Utilization Data", "Hourly Aggregations", "Charging Sessions"]
    )

    # Get the appropriate dataframe
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
            status_filter = st.multiselect("Status", df['status'].unique())
            if status_filter:
                filters['status'] = status_filter

        with col2:
            operator_filter = st.multiselect("Operator", df['operator'].unique())
            if operator_filter:
                filters['operator'] = operator_filter

        with col3:
            connector_range = st.slider(
                "Total Connectors",
                int(df['total_connectors'].min()),
                int(df['total_connectors'].max()),
                (int(df['total_connectors'].min()), int(df['total_connectors'].max()))
            )
            filters['total_connectors'] = connector_range

    elif dataset == "Utilization Data":
        with col1:
            status_filter = st.multiselect("Status", df['status'].unique())
            if status_filter:
                filters['status'] = status_filter

        with col2:
            connector_type_filter = st.multiselect("Connector Type", df['connector_type'].unique())
            if connector_type_filter:
                filters['connector_type'] = connector_type_filter

        with col3:
            occupied_filter = st.radio("Occupancy", ["All", "Occupied", "Available"])
            if occupied_filter == "Occupied":
                filters['is_occupied'] = 1
            elif occupied_filter == "Available":
                filters['is_occupied'] = 0

    elif dataset == "Charging Sessions":
        with col1:
            # Revenue range filter
            revenue_range = st.slider(
                "Revenue Range (NOK)",
                float(df['revenue_nok'].min()),
                float(df['revenue_nok'].max()),
                (float(df['revenue_nok'].min()), float(df['revenue_nok'].max()))
            )
            filters['revenue_nok'] = revenue_range

        with col2:
            # Duration filter
            duration_range = st.slider(
                "Duration (hours)",
                float(df['duration_hours'].min()),
                float(df['duration_hours'].max()),
                (float(df['duration_hours'].min()), float(df['duration_hours'].max()))
            )
            filters['duration_hours'] = duration_range

        with col3:
            # Date filter
            date_range = st.date_input(
                "Date Range",
                value=(df['start_time'].min().date(), df['start_time'].max().date()),
                min_value=df['start_time'].min().date(),
                max_value=df['start_time'].max().date()
            )
            if len(date_range) == 2:
                filters['date_range'] = date_range

    # Apply filters
    filtered_df = df.copy()

    for col, value in filters.items():
        if col in ['total_connectors', 'revenue_nok', 'duration_hours']:
            filtered_df = filtered_df[
                (filtered_df[col] >= value[0]) &
                (filtered_df[col] <= value[1])
                ]
        elif col == 'date_range':
            filtered_df = filtered_df[
                (filtered_df['start_time'].dt.date >= value[0]) &
                (filtered_df['start_time'].dt.date <= value[1])
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
        if dataset == "Charging Sessions" and not filtered_df.empty:
            total_revenue = filtered_df['revenue_nok'].sum()
            st.metric("Total Revenue", f"NOK {total_revenue:,.0f}")
        elif 'timestamp' in filtered_df.columns:
            date_range = f"{filtered_df['timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['timestamp'].max().strftime('%Y-%m-%d')}"
            st.metric("Date Range", date_range)
        else:
            st.metric("Memory Usage", f"{filtered_df.memory_usage().sum() / 1024 ** 2:.1f} MB")

    # Data preview
    st.subheader("📋 Data Preview")

    # Column selector
    if st.checkbox("Select specific columns"):
        selected_columns = st.multiselect("Choose columns", filtered_df.columns.tolist(),
                                          default=filtered_df.columns.tolist())
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.number_input("Number of rows to display", min_value=10, max_value=1000, value=100, step=10)
    with col2:
        sort_by = st.selectbox("Sort by", display_df.columns.tolist())
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

    # Sort and display
    display_df = display_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

    st.dataframe(
        display_df.head(n_rows),
        use_container_width=True,
        hide_index=True
    )

    # Download options
    st.subheader("💾 Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{dataset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
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
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install xlsxwriter for Excel export: pip install xlsxwriter")

    with col3:
        # Summary statistics
        if st.button("📊 Generate Summary Report"):
            summary = display_df.describe(include='all').to_string()
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name=f"{dataset.lower().replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ EV Charging Analytics Dashboard | Built with Streamlit</p>
        <p>Data updates every hour | Last refresh: {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    add_footer()
