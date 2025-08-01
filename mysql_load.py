"""
Improved MySQL Load module for ETL pipeline
Enhanced with better error handling and debugging
"""
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import text, inspect
from mysql_config import engine, get_connection
import time

logger = logging.getLogger(__name__)


def test_connection():
    """Test database connection before operations"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


def ensure_tables_exist():
    """Ensure all required tables exist with proper schema"""

    table_schemas = {
        'charging_stations': """
            CREATE TABLE IF NOT EXISTS charging_stations (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                operator VARCHAR(100) DEFAULT 'Eviny',
                status ENUM('Available', 'Occupied', 'OutOfOrder') NOT NULL,
                address TEXT,
                description TEXT,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                total_connectors INT DEFAULT 0,
                ccs_connectors INT DEFAULT 0,
                chademo_connectors INT DEFAULT 0,
                type2_connectors INT DEFAULT 0,
                other_connectors INT DEFAULT 0,
                amenities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_location (latitude, longitude),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,

        'utilization_data': """
            CREATE TABLE IF NOT EXISTS utilization_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                station_id VARCHAR(50) NOT NULL,
                connector_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                is_occupied TINYINT(1) DEFAULT 0,
                is_available TINYINT(1) DEFAULT 0,
                is_out_of_order TINYINT(1) DEFAULT 0,
                total_connectors INT DEFAULT 0,
                occupancy_rate DECIMAL(5,4) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_station_connector (station_id, connector_id),
                INDEX idx_timestamp (timestamp),
                INDEX idx_status (is_occupied, is_available, is_out_of_order),
                FOREIGN KEY (station_id) REFERENCES charging_stations(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,

        'hourly_utilization': """
            CREATE TABLE IF NOT EXISTS hourly_utilization (
                id INT AUTO_INCREMENT PRIMARY KEY,
                hourly_timestamp TIMESTAMP NOT NULL,
                station_id VARCHAR(50),
                is_occupied INT DEFAULT 0,
                is_available INT DEFAULT 0,
                is_out_of_order INT DEFAULT 0,
                total_connectors INT DEFAULT 0,
                occupancy_rate DECIMAL(5,4) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_hourly_timestamp (hourly_timestamp),
                INDEX idx_station (station_id),
                FOREIGN KEY (station_id) REFERENCES charging_stations(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,

        'charging_sessions': """
            CREATE TABLE IF NOT EXISTS charging_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) UNIQUE,
                station_id VARCHAR(50) NOT NULL,
                connector_id VARCHAR(50),
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                duration_hours DECIMAL(6,3) DEFAULT 0,
                energy_kwh DECIMAL(8,3) DEFAULT 0,
                revenue_nok DECIMAL(10,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_station (station_id),
                INDEX idx_time_range (start_time, end_time),
                INDEX idx_end_time (end_time),
                FOREIGN KEY (station_id) REFERENCES charging_stations(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    }

    try:
        with engine.connect() as conn:
            for table_name, schema in table_schemas.items():
                conn.execute(text(schema))
                logger.info(f"Ensured table {table_name} exists")

            conn.commit()

            # Create the view for latest connector status
            create_latest_status_view(conn)

        return True

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False


def create_latest_status_view(conn):
    """Create or recreate the latest_connector_status view"""
    try:
        # Drop existing view
        conn.execute(text("DROP VIEW IF EXISTS latest_connector_status"))

        # Create new view
        view_sql = text("""
            CREATE VIEW latest_connector_status AS
            SELECT 
                u1.*
            FROM utilization_data u1
            INNER JOIN (
                SELECT 
                    station_id, 
                    connector_id, 
                    MAX(timestamp) as max_timestamp
                FROM utilization_data
                GROUP BY station_id, connector_id
            ) u2 ON u1.station_id = u2.station_id 
                 AND u1.connector_id = u2.connector_id 
                 AND u1.timestamp = u2.max_timestamp
        """)

        conn.execute(view_sql)
        logger.info("Created latest_connector_status view")

    except Exception as e:
        logger.error(f"Error creating view: {e}")


def save_to_mysql(df, table_name, mode='append'):
    """
    Save DataFrame to MySQL table with improved error handling
    """
    if df is None or df.empty:
        logger.warning(f"No data to save to {table_name}")
        return False

    # Test connection first
    if not test_connection():
        logger.error("Database connection failed")
        return False

    # Ensure tables exist
    if not ensure_tables_exist():
        logger.error("Failed to ensure tables exist")
        return False

    try:
        # Log data info
        logger.info(f"Saving {len(df)} rows to {table_name}")
        logger.debug(f"Columns: {list(df.columns)}")

        # Special handling for charging_stations (upsert behavior)
        if table_name == 'charging_stations':
            return upsert_stations(df)

        # Clean data before saving
        df_clean = clean_dataframe(df, table_name)

        # For other tables, use standard insert with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df_clean.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists=mode,
                    index=False,
                    method='multi',
                    chunksize=1000
                )

                logger.info(f"Successfully saved {len(df_clean)} records to {table_name}")
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {table_name}: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

    except Exception as e:
        logger.error(f"Error saving to MySQL table {table_name}: {e}")
        logger.error(f"DataFrame shape: {df.shape}")
        logger.error(f"DataFrame dtypes: {df.dtypes}")
        return False


def clean_dataframe(df, table_name):
    """Clean DataFrame before saving to database"""
    df_clean = df.copy()

    # Handle NaN values
    df_clean = df_clean.fillna({
        'operator': 'Eviny',
        'description': '',
        'amenities': '',
        'address': '',
        'total_connectors': 0,
        'ccs_connectors': 0,
        'chademo_connectors': 0,
        'type2_connectors': 0,
        'other_connectors': 0,
        'occupancy_rate': 0,
        'is_occupied': 0,
        'is_available': 0,
        'is_out_of_order': 0,
        'energy_kwh': 0,
        'revenue_nok': 0,
        'duration_hours': 0
    })

    # Ensure proper data types
    int_columns = ['total_connectors', 'ccs_connectors', 'chademo_connectors',
                   'type2_connectors', 'other_connectors', 'is_occupied',
                   'is_available', 'is_out_of_order']

    for col in int_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    # Handle string length limits
    string_limits = {
        'id': 50,
        'name': 255,
        'operator': 100,
        'connector_id': 50,
        'session_id': 100
    }

    for col, limit in string_limits.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str[:limit]

    return df_clean


def upsert_stations(df):
    """
    Upsert stations data with better error handling
    """
    try:
        with engine.connect() as conn:
            success_count = 0

            for _, row in df.iterrows():
                try:
                    # Prepare the upsert query
                    query = text("""
                        INSERT INTO charging_stations 
                        (id, name, operator, status, address, description, 
                         latitude, longitude, total_connectors, ccs_connectors, 
                         chademo_connectors, type2_connectors, other_connectors, amenities)
                        VALUES 
                        (:id, :name, :operator, :status, :address, :description,
                         :latitude, :longitude, :total_connectors, :ccs_connectors,
                         :chademo_connectors, :type2_connectors, :other_connectors, :amenities)
                        ON DUPLICATE KEY UPDATE
                            name = VALUES(name),
                            operator = VALUES(operator),
                            status = VALUES(status),
                            address = VALUES(address),
                            description = VALUES(description),
                            latitude = VALUES(latitude),
                            longitude = VALUES(longitude),
                            total_connectors = VALUES(total_connectors),
                            ccs_connectors = VALUES(ccs_connectors),
                            chademo_connectors = VALUES(chademo_connectors),
                            type2_connectors = VALUES(type2_connectors),
                            other_connectors = VALUES(other_connectors),
                            amenities = VALUES(amenities),
                            last_updated = CURRENT_TIMESTAMP
                    """)

                    params = {
                        'id': str(row['id'])[:50],
                        'name': str(row['name'])[:255],
                        'operator': str(row.get('operator', 'Eviny'))[:100],
                        'status': str(row['status']),
                        'address': str(row.get('address', ''))[:500],
                        'description': str(row.get('description', ''))[:1000],
                        'latitude': float(row.get('latitude', 0)) if pd.notna(row.get('latitude')) else None,
                        'longitude': float(row.get('longitude', 0)) if pd.notna(row.get('longitude')) else None,
                        'total_connectors': int(row.get('total_connectors', 0)),
                        'ccs_connectors': int(row.get('ccs_connectors', 0)),
                        'chademo_connectors': int(row.get('chademo_connectors', 0)),
                        'type2_connectors': int(row.get('type2_connectors', 0)),
                        'other_connectors': int(row.get('other_connectors', 0)),
                        'amenities': str(row.get('amenities', ''))[:1000]
                    }

                    conn.execute(query, params)
                    success_count += 1

                except Exception as row_error:
                    logger.error(f"Error upserting station {row.get('id', 'unknown')}: {row_error}")
                    continue

            conn.commit()
            logger.info(f"Successfully upserted {success_count}/{len(df)} stations")
            return success_count > 0

    except Exception as e:
        logger.error(f"Error upserting stations: {e}")
        return False


def load_from_mysql(table_name, where_clause=None, limit=None):
    """
    Load data from MySQL table with improved error handling
    """
    if not test_connection():
        logger.error("Database connection failed")
        return pd.DataFrame()

    try:
        # Check if table exists
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            logger.error(f"Table {table_name} does not exist")
            return pd.DataFrame()

        query = f"SELECT * FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        logger.debug(f"Executing query: {query}")

        df = pd.read_sql(query, engine)

        # Parse datetime columns
        datetime_columns = ['timestamp', 'hourly_timestamp', 'start_time', 'end_time', 'created_at', 'last_updated']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        logger.info(f"Loaded {len(df)} records from {table_name}")
        return df

    except Exception as e:
        logger.error(f"Error loading from MySQL table {table_name}: {e}")
        return pd.DataFrame()


def get_latest_utilization():
    """
    Get the latest utilization status for each connector with fallback
    """
    try:
        # First try the view
        query = "SELECT * FROM latest_connector_status"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("latest_connector_status view is empty, trying direct query")
            # Fallback to direct query
            query = """
                SELECT 
                    u1.*
                FROM utilization_data u1
                INNER JOIN (
                    SELECT 
                        station_id, 
                        connector_id, 
                        MAX(timestamp) as max_timestamp
                    FROM utilization_data
                    WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                    GROUP BY station_id, connector_id
                ) u2 ON u1.station_id = u2.station_id 
                     AND u1.connector_id = u2.connector_id 
                     AND u1.timestamp = u2.max_timestamp
            """
            df = pd.read_sql(query, engine)

        logger.info(f"Retrieved {len(df)} latest utilization records")
        return df

    except Exception as e:
        logger.error(f"Error getting latest utilization: {e}")
        return pd.DataFrame()


def get_hourly_stats(hours=24):
    """
    Get hourly statistics with better error handling
    """
    try:
        query = """
            SELECT 
                hourly_timestamp,
                SUM(is_occupied) as total_occupied,
                SUM(is_available) as total_available,
                SUM(is_out_of_order) as total_out_of_order,
                SUM(total_connectors) as total_connectors,
                AVG(occupancy_rate) as avg_occupancy_rate
            FROM hourly_utilization
            WHERE hourly_timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            GROUP BY hourly_timestamp
            ORDER BY hourly_timestamp DESC
        """

        df = pd.read_sql(query, engine, params=[hours])

        if df.empty:
            logger.warning(f"No hourly stats found for last {hours} hours")
            # Try to get any available data
            fallback_query = """
                SELECT 
                    hourly_timestamp,
                    SUM(is_occupied) as total_occupied,
                    SUM(is_available) as total_available,
                    SUM(is_out_of_order) as total_out_of_order,
                    SUM(total_connectors) as total_connectors,
                    AVG(occupancy_rate) as avg_occupancy_rate
                FROM hourly_utilization
                GROUP BY hourly_timestamp
                ORDER BY hourly_timestamp DESC
                LIMIT 100
            """
            df = pd.read_sql(fallback_query, engine)

        logger.info(f"Retrieved {len(df)} hourly statistics records")
        return df

    except Exception as e:
        logger.error(f"Error getting hourly stats: {e}")
        return pd.DataFrame()


def get_recent_sessions(hours=24):
    """
    Get charging sessions with better error handling
    """
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

        df = pd.read_sql(query, engine, params=[hours])

        if df.empty:
            logger.warning(f"No sessions found for last {hours} hours")
            # Try to get any available data
            fallback_query = """
                SELECT 
                    cs.*,
                    st.name as station_name,
                    st.address as station_address
                FROM charging_sessions cs
                LEFT JOIN charging_stations st ON cs.station_id = st.id
                ORDER BY cs.end_time DESC
                LIMIT 100
            """
            df = pd.read_sql(fallback_query, engine)

        logger.info(f"Retrieved {len(df)} charging session records")
        return df

    except Exception as e:
        logger.error(f"Error getting recent sessions: {e}")
        return pd.DataFrame()


def cleanup_old_data(days_to_keep=7):
    """
    Clean up old utilization data with better error handling
    """
    try:
        with engine.connect() as conn:
            # Clean up old utilization data
            query = text("""
                DELETE FROM utilization_data 
                WHERE timestamp < DATE_SUB(NOW(), INTERVAL :days DAY)
            """)
            result = conn.execute(query, {'days': days_to_keep})
            util_deleted = result.rowcount

            # Clean up old hourly data
            query = text("""
                DELETE FROM hourly_utilization 
                WHERE hourly_timestamp < DATE_SUB(NOW(), INTERVAL :days DAY)
            """)
            result = conn.execute(query, {'days': days_to_keep})
            hourly_deleted = result.rowcount

            conn.commit()

            logger.info(f"Cleaned up {util_deleted} old utilization records and {hourly_deleted} old hourly records")

    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")


def validate_data_integrity():
    """
    Validate data integrity and fix common issues
    """
    issues_found = []
    fixes_applied = []

    try:
        with engine.connect() as conn:
            # Check for orphaned utilization data
            orphan_check = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM utilization_data u
                LEFT JOIN charging_stations s ON u.station_id = s.id
                WHERE s.id IS NULL
            """)).fetchone()[0]

            if orphan_check > 0:
                issues_found.append(f"Found {orphan_check} orphaned utilization records")

                # Clean up orphaned records
                result = conn.execute(text("""
                    DELETE u FROM utilization_data u
                    LEFT JOIN charging_stations s ON u.station_id = s.id
                    WHERE s.id IS NULL
                """))
                fixes_applied.append(f"Removed {result.rowcount} orphaned utilization records")

            # Check for stations without recent utilization data
            stale_stations = conn.execute(text("""
                SELECT s.id, s.name, MAX(u.timestamp) as last_update
                FROM charging_stations s
                LEFT JOIN utilization_data u ON s.id = u.station_id
                GROUP BY s.id, s.name
                HAVING last_update IS NULL OR last_update < DATE_SUB(NOW(), INTERVAL 6 HOUR)
            """)).fetchall()

            if stale_stations:
                issues_found.append(f"Found {len(stale_stations)} stations with stale data")

            # Recreate view if needed
            try:
                conn.execute(text("SELECT COUNT(*) FROM latest_connector_status"))
            except:
                create_latest_status_view(conn)
                fixes_applied.append("Recreated latest_connector_status view")

            conn.commit()

        logger.info(f"Data validation complete. Issues: {len(issues_found)}, Fixes: {len(fixes_applied)}")

        for issue in issues_found:
            logger.warning(f"Issue: {issue}")

        for fix in fixes_applied:
            logger.info(f"Fix applied: {fix}")

        return len(issues_found) == 0

    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        return False


# Enhanced backward compatibility functions
def save_to_csv(df, filename, output_dir="data", append=False):
    """Enhanced backward compatibility wrapper"""
    table_mapping = {
        'charging_stations.csv': 'charging_stations',
        'utilization_data.csv': 'utilization_data',
        'hourly_utilization.csv': 'hourly_utilization',
        'sessions.csv': 'charging_sessions'
    }

    table_name = table_mapping.get(filename)
    if table_name:
        # Validate data before saving
        validate_data_integrity()
        return save_to_mysql(df, table_name, mode='append' if append else 'replace')
    else:
        logger.warning(f"Unknown filename {filename}, skipping save")
        return False


def load_data(stations_path=None, utilization_path=None, hourly_path=None):
    """Enhanced backward compatibility wrapper"""
    # Validate data integrity first
    validate_data_integrity()

    stations_df = None
    utilization_df = None
    hourly_df = None

    if stations_path:
        stations_df = load_from_mysql('charging_stations')
        if stations_df.empty:
            logger.warning("No stations data found")

    if utilization_path:
        utilization_df = load_from_mysql('utilization_data',
                                         where_clause="timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
        if utilization_df.empty:
            logger.warning("No recent utilization data found")

    if hourly_path:
        hourly_df = load_from_mysql('hourly_utilization',
                                    where_clause="hourly_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
        if hourly_df.empty:
            logger.warning("No recent hourly data found")

    return stations_df, utilization_df, hourly_df


# Diagnostic functions
def diagnose_empty_results():
    """Diagnose why queries might return empty results"""
    logger.info("🔍 Diagnosing empty results...")

    try:
        with engine.connect() as conn:
            # Check table row counts
            tables = ['charging_stations', 'utilization_data', 'hourly_utilization', 'charging_sessions']

            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                    count = result.fetchone()[0]
                    logger.info(f"📊 {table}: {count} total rows")

                    # Check recent data
                    if table in ['utilization_data', 'hourly_utilization']:
                        timestamp_col = 'timestamp' if table == 'utilization_data' else 'hourly_timestamp'
                        recent_result = conn.execute(text(f"""
                            SELECT COUNT(*) as count FROM {table} 
                            WHERE {timestamp_col} >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                        """))
                        recent_count = recent_result.fetchone()[0]
                        logger.info(f"📅 {table}: {recent_count} rows in last 24h")

                except Exception as e:
                    logger.error(f"❌ Error checking {table}: {e}")

            # Check view
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM latest_connector_status"))
                view_count = result.fetchone()[0]
                logger.info(f"👁️  latest_connector_status view: {view_count} rows")
            except Exception as e:
                logger.error(f"❌ View error: {e}")
                logger.info("🔧 Attempting to recreate view...")
                create_latest_status_view(conn)

    except Exception as e:
        logger.error(f"❌ Diagnostic error: {e}")


if __name__ == "__main__":
    # Run diagnostics
    logging.basicConfig(level=logging.INFO)
    logger.info("Running MySQL diagnostics...")

    if test_connection():
        ensure_tables_exist()
        validate_data_integrity()
        diagnose_empty_results()
    else:
        logger.error("Cannot connect to database")
