"""
MySQL Load module for ETL pipeline
Handles saving and loading data from MySQL database
"""
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from mysql_config import engine, get_connection

logger = logging.getLogger(__name__)

def save_to_mysql(df, table_name, mode='append'):
    """
    Save DataFrame to MySQL table
    
    Args:
        df: DataFrame to save
        table_name: Target table name
        mode: 'append' or 'replace'
    """
    if df is None or df.empty:
        logger.warning(f"No data to save to {table_name}")
        return False
    
    try:
        # Special handling for charging_stations (upsert behavior)
        if table_name == 'charging_stations':
            return upsert_stations(df)
        
        # For other tables, use standard insert
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=mode,
            index=False,
            method='multi',
            chunksize=1000
        )
        
        logger.info(f"Saved {len(df)} records to {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to MySQL table {table_name}: {e}")
        return False

def upsert_stations(df):
    """
    Upsert stations data (update if exists, insert if new)
    """
    try:
        with engine.connect() as conn:
            for _, row in df.iterrows():
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
                
                conn.execute(query, {
                    'id': row['id'],
                    'name': row['name'],
                    'operator': row.get('operator', 'Eviny'),
                    'status': row['status'],
                    'address': row.get('address'),
                    'description': row.get('description'),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'total_connectors': row.get('total_connectors', 0),
                    'ccs_connectors': row.get('ccs_connectors', 0),
                    'chademo_connectors': row.get('chademo_connectors', 0),
                    'type2_connectors': row.get('type2_connectors', 0),
                    'other_connectors': row.get('other_connectors', 0),
                    'amenities': row.get('amenities', '')
                })
            
            conn.commit()
            logger.info(f"Upserted {len(df)} stations")
            return True
            
    except Exception as e:
        logger.error(f"Error upserting stations: {e}")
        return False

def load_from_mysql(table_name, where_clause=None, limit=None):
    """
    Load data from MySQL table
    
    Args:
        table_name: Table to load from
        where_clause: Optional WHERE clause
        limit: Optional row limit
    
    Returns:
        DataFrame with the data
    """
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
        
        logger.info(f"Loaded {len(df)} records from {table_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading from MySQL table {table_name}: {e}")
        return pd.DataFrame()

def cleanup_old_data(days_to_keep=7):
    """
    Clean up old utilization data to prevent database growth
    """
    try:
        with engine.connect() as conn:
            # Clean up old utilization data
            query = text("""
                DELETE FROM utilization_data 
                WHERE timestamp < DATE_SUB(NOW(), INTERVAL :days DAY)
            """)
            result = conn.execute(query, {'days': days_to_keep})
            conn.commit()
            
            logger.info(f"Cleaned up {result.rowcount} old utilization records")
            
            # Clean up old hourly data
            query = text("""
                DELETE FROM hourly_utilization 
                WHERE hourly_timestamp < DATE_SUB(NOW(), INTERVAL :days DAY)
            """)
            result = conn.execute(query, {'days': days_to_keep})
            conn.commit()
            
            logger.info(f"Cleaned up {result.rowcount} old hourly records")
            
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")

def get_latest_utilization():
    """
    Get the latest utilization status for each connector
    """
    query = """
        SELECT * FROM latest_connector_status
    """
    return pd.read_sql(query, engine)

def get_hourly_stats(hours=24):
    """
    Get hourly statistics for the last N hours
    """
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
    return pd.read_sql(query, engine, params=(hours,))

def get_recent_sessions(hours=24):
    """
    Get charging sessions from the last N hours
    """
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
    return pd.read_sql(query, engine, params=(hours,))

# Backward compatibility functions for CSV-based code
def save_to_csv(df, filename, output_dir="data", append=False):
    """Backward compatibility wrapper - saves to MySQL instead"""
    table_mapping = {
        'charging_stations.csv': 'charging_stations',
        'utilization_data.csv': 'utilization_data',
        'hourly_utilization.csv': 'hourly_utilization',
        'sessions.csv': 'charging_sessions'
    }
    
    table_name = table_mapping.get(filename)
    if table_name:
        return save_to_mysql(df, table_name, mode='append' if append else 'replace')
    else:
        logger.warning(f"Unknown filename {filename}, skipping save")
        return False

def load_data(stations_path=None, utilization_path=None, hourly_path=None):
    """Backward compatibility wrapper - loads from MySQL instead"""
    stations_df = load_from_mysql('charging_stations') if stations_path else None
    utilization_df = load_from_mysql('utilization_data', 
                                   where_clause="timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)") if utilization_path else None
    hourly_df = load_from_mysql('hourly_utilization',
                               where_clause="hourly_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)") if hourly_path else None
    
    return stations_df, utilization_df, hourly_df