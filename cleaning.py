import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
import io
from typing import Dict, Any, List, Optional
import re
from shapely.geometry import Point
import geopandas as gpd

# Page configuration
st.set_page_config(
    page_title="PostgreSQL Data Manager",
    page_icon="📊",
    layout="wide"
)

class DatabaseManager:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.engine = None
        self.connection = None
    
    def connect(self):
        """Create database connection"""
        try:
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['name']}"
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            return True
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        if self.connect():
            try:
                result = self.connection.execute(text("SELECT 1"))
                return True
            except Exception as e:
                st.error(f"Connection test failed: {str(e)}")
                return False
        return False
    
    def get_tables(self):
        """Get list of tables in the schema"""
        if not self.connection:
            return []
        
        try:
            schema = self.db_config.get('schema', 'public')
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = :schema
                ORDER BY table_name
            """)
            result = self.connection.execute(query, {'schema': schema})
            return [row[0] for row in result]
        except Exception as e:
            st.error(f"Failed to get tables: {str(e)}")
            return []
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, schema: str = None):
        """Create table from DataFrame with geometry support"""
        if not self.connection:
            return False
        
        try:
            schema = schema or self.db_config.get('schema', 'public')
            
            # Check if DataFrame has geometry column
            has_geometry = 'geometry' in df.columns
            
            if has_geometry:
                # Create GeoDataFrame for geometry support
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
                
                # Create table with PostGIS support
                gdf.to_postgis(
                    table_name,
                    self.engine,
                    schema=schema,
                    if_exists='replace',
                    index=False
                )
            else:
                # Regular DataFrame
                df.to_sql(
                    table_name, 
                    self.engine, 
                    schema=schema,
                    if_exists='replace', 
                    index=False
                )
            
            return True
        except Exception as e:
            st.error(f"Failed to create table: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()

def authenticate():
    """Handle authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        
        auth_config = st.secrets.get("auth", {})
        expected_username = auth_config.get("username", "")
        expected_password = auth_config.get("password", "")
        
        if not expected_username or not expected_password:
            st.error("Authentication credentials not configured in secrets")
            return False
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username == expected_username and password == expected_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        return False
    
    return True

def sanitize_table_name(name: str) -> str:
    """Sanitize table name for PostgreSQL"""
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w]', '_', name)
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = 'table_' + name
    return name.lower()

def get_postgres_type(pandas_dtype):
    """Convert pandas dtype to PostgreSQL type"""
    dtype_mapping = {
        'int64': 'BIGINT',
        'int32': 'INTEGER',
        'int16': 'SMALLINT',
        'int8': 'SMALLINT',
        'float64': 'DOUBLE PRECISION',
        'float32': 'REAL',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'object': 'TEXT',
        'category': 'TEXT',
        'geometry': 'GEOMETRY(POINT, 4326)'
    }
    return dtype_mapping.get(str(pandas_dtype), 'TEXT')

def detect_coordinate_columns(df):
    """Detect potential longitude and latitude columns"""
    columns = df.columns.str.lower()
    
    # Common longitude column names
    lon_patterns = ['lon', 'long', 'longitude', 'lng', 'x']
    lat_patterns = ['lat', 'latitude', 'y']
    
    lon_col = None
    lat_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in lon_patterns):
            lon_col = col
        if any(pattern in col_lower for pattern in lat_patterns):
            lat_col = col
    
    return lon_col, lat_col

def create_geometry_column(df, lon_col, lat_col):
    """Create geometry column from longitude and latitude"""
    try:
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Ensure lon/lat are numeric
        df_copy[lon_col] = pd.to_numeric(df_copy[lon_col], errors='coerce')
        df_copy[lat_col] = pd.to_numeric(df_copy[lat_col], errors='coerce')
        
        # Remove rows with invalid coordinates
        valid_coords = df_copy[lon_col].notna() & df_copy[lat_col].notna()
        df_copy = df_copy[valid_coords]
        
        # Create geometry column
        geometry = [Point(lon, lat) for lon, lat in zip(df_copy[lon_col], df_copy[lat_col])]
        df_copy['geometry'] = geometry
        
        return df_copy
    except Exception as e:
        st.error(f"Error creating geometry column: {str(e)}")
        return df

def main():
    if not authenticate():
        return
    
    st.title("📊 PostgreSQL Data Manager")
    st.markdown("Upload Excel files, clean data, and create PostgreSQL tables")
    
    # Initialize database manager
    db_config = st.secrets.get("database", {})
    
    if not all(key in db_config for key in ['user', 'password', 'host', 'port', 'name']):
        st.error("Database configuration incomplete in secrets")
        return
    
    db_manager = DatabaseManager(db_config)
    
    # Test database connection
    with st.spinner("Testing database connection..."):
        if not db_manager.test_connection():
            st.error("Cannot connect to database. Please check your configuration.")
            return
    
    st.success("✅ Database connection successful!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload & Clean Data", "Database Tables"])
    
    if page == "Upload & Clean Data":
        st.header("📤 Upload & Clean Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload your Excel file to get started"
        )
        
        if uploaded_file is not None:
            try:
                # Read Excel file
                with st.spinner("Reading Excel file..."):
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
                # Display raw data
                st.subheader("📋 Raw Data Preview")
                st.dataframe(df.head(10))
                
                # Data cleaning section
                st.subheader("🧹 Data Cleaning")
                
                # Detect coordinate columns
                lon_col, lat_col = detect_coordinate_columns(df)
                
                if lon_col and lat_col:
                    st.info(f"🌍 Detected coordinate columns: **{lon_col}** (longitude) and **{lat_col}** (latitude)")
                    create_geometry = st.checkbox(
                        "Create geometry column (EPSG:4326)",
                        value=True,
                        help="Creates a PostGIS geometry column from longitude and latitude"
                    )
                else:
                    create_geometry = False
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Column Selection**")
                    available_columns = df.columns.tolist()
                    selected_columns = st.multiselect(
                        "Select columns to keep",
                        options=available_columns,
                        default=available_columns,
                        help="Choose which columns to include in the final table"
                    )
                
                with col2:
                    st.markdown("**Data Type Configuration**")
                    dtype_config = {}
                    
                    for col in selected_columns:
                        current_dtype = str(df[col].dtype)
                        dtype_options = ['object', 'int64', 'float64', 'bool', 'datetime64[ns]']
                        
                        # Suggest float64 for coordinate columns
                        if col == lon_col or col == lat_col:
                            suggested_dtype = 'float64'
                        else:
                            suggested_dtype = current_dtype if current_dtype in dtype_options else 'object'
                        
                        dtype_config[col] = st.selectbox(
                            f"Data type for '{col}'",
                            options=dtype_options,
                            index=dtype_options.index(suggested_dtype),
                            key=f"dtype_{col}"
                        )
                
                # Apply cleaning
                if st.button("🔄 Apply Cleaning", type="primary"):
                    try:
                        # Filter columns
                        cleaned_df = df[selected_columns].copy()
                        
                        # Apply data type changes
                        for col, dtype in dtype_config.items():
                            if dtype == 'datetime64[ns]':
                                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                            elif dtype == 'bool':
                                cleaned_df[col] = cleaned_df[col].astype('bool')
                            elif dtype in ['int64', 'float64']:
                                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                            else:
                                cleaned_df[col] = cleaned_df[col].astype(dtype)
                        
                        # Create geometry column if requested
                        if create_geometry and lon_col and lat_col and lon_col in selected_columns and lat_col in selected_columns:
                            with st.spinner("Creating geometry column..."):
                                cleaned_df = create_geometry_column(cleaned_df, lon_col, lat_col)
                                st.success("✅ Geometry column created successfully!")
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.success("✅ Data cleaning applied successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during cleaning: {str(e)}")
                
                # Display cleaned data
                if 'cleaned_df' in st.session_state:
                    st.subheader("✨ Cleaned Data Preview")
                    cleaned_df = st.session_state.cleaned_df
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", len(cleaned_df))
                    with col2:
                        st.metric("Columns", len(cleaned_df.columns))
                    
                    st.dataframe(cleaned_df.head(10))
                    
                    # Data type summary
                    st.subheader("📊 Data Type Summary")
                    dtype_df = pd.DataFrame({
                        'Column': cleaned_df.columns,
                        'Data Type': [str(dtype) for dtype in cleaned_df.dtypes],
                        'PostgreSQL Type': [get_postgres_type(dtype) for dtype in cleaned_df.dtypes],
                        'Non-Null Count': [cleaned_df[col].count() for col in cleaned_df.columns]
                    })
                    st.dataframe(dtype_df)
                    
                    # Show geometry info if present
                    if 'geometry' in cleaned_df.columns:
                        st.info("🗺️ **Geometry Column Created**: Ready for PostGIS operations with EPSG:4326 coordinate system")
                        
                        # Show coordinate range
                        if lon_col and lat_col and lon_col in cleaned_df.columns and lat_col in cleaned_df.columns:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Longitude Range", f"{cleaned_df[lon_col].min():.6f} to {cleaned_df[lon_col].max():.6f}")
                            with col2:
                                st.metric("Latitude Range", f"{cleaned_df[lat_col].min():.6f} to {cleaned_df[lat_col].max():.6f}")
                    
                    # Table creation
                    st.subheader("🗄️ Create Database Table")
                    
                    # Check if PostGIS is required
                    needs_postgis = 'geometry' in cleaned_df.columns
                    if needs_postgis:
                        st.warning("⚠️ **PostGIS Required**: This table contains geometry data and requires PostGIS extension in your PostgreSQL database.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        table_name = st.text_input(
                            "Table Name",
                            value=sanitize_table_name(uploaded_file.name.split('.')[0]),
                            help="Enter the name for your new table"
                        )
                    
                    with col2:
                        schema_name = st.text_input(
                            "Schema",
                            value=db_config.get('schema', 'public'),
                            help="Database schema name"
                        )
                    
                    if st.button("🚀 Create Table", type="primary"):
                        if table_name:
                            sanitized_name = sanitize_table_name(table_name)
                            
                            with st.spinner(f"Creating table '{sanitized_name}'..."):
                                if db_manager.create_table_from_dataframe(
                                    cleaned_df, 
                                    sanitized_name, 
                                    schema_name
                                ):
                                    st.success(f"✅ Table '{sanitized_name}' created successfully!")
                                    if needs_postgis:
                                        st.info("🗺️ Geometry column created with EPSG:4326 coordinate system")
                                    st.balloons()
                                else:
                                    st.error("Failed to create table")
                        else:
                            st.warning("Please enter a table name")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif page == "Database Tables":
        st.header("🗄️ Database Tables")
        
        # Get existing tables
        tables = db_manager.get_tables()
        
        if tables:
            st.subheader("📋 Existing Tables")
            
            for i, table in enumerate(tables):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{table}**")
                with col2:
                    if st.button("View", key=f"view_{i}"):
                        try:
                            query = text(f"SELECT * FROM {table} LIMIT 100")
                            result_df = pd.read_sql(query, db_manager.connection)
                            st.dataframe(result_df)
                        except Exception as e:
                            st.error(f"Error viewing table: {str(e)}")
        else:
            st.info("No tables found in the database")
    
    # Close database connection
    db_manager.close()

if __name__ == "__main__":
    main()