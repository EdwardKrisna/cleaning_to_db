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

def init_session_state():
    """Initialize all session state variables"""
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Database connection
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    
    # File handling
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    
    # Data cleaning
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'cleaning_applied' not in st.session_state:
        st.session_state.cleaning_applied = False

def setup_database():
    """Setup database connection (only once)"""
    if st.session_state.db_connected:
        return st.session_state.db_manager
    
    db_config = st.secrets.get("database", {})
    
    if not all(key in db_config for key in ['user', 'password', 'host', 'port', 'name']):
        st.error("Database configuration incomplete in secrets")
        return None
    
    with st.spinner("Testing database connection..."):
        db_manager = DatabaseManager(db_config)
        if db_manager.test_connection():
            st.session_state.db_manager = db_manager
            st.session_state.db_connected = True
            st.success("✅ Database connection successful!")
            return db_manager
        else:
            st.error("Cannot connect to database. Please check your configuration.")
            return None

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

def create_geometry_column(df, lon_col, lat_col):
    """Create geometry column from longitude and latitude"""
    try:
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Ensure lon/lat are numeric (keep NaN values, don't coerce to NaN)
        df_copy[lon_col] = pd.to_numeric(df_copy[lon_col], errors='coerce')
        df_copy[lat_col] = pd.to_numeric(df_copy[lat_col], errors='coerce')
        
        # Create geometry column with None for invalid coordinates
        geometry = []
        for lon, lat in zip(df_copy[lon_col], df_copy[lat_col]):
            if pd.notna(lon) and pd.notna(lat):
                geometry.append(Point(lon, lat))
            else:
                geometry.append(None)  # Keep blank/None for invalid coordinates
        
        df_copy['geometry'] = geometry
        
        # Count valid geometries
        valid_geometries = sum(1 for geom in geometry if geom is not None)
        total_rows = len(df_copy)
        
        st.info(f"🗺️ Geometry created: {valid_geometries} valid points out of {total_rows} total rows")
        
        return df_copy
    except Exception as e:
        st.error(f"Error creating geometry column: {str(e)}")
        return df

def process_uploaded_file(uploaded_file):
    """Process uploaded file (only when file changes)"""
    if uploaded_file is None:
        st.session_state.current_file = None
        st.session_state.df = None
        st.session_state.file_processed = False
        st.session_state.cleaned_df = None
        st.session_state.cleaning_applied = False
        return None
    
    # Check if this is a new file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.current_file != file_id:
        # New file - process it
        try:
            with st.spinner("Reading Excel file..."):
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.current_file = file_id
            st.session_state.file_processed = True
            st.session_state.cleaned_df = None
            st.session_state.cleaning_applied = False
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    return st.session_state.df

def main():
    # Initialize session state
    init_session_state()
    
    # Authentication
    if not authenticate():
        return
    
    st.title("📊 PostgreSQL Data Manager")
    st.markdown("Upload Excel files, clean data, and create PostgreSQL tables")
    
    # Setup database (only once)
    db_manager = setup_database()
    if not db_manager:
        return
    
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
        
        # Process file (cached)
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            # Display raw data and analysis (always shown for uploaded file)
            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head(10))
            
            # Display null/nan statistics
            st.subheader("🔍 Data Quality Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            
            with col2:
                # Calculate null statistics
                null_stats = df.isnull().sum()
                total_nulls = null_stats.sum()
                st.metric("Total Null Values", int(total_nulls))
                
                if total_nulls > 0:
                    null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
                    st.metric("Null Percentage", f"{null_percentage:.2f}%")
            
            # Show null values per column
            if total_nulls > 0:
                st.markdown("**Null Values by Column:**")
                null_df = pd.DataFrame({
                    'Column': df.columns,
                    'Null Count': [df[col].isnull().sum() for col in df.columns],
                    'Null %': [f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%" for col in df.columns]
                })
                # Only show columns with null values
                null_df = null_df[null_df['Null Count'] > 0]
                if not null_df.empty:
                    st.dataframe(null_df, hide_index=True)
                else:
                    st.success("✅ No null values found in the dataset!")
            else:
                st.success("✅ No null values found in the dataset!")
            
            # Data cleaning section
            st.subheader("🧹 Data Cleaning")
            
            # User selects coordinate columns for geometry creation
            st.markdown("**🌍 Geometry Column Creation (Optional)**")
            create_geometry = st.checkbox(
                "Create geometry column (EPSG:4326)",
                value=False,
                help="Creates a PostGIS geometry column from longitude and latitude"
            )
            
            lon_col = None
            lat_col = None
            
            if create_geometry:
                col1, col2 = st.columns(2)
                with col1:
                    lon_col = st.selectbox(
                        "Select Longitude Column",
                        options=[None] + df.columns.tolist(),
                        help="Choose the column containing longitude values"
                    )
                with col2:
                    lat_col = st.selectbox(
                        "Select Latitude Column", 
                        options=[None] + df.columns.tolist(),
                        help="Choose the column containing latitude values"
                    )
                
                if lon_col and lat_col:
                    if lon_col == lat_col:
                        st.error("⚠️ Longitude and latitude columns must be different!")
                    else:
                        st.success(f"✅ Will create geometry from **{lon_col}** (longitude) and **{lat_col}** (latitude)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Column Selection**")
                available_columns = df.columns.tolist()
                selected_columns = st.multiselect(
                    "Select columns to keep",
                    options=available_columns,
                    default=[],  # Start with empty selection
                    help="Choose which columns to include in the final table"
                )
            
            with col2:
                st.markdown("**Data Type Configuration**")
                
                if selected_columns:
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
                else:
                    st.info("👆 Please select columns first")
                    dtype_config = {}
            
            # Apply cleaning button - only show if columns are selected
            if selected_columns:
                if st.button("🔄 Apply Cleaning", type="primary"):
                    try:
                        # Validate geometry inputs
                        if create_geometry:
                            if not lon_col or not lat_col:
                                st.error("⚠️ Please select both longitude and latitude columns for geometry creation")
                                st.stop()
                            if lon_col == lat_col:
                                st.error("⚠️ Longitude and latitude columns must be different!")
                                st.stop()
                            if lon_col not in selected_columns or lat_col not in selected_columns:
                                st.error("⚠️ Both coordinate columns must be selected in 'Columns to keep'")
                                st.stop()
                        
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
                        if create_geometry and lon_col and lat_col:
                            with st.spinner("Creating geometry column..."):
                                cleaned_df = create_geometry_column(cleaned_df, lon_col, lat_col)
                                st.success("✅ Geometry column created successfully!")
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_applied = True
                        st.success("✅ Data cleaning applied successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during cleaning: {str(e)}")
            else:
                st.info("👆 Please select at least one column to proceed with cleaning")
            
            # Display cleaned data (only if cleaning was applied)
            if st.session_state.cleaning_applied and st.session_state.cleaned_df is not None:
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
                        value=st.secrets.get("database", {}).get('schema', 'public'),
                        help="Database schema name"
                    )
                
                if st.button("🚀 Create Table", type="primary"):
                    if table_name:
                        sanitized_name = sanitize_table_name(table_name)
                        
                        with st.spinner(f"Creating table '{sanitized_name}'..."):
                            if st.session_state.db_manager.create_table_from_dataframe(
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
    
    elif page == "Database Tables":
        st.header("🗄️ Database Tables")
        
        # Get existing tables
        tables = st.session_state.db_manager.get_tables()
        
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
                            result_df = pd.read_sql(query, st.session_state.db_manager.connection)
                            st.dataframe(result_df)
                        except Exception as e:
                            st.error(f"Error viewing table: {str(e)}")
        else:
            st.info("No tables found in the database")

if __name__ == "__main__":
    main()