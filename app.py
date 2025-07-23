import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sqlite3

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #262730;
        border: 1px solid #464649;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .metric-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
    .stMultiSelect > div > div {
        background-color: #262730;
        color: white;
    }
    .chart-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed data with error handling"""
    try:
        df = pd.read_csv('ecommerce_data.csv')
        # Convert date columns with error handling
        if 'Order_Date' in df.columns:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        if 'Delivery_Date' in df.columns:
            df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Order_Date'])
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'ecommerce_data.csv' not found in repository.")
        st.info("Please ensure the file exists in your GitHub repository root directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_data
def load_summary_stats():
    """Load summary statistics with error handling"""
    try:
        with open('summary_stats.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Summary stats file not found. Continuing without cached stats.")
        return {}
    except Exception as e:
        st.warning(f"⚠️ Error loading summary stats: {str(e)}")
        return {}

def create_kpi_metrics(df, filtered_df):
    """Create KPI metrics with error handling"""
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = filtered_df['Total_Revenue'].sum() if 'Total_Revenue' in filtered_df.columns else 0
        total_revenue_all = df['Total_Revenue'].sum() if 'Total_Revenue' in df.columns else 1
        percentage = (total_revenue / total_revenue_all * 100) if total_revenue_all > 0 else 0
        
        st.metric(
            label="Total Revenue",
            value=f"Rs. {total_revenue:,.0f}",
            delta=f"{percentage:.1f}% of total"
        )

    with col2:
        if 'Rating' in filtered_df.columns:
            avg_rating = filtered_df['Rating'].mean()
            overall_avg = df['Rating'].mean()
            delta_rating = avg_rating - overall_avg if not pd.isna(avg_rating) and not pd.isna(overall_avg) else 0
            
            st.metric(
                label="Average Rating",
                value=f"{avg_rating:.2f}/5" if not pd.isna(avg_rating) else "N/A",
                delta=f"{delta_rating:.2f}" if delta_rating != 0 else None
            )
        else:
            st.metric(label="Average Rating", value="N/A")

    with col3:
        if 'Customer_Type' in filtered_df.columns:
            repeat_rate = len(filtered_df[filtered_df['Customer_Type'] == 'Returning']) / len(filtered_df) * 100
            st.metric(
                label="Repeat Customer %",
                value=f"{repeat_rate:.1f}%"
            )
        else:
            st.metric(label="Repeat Customer %", value="N/A")

    with col4:
        if 'Order_Status' in filtered_df.columns:
            return_rate = len(filtered_df[filtered_df['Order_Status'] == 'Returned']) / len(filtered_df) * 100
            st.metric(
                label="Return Rate",
                value=f"{return_rate:.1f}%"
            )
        else:
            st.metric(label="Return Rate", value="N/A")

def create_revenue_chart(df):
    """Create revenue by category chart with error handling"""
    if df.empty or 'Product_Category' not in df.columns or 'Total_Revenue' not in df.columns:
        st.warning("Cannot create revenue chart: missing required columns.")
        return None
    
    try:
        category_revenue = df.groupby('Product_Category')['Total_Revenue'].sum().sort_values(ascending=False)
        
        if category_revenue.empty:
            st.warning("No revenue data available.")
            return None

        fig = px.bar(
            x=category_revenue.index,
            y=category_revenue.values,
            title="Revenue by Product Category",
            labels={'x': 'Category', 'y': 'Revenue (Rs)'},
            color=category_revenue.values,
            color_continuous_scale='viridis'
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Error creating revenue chart: {str(e)}")
        return None

def create_order_status_chart(df):
    """Create order status distribution chart with error handling"""
    if df.empty or 'Order_Status' not in df.columns:
        st.warning("Cannot create order status chart: missing Order_Status column.")
        return None
    
    try:
        status_counts = df['Order_Status'].value_counts()
        
        if status_counts.empty:
            st.warning("No order status data available.")
            return None

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Order Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16
        )

        return fig
    except Exception as e:
        st.error(f"Error creating order status chart: {str(e)}")
        return None

def create_regional_performance_chart(df):
    """Create regional performance chart with error handling"""
    required_cols = ['Region', 'Total_Revenue', 'Order_ID']
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning("Cannot create regional chart: missing required columns.")
        return None
    
    try:
        region_data = df.groupby('Region').agg({
            'Total_Revenue': 'sum',
            'Order_ID': 'count'
        }).reset_index()

        if region_data.empty:
            st.warning("No regional data available.")
            return None

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Revenue',
            x=region_data['Region'],
            y=region_data['Total_Revenue'],
            marker_color='lightblue',
            yaxis='y'
        ))

        fig.add_trace(go.Bar(
            name='Orders',
            x=region_data['Region'],
            y=region_data['Order_ID'],
            marker_color='orange',
            yaxis='y2',
            opacity=0.7
        ))

        fig.update_layout(
            title='Regional Performance: Revenue vs Orders',
            xaxis=dict(title='Region'),
            yaxis=dict(title='Revenue (Rs)', side='left'),
            yaxis2=dict(title='Number of Orders', side='right', overlaying='y'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            barmode='group'
        )

        return fig
    except Exception as e:
        st.error(f"Error creating regional chart: {str(e)}")
        return None

def create_monthly_trend_chart(df):
    """Create monthly revenue trend chart with error handling"""
    if df.empty or 'Order_Date' not in df.columns or 'Total_Revenue' not in df.columns:
        st.warning("Cannot create monthly trend: missing required columns.")
        return None
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        df_copy['YearMonth'] = df_copy['Order_Date'].dt.to_period('M')
        monthly_revenue = df_copy.groupby('YearMonth')['Total_Revenue'].sum().reset_index()
        
        if monthly_revenue.empty:
            st.warning("No monthly data available.")
            return None
            
        monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].astype(str)

        fig = px.line(
            monthly_revenue,
            x='YearMonth',
            y='Total_Revenue',
            title='Monthly Revenue Trend',
            markers=True
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            xaxis_title='Month',
            yaxis_title='Revenue (Rs)'
        )

        fig.update_traces(line_color='cyan', marker_color='cyan')

        return fig
    except Exception as e:
        st.error(f"Error creating monthly trend: {str(e)}")
        return None

def create_payment_mode_chart(df):
    """Create payment mode analysis chart with error handling"""
    required_cols = ['Payment_Mode', 'Total_Revenue']
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning("Cannot create payment mode chart: missing required columns.")
        return None
    
    try:
        payment_revenue = df.groupby('Payment_Mode')['Total_Revenue'].sum().sort_values(ascending=True)
        
        if payment_revenue.empty:
            st.warning("No payment mode data available.")
            return None

        fig = px.bar(
            x=payment_revenue.values,
            y=payment_revenue.index,
            orientation='h',
            title="Revenue by Payment Mode",
            labels={'x': 'Revenue (Rs)', 'y': 'Payment Mode'},
            color=payment_revenue.values,
            color_continuous_scale='plasma'
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Error creating payment mode chart: {str(e)}")
        return None

def create_correlation_heatmap(df):
    """Create correlation heatmap with error handling"""
    numeric_cols = ['Price', 'Discount_Percent', 'Quantity', 'Total_Revenue',
                    'Shipping_Cost', 'Delivery_Time', 'Rating']

    # Filter only numeric columns that exist and have data
    available_cols = [col for col in numeric_cols if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if len(available_cols) < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return None
    
    try:
        corr_data = df[available_cols].corr()

        fig = px.imshow(
            corr_data,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16
        )

        return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return None

def create_customer_segment_analysis(df):
    """Create customer segment analysis with error handling"""
    required_cols = ['Customer_Type', 'Total_Revenue', 'Order_ID', 'Rating']
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning("Cannot create customer segment analysis: missing required columns.")
        return None
    
    try:
        customer_data = df.groupby('Customer_Type').agg({
            'Total_Revenue': 'sum',
            'Order_ID': 'count',
            'Rating': 'mean'
        }).reset_index()

        if customer_data.empty:
            st.warning("No customer segment data available.")
            return None

        fig = px.bar(
            customer_data,
            x='Customer_Type',
            y='Total_Revenue',
            title='Revenue by Customer Type',
            color='Customer_Type',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Error creating customer segment analysis: {str(e)}")
        return None

def perform_sql_analysis(df):
    """Perform SQL analysis with error handling"""
    if df.empty:
        st.warning("No data available for SQL analysis.")
        return {}
    
    try:
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql('ecommerce_data', conn, index=False, if_exists='replace')

        # SQL queries with error handling
        queries = {}
        
        # Only add queries for available columns
        if all(col in df.columns for col in ['Product_Category', 'Total_Revenue']):
            queries['Top 5 Categories by Revenue'] = """
                SELECT Product_Category, 
                       SUM(Total_Revenue) as Total_Revenue,
                       COUNT(*) as Order_Count
                FROM ecommerce_data 
                GROUP BY Product_Category
                ORDER BY Total_Revenue DESC 
                LIMIT 5
            """

        if all(col in df.columns for col in ['Region', 'Total_Revenue', 'Rating', 'Order_Status']):
            queries['Region-wise Performance'] = """
                SELECT Region,
                       COUNT(*) as Total_Orders,
                       SUM(Total_Revenue) as Total_Revenue,
                       AVG(Rating) as Avg_Rating
                FROM ecommerce_data
                WHERE Order_Status = 'Delivered'
                GROUP BY Region
                ORDER BY Total_Revenue DESC
            """

        if all(col in df.columns for col in ['Payment_Mode', 'Total_Revenue']):
            queries['Payment Mode Analysis'] = """
                SELECT Payment_Mode,
                       COUNT(*) as Order_Count,
                       SUM(Total_Revenue) as Revenue,
                       AVG(Total_Revenue) as Avg_Order_Value
                FROM ecommerce_data
                GROUP BY Payment_Mode
                ORDER BY Revenue DESC
            """

        results = {}
        for query_name, query in queries.items():
            try:
                result = pd.read_sql_query(query, conn)
                results[query_name] = result
            except Exception as e:
                st.warning(f"Error executing query '{query_name}': {str(e)}")

        conn.close()
        return results
    except Exception as e:
        st.error(f"Error in SQL analysis: {str(e)}")
        return {}

def main():
    """Main Streamlit app"""
    st.title("E-commerce Business Analytics Dashboard")
    st.markdown("---")

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Load summary stats (optional)
    summary_stats = load_summary_stats()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    if 'Order_Date' in df.columns:
        min_date = df['Order_Date'].min().date()
        max_date = df['Order_Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        st.sidebar.warning("Order_Date column not found.")
        date_range = None

    # Category filter
    if 'Product_Category' in df.columns:
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['Product_Category'].unique(),
            default=df['Product_Category'].unique()
        )
    else:
        categories = []

    # Region filter
    if 'Region' in df.columns:
        regions = st.sidebar.multiselect(
            "Select Regions",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
    else:
        regions = []

    # Order status filter
    if 'Order_Status' in df.columns:
        order_status = st.sidebar.multiselect(
            "Select Order Status",
            options=df['Order_Status'].unique(),
            default=df['Order_Status'].unique()
        )
    else:
        order_status = []

    # Apply filters
    filtered_df = df.copy()
    
    if date_range and 'Order_Date' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['Order_Date'].dt.date >= date_range[0]) &
            (filtered_df['Order_Date'].dt.date <= date_range[1])
        ]
    
    if categories and 'Product_Category' in df.columns:
        filtered_df = filtered_df[filtered_df['Product_Category'].isin(categories)]
    
    if regions and 'Region' in df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if order_status and 'Order_Status' in df.columns:
        filtered_df = filtered_df[filtered_df['Order_Status'].isin(order_status)]

    # Main dashboard
    st.header("Key Performance Indicators")
    create_kpi_metrics(df, filtered_df)

    st.markdown("---")

    # Charts section
    st.header("Business Analytics")

    # Row 1: Revenue and Order Status
    col1, col2 = st.columns(2)

    with col1:
        fig1 = create_revenue_chart(filtered_df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = create_order_status_chart(filtered_df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Regional Performance and Monthly Trend
    col3, col4 = st.columns(2)

    with col3:
        fig3 = create_regional_performance_chart(filtered_df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = create_monthly_trend_chart(filtered_df)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Payment Mode and Customer Analysis
    col5, col6 = st.columns(2)

    with col5:
        fig5 = create_payment_mode_chart(filtered_df)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = create_customer_segment_analysis(filtered_df)
        if fig6:
            st.plotly_chart(fig6, use_container_width=True)

    # Correlation Matrix
    st.header("Data Insights")
    fig7 = create_correlation_heatmap(filtered_df)
    if fig7:
        st.plotly_chart(fig7, use_container_width=True)

    # SQL Analysis Section
    st.header("SQL Analysis Results")
    sql_results = perform_sql_analysis(filtered_df)

    if sql_results:
        for query_name, result in sql_results.items():
            st.subheader(query_name)
            st.dataframe(result, use_container_width=True)
            st.markdown("---")
    else:
        st.warning("No SQL analysis results available.")

    # Business Insights
    st.header("Key Business Insights")

    insights = []

    try:
        # Generate insights based on filtered data
        if not filtered_df.empty and 'Product_Category' in filtered_df.columns and 'Total_Revenue' in filtered_df.columns:
            top_category = filtered_df.groupby('Product_Category')['Total_Revenue'].sum().idxmax()
            top_category_revenue = filtered_df.groupby('Product_Category')['Total_Revenue'].sum().max()
            insights.append(f"**Top Category**: {top_category} generates Rs. {top_category_revenue:,.0f} in revenue")

        if not filtered_df.empty and 'Order_Status' in filtered_df.columns and 'Delivery_Time' in filtered_df.columns:
            delivered_orders = filtered_df[filtered_df['Order_Status'] == 'Delivered']
            if not delivered_orders.empty:
                avg_delivery = delivered_orders['Delivery_Time'].mean()
                if not pd.isna(avg_delivery):
                    insights.append(f"**Delivery Performance**: Average delivery time is {avg_delivery:.1f} days")

        if not filtered_df.empty and 'Order_Status' in filtered_df.columns:
            return_rate = len(filtered_df[filtered_df['Order_Status'] == 'Returned']) / len(filtered_df) * 100
            insights.append(f"**Return Rate**: {return_rate:.1f}% of orders are returned")

        if not filtered_df.empty and 'Region' in filtered_df.columns and 'Total_Revenue' in filtered_df.columns:
            best_region = filtered_df.groupby('Region')['Total_Revenue'].sum().idxmax()
            best_region_revenue = filtered_df.groupby('Region')['Total_Revenue'].sum().max()
            insights.append(f"**Best Region**: {best_region} with Rs. {best_region_revenue:,.0f} revenue")

        if not filtered_df.empty and 'Payment_Mode' in filtered_df.columns:
            popular_payment = filtered_df['Payment_Mode'].value_counts().index[0]
            insights.append(f"**Popular Payment**: {popular_payment} is the most used payment method")

        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("No insights available for the current data selection.")
            
    except Exception as e:
        st.warning(f"Error generating insights: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("**Technologies Used**: Python, Pandas, Streamlit, Plotly, SQLite")
    st.markdown("**Created by**: Business Analyst Portfolio Project")

if __name__ == "__main__":
    main()
