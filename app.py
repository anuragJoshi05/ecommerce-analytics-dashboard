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
    """Load the processed data"""
    try:
        df = pd.read_csv('ecommerce_data.csv')
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the main analytics script first.")
        return None


@st.cache_data
def load_summary_stats():
    """Load summary statistics"""
    try:
        with open('summary_stats.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def create_kpi_metrics(df, filtered_df):
    """Create KPI metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = filtered_df['Total_Revenue'].sum()
        st.metric(
            label="Total Revenue",
            value=f"Rs. {total_revenue:,.0f}",
            delta=f"{(total_revenue / df['Total_Revenue'].sum() * 100):.1f}% of total"
        )

    with col2:
        avg_rating = filtered_df['Rating'].mean()
        overall_avg = df['Rating'].mean()
        st.metric(
            label="Average Rating",
            value=f"{avg_rating:.2f}/5",
            delta=f"{(avg_rating - overall_avg):.2f}"
        )

    with col3:
        repeat_rate = len(filtered_df[filtered_df['Customer_Type'] == 'Returning']) / len(filtered_df) * 100
        st.metric(
            label="Repeat Customer %",
            value=f"{repeat_rate:.1f}%"
        )

    with col4:
        return_rate = len(filtered_df[filtered_df['Order_Status'] == 'Returned']) / len(filtered_df) * 100
        st.metric(
            label="Return Rate",
            value=f"{return_rate:.1f}%"
        )


def create_revenue_chart(df):
    """Create revenue by category chart"""
    category_revenue = df.groupby('Product_Category')['Total_Revenue'].sum().sort_values(ascending=False)

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


def create_order_status_chart(df):
    """Create order status distribution chart"""
    status_counts = df['Order_Status'].value_counts()

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


def create_regional_performance_chart(df):
    """Create regional performance chart"""
    region_data = df.groupby('Region').agg({
        'Total_Revenue': 'sum',
        'Order_ID': 'count'
    }).reset_index()

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


def create_monthly_trend_chart(df):
    """Create monthly revenue trend chart"""
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')
    monthly_revenue = df.groupby('YearMonth')['Total_Revenue'].sum().reset_index()
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


def create_payment_mode_chart(df):
    """Create payment mode analysis chart"""
    payment_revenue = df.groupby('Payment_Mode')['Total_Revenue'].sum().sort_values(ascending=True)

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


def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = ['Price', 'Discount_Percent', 'Quantity', 'Total_Revenue',
                    'Shipping_Cost', 'Delivery_Time', 'Rating']

    # Filter only numeric columns that exist and have data
    available_cols = [col for col in numeric_cols if col in df.columns]
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


def create_customer_segment_analysis(df):
    """Create customer segment analysis"""
    customer_data = df.groupby('Customer_Type').agg({
        'Total_Revenue': 'sum',
        'Order_ID': 'count',
        'Rating': 'mean'
    }).reset_index()

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


def perform_sql_analysis(df):
    """Perform SQL analysis"""
    # Create in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    df.to_sql('ecommerce_data', conn, index=False, if_exists='replace')

    # SQL queries
    queries = {
        'Top 5 Categories by Revenue': """
            SELECT Product_Category, 
                   SUM(Total_Revenue) as Total_Revenue,
                   COUNT(*) as Order_Count
            FROM ecommerce_data 
            GROUP BY Product_Category
            ORDER BY Total_Revenue DESC 
            LIMIT 5
        """,

        'Region-wise Performance': """
            SELECT Region,
                   COUNT(*) as Total_Orders,
                   SUM(Total_Revenue) as Total_Revenue,
                   AVG(Rating) as Avg_Rating
            FROM ecommerce_data
            WHERE Order_Status = 'Delivered'
            GROUP BY Region
            ORDER BY Total_Revenue DESC
        """,

        'Payment Mode Analysis': """
            SELECT Payment_Mode,
                   COUNT(*) as Order_Count,
                   SUM(Total_Revenue) as Revenue,
                   AVG(Total_Revenue) as Avg_Order_Value
            FROM ecommerce_data
            GROUP BY Payment_Mode
            ORDER BY Revenue DESC
        """
    }

    results = {}
    for query_name, query in queries.items():
        result = pd.read_sql_query(query, conn)
        results[query_name] = result

    conn.close()
    return results


def main():
    """Main Streamlit app"""
    st.title("E-commerce Business Analytics Dashboard")
    st.markdown("---")

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Order_Date'].min(), df['Order_Date'].max()),
        min_value=df['Order_Date'].min(),
        max_value=df['Order_Date'].max()
    )

    # Category filter
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['Product_Category'].unique(),
        default=df['Product_Category'].unique()
    )

    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )

    # Order status filter
    order_status = st.sidebar.multiselect(
        "Select Order Status",
        options=df['Order_Status'].unique(),
        default=df['Order_Status'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['Order_Date'].dt.date >= date_range[0]) &
        (df['Order_Date'].dt.date <= date_range[1]) &
        (df['Product_Category'].isin(categories)) &
        (df['Region'].isin(regions)) &
        (df['Order_Status'].isin(order_status))
        ]

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
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = create_order_status_chart(filtered_df)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Regional Performance and Monthly Trend
    col3, col4 = st.columns(2)

    with col3:
        fig3 = create_regional_performance_chart(filtered_df)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = create_monthly_trend_chart(filtered_df)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Payment Mode and Customer Analysis
    col5, col6 = st.columns(2)

    with col5:
        fig5 = create_payment_mode_chart(filtered_df)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = create_customer_segment_analysis(filtered_df)
        st.plotly_chart(fig6, use_container_width=True)

    # Correlation Matrix
    st.header("Data Insights")
    fig7 = create_correlation_heatmap(filtered_df)
    st.plotly_chart(fig7, use_container_width=True)

    # SQL Analysis Section
    st.header("SQL Analysis Results")
    sql_results = perform_sql_analysis(filtered_df)

    for query_name, result in sql_results.items():
        st.subheader(query_name)
        st.dataframe(result, use_container_width=True)
        st.markdown("---")

    # Business Insights
    st.header("Key Business Insights")

    insights = []

    # Generate insights based on filtered data
    top_category = filtered_df.groupby('Product_Category')['Total_Revenue'].sum().idxmax()
    top_category_revenue = filtered_df.groupby('Product_Category')['Total_Revenue'].sum().max()
    insights.append(f"**Top Category**: {top_category} generates Rs. {top_category_revenue:,.0f} in revenue")

    avg_delivery = filtered_df[filtered_df['Order_Status'] == 'Delivered']['Delivery_Time'].mean()
    if not pd.isna(avg_delivery):
        insights.append(f"**Delivery Performance**: Average delivery time is {avg_delivery:.1f} days")

    return_rate = len(filtered_df[filtered_df['Order_Status'] == 'Returned']) / len(filtered_df) * 100
    insights.append(f"**Return Rate**: {return_rate:.1f}% of orders are returned")

    best_region = filtered_df.groupby('Region')['Total_Revenue'].sum().idxmax()
    best_region_revenue = filtered_df.groupby('Region')['Total_Revenue'].sum().max()
    insights.append(f"**Best Region**: {best_region} with Rs. {best_region_revenue:,.0f} revenue")

    popular_payment = filtered_df['Payment_Mode'].value_counts().index[0]
    insights.append(f"**Popular Payment**: {popular_payment} is the most used payment method")

    for insight in insights:
        st.markdown(f"• {insight}")

    # Footer
    st.markdown("---")
    st.markdown("**Technologies Used**: Python, Pandas, Streamlit, Plotly, SQLite")
    st.markdown("**Created by**: Business Analyst Portfolio Project")


if __name__ == "__main__":
    main()