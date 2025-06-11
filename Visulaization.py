import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Customer Support Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    .stSelectbox label, .stRadio label {
        font-weight: 600;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Data generation with time series
@st.cache_data
def generate_data():
    length = 1000
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range(start='2024-01-01', end='2025-06-11', freq='H')
    length = len(dates)
    
    metrics = {
        'datetime': dates,
        'queries_handled': np.random.poisson(150, size=length) + np.sin(np.arange(length) * 2 * np.pi / 24) * 20,
        'response_time': np.random.lognormal(1.5, 0.5, size=length),
        'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], size=length, p=[0.5, 0.2, 0.3]),
        'churn_risk': np.random.beta(2, 5, size=length),
        'intent': np.random.choice(['Billing', 'Support', 'Sales', 'Technical', 'Account'], size=length, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'effort_score': np.random.normal(3.2, 0.8, size=length),
        'satisfaction': np.random.normal(4.1, 0.9, size=length),
        'resolution_time': np.random.exponential(2, size=length),
        'agent_rating': np.random.normal(4.3, 0.6, size=length)
    }
    
    df = pd.DataFrame(metrics)
    df['effort_score'] = np.clip(df['effort_score'], 1, 5)
    df['satisfaction'] = np.clip(df['satisfaction'], 1, 5)
    df['agent_rating'] = np.clip(df['agent_rating'], 1, 5)
    
    return df

# Load data
df = generate_data()

# Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; color: white; border-radius: 0 0 20px 20px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🎯 Customer Support Analytics Dashboard</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Real-time insights into customer support performance</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(df['datetime'].dt.date.min(), df['datetime'].dt.date.max()),
        min_value=df['datetime'].dt.date.min(),
        max_value=df['datetime'].dt.date.max()
    )
    
    # Intent filter
    intent_options = ['All'] + list(df['intent'].unique())
    selected_intent = st.selectbox("🎯 Filter by Intent", intent_options)
    
    # Sentiment filter
    sentiment_options = ['All'] + list(df['sentiment'].unique())
    selected_sentiment = st.selectbox("😊 Filter by Sentiment", sentiment_options)
    
    # Metric selector
    st.markdown("### 📊 Visualization Options")
    selected_metrics = st.multiselect(
        "Choose Metrics to Display",
        ['Queries Handled', 'Response Time', 'Sentiment Analysis', 'Churn Risk', 
         'Intent Analysis', 'Effort Score', 'Satisfaction Trends', 'Agent Performance'],
        default=['Queries Handled', 'Sentiment Analysis', 'Churn Risk']
    )

# Filter data
df_filtered = df.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['datetime'].dt.date >= date_range[0]) & 
        (df_filtered['datetime'].dt.date <= date_range[1])
    ]

if selected_intent != 'All':
    df_filtered = df_filtered[df_filtered['intent'] == selected_intent]

if selected_sentiment != 'All':
    df_filtered = df_filtered[df_filtered['sentiment'] == selected_sentiment]

# Key Metrics Row
st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_queries = int(df_filtered['queries_handled'].sum())
    st.metric("Total Queries", f"{total_queries:,}", delta=f"{int(df_filtered['queries_handled'].tail(7).mean() - df_filtered['queries_handled'].head(7).mean())}")

with col2:
    avg_response_time = df_filtered['response_time'].mean()
    st.metric("Avg Response Time", f"{avg_response_time:.1f}s", delta=f"{avg_response_time - df_filtered['response_time'].median():.1f}s")

with col3:
    satisfaction_score = df_filtered['satisfaction'].mean()
    st.metric("Satisfaction Score", f"{satisfaction_score:.2f}/5", delta=f"{satisfaction_score - 4.0:.2f}")

with col4:
    avg_churn_risk = df_filtered['churn_risk'].mean()
    churn_color = "inverse" if avg_churn_risk < 0.3 else "normal"
    st.metric("Churn Risk", f"{avg_churn_risk:.1%}", delta=f"{(avg_churn_risk - 0.25):.1%}", delta_color=churn_color)

with col5:
    resolution_time = df_filtered['resolution_time'].mean()
    st.metric("Avg Resolution", f"{resolution_time:.1f}h", delta=f"{resolution_time - 2.0:.1f}h")

st.markdown("---")

# Dynamic visualizations based on selection
if 'Queries Handled' in selected_metrics:
    st.markdown("### 📞 Queries Handled Over Time")
    
    # Resample data for time series
    hourly_queries = df_filtered.set_index('datetime')['queries_handled'].resample('D').sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_queries['datetime'],
        y=hourly_queries['queries_handled'],
        mode='lines+markers',
        name='Daily Queries',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(hourly_queries)), hourly_queries['queries_handled'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=hourly_queries['datetime'],
        y=p(range(len(hourly_queries))),
        mode='lines',
        name='Trend',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Query Volume Trends with Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Queries",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

if 'Sentiment Analysis' in selected_metrics:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😊 Sentiment Distribution")
        sentiment_counts = df_filtered['sentiment'].value_counts()
        
        colors = {'Positive': '#4CAF50', 'Neutral': '#FF9800', 'Negative': '#F44336'}
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=[colors[label] for label in sentiment_counts.index],
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Customer Sentiment Breakdown",
            showlegend=True,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Sentiment Over Time")
        sentiment_time = df_filtered.groupby([df_filtered['datetime'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        for sentiment in sentiment_time.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_time.index,
                y=sentiment_time[sentiment],
                mode='lines+markers',
                name=sentiment,
                stackgroup='one',
                line=dict(color=colors[sentiment])
            ))
        
        fig.update_layout(
            title="Sentiment Trends",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

if 'Churn Risk' in selected_metrics:
    st.markdown("### ⚠️ Churn Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn risk by intent
        churn_by_intent = df_filtered.groupby('intent')['churn_risk'].mean().sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=churn_by_intent.values,
            y=churn_by_intent.index,
            orientation='h',
            marker_color='rgba(255, 107, 107, 0.7)',
            text=[f'{val:.1%}' for val in churn_by_intent.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Churn Risk by Intent Category",
            xaxis_title="Average Churn Risk",
            yaxis_title="Intent Category",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn risk distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_filtered['churn_risk'],
            nbinsx=30,
            marker_color='rgba(255, 107, 107, 0.7)',
            name='Churn Risk Distribution'
        ))
        
        fig.add_vline(x=df_filtered['churn_risk'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Avg: {df_filtered['churn_risk'].mean():.1%}")
        
        fig.update_layout(
            title="Churn Risk Distribution",
            xaxis_title="Churn Risk Score",
            yaxis_title="Frequency",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

if 'Intent Analysis' in selected_metrics:
    st.markdown("### 🎯 Customer Intent Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        intent_counts = df_filtered['intent'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=intent_counts.index,
            values=intent_counts.values,
            hole=0.3,
            textinfo='label+percent+value',
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title="Intent Distribution",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Intent vs satisfaction
        intent_satisfaction = df_filtered.groupby('intent')['satisfaction'].mean().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=intent_satisfaction.index,
            y=intent_satisfaction.values,
            marker_color='rgba(78, 205, 196, 0.7)',
            text=[f'{val:.2f}' for val in intent_satisfaction.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Average Satisfaction by Intent",
            xaxis_title="Intent Category",
            yaxis_title="Satisfaction Score",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

if 'Response Time' in selected_metrics:
    st.markdown("### ⏱️ Response Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df_filtered['response_time'],
            name='Response Time',
            marker_color='rgba(102, 126, 234, 0.7)'
        ))
        
        fig.update_layout(
            title="Response Time Distribution",
            yaxis_title="Response Time (seconds)",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Response time by intent
        fig = go.Figure()
        for intent in df_filtered['intent'].unique():
            intent_data = df_filtered[df_filtered['intent'] == intent]['response_time']
            fig.add_trace(go.Violin(
                y=intent_data,
                name=intent,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="Response Time by Intent",
            yaxis_title="Response Time (seconds)",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

if 'Agent Performance' in selected_metrics:
    st.markdown("### 👥 Agent Performance Dashboard")
    
    # Generate agent data
    agents = [f"Agent_{i:02d}" for i in range(1, 21)]
    agent_data = pd.DataFrame({
        'agent': np.random.choice(agents, size=len(df_filtered)),
        'rating': df_filtered['agent_rating'],
        'resolution_time': df_filtered['resolution_time'],
        'satisfaction': df_filtered['satisfaction']
    })
    
    agent_summary = agent_data.groupby('agent').agg({
        'rating': 'mean',
        'resolution_time': 'mean',
        'satisfaction': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_agents = agent_summary.nlargest(10, 'rating')
        
        fig = go.Figure(go.Bar(
            x=top_agents.index,
            y=top_agents['rating'],
            marker_color='rgba(78, 205, 196, 0.7)',
            text=top_agents['rating'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 10 Agents by Rating",
            xaxis_title="Agent",
            yaxis_title="Average Rating",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Agent performance scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agent_summary['resolution_time'],
            y=agent_summary['rating'],
            mode='markers',
            marker=dict(
                size=agent_summary['satisfaction']*10,
                color=agent_summary['satisfaction'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Satisfaction")
            ),
            text=agent_summary.index,
            hovertemplate='<b>%{text}</b><br>Resolution Time: %{x:.1f}h<br>Rating: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Agent Performance Matrix",
            xaxis_title="Average Resolution Time (hours)",
            yaxis_title="Average Rating",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("### 📋 Detailed Data View")
if st.checkbox("Show Filtered Data"):
    st.dataframe(
        df_filtered[['datetime', 'queries_handled', 'response_time', 'sentiment', 
                    'churn_risk', 'intent', 'satisfaction', 'effort_score']],
        use_container_width=True
    )

# Export functionality
st.markdown("### 💾 Export Data")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate Report Summary"):
        st.markdown(f"""
        **📈 Dashboard Summary Report**
        
        - **Total Queries Processed**: {total_queries:,}
        - **Average Response Time**: {avg_response_time:.1f} seconds
        - **Customer Satisfaction**: {satisfaction_score:.2f}/5.0
        - **Churn Risk Level**: {avg_churn_risk:.1%}
        - **Most Common Intent**: {df_filtered['intent'].mode().iloc[0]}
        - **Positive Sentiment**: {(df_filtered['sentiment'] == 'Positive').mean():.1%}
        """)

with col2:
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"support_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )