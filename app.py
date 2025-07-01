# AI-Powered Customer Snapshot Dashboard POC
# Save this as app.py and run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

# Page configuration
st.set_page_config(
    page_title="AI Customer Snapshot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - Optimized for dark theme
st.markdown("""
<style>
    /* Force metric containers to have good contrast */
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    
    /* Light text for dark theme */
    .stApp [data-testid="stHeader"] {
        background-color: transparent;
    }
    
    .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
    }
    
    .stApp .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Tab text visibility */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Health score styles */
    .health-score-good { background-color: #00C851; }
    .health-score-warning { background-color: #ffbb33; }
    .health-score-critical { background-color: #ff4444; }
    
    /* Recommendation box with light text */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(243, 229, 245, 0.1) 0%, rgba(225, 190, 231, 0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(206, 147, 216, 0.3);
        margin: 10px 0;
        color: #ffffff !important;
    }
    
    .recommendation-box h3, .recommendation-box h4 {
        color: #e1bee7 !important;
    }
    
    .recommendation-box li {
        color: #f3e5f5 !important;
    }
    
    /* Expander content */
    .streamlit-expanderHeader {
        color: #e0e0e0 !important;
    }
    
    /* Ensure caption text is visible */
    .stCaption {
        color: #b0b0b0 !important;
    }
    
    /* Button styling for dark theme */
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #e0e0e0 !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = 'First National Bank'
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'CSM'

# Sample data generation
@st.cache_data
def generate_customer_data():
    """Generate sample customer data for 1,008 customers"""
    np.random.seed(42)
    
    customers = []
    for i in range(1008):
        # Generate more realistic bank names
        bank_types = ['National', 'Regional', 'Community', 'First', 'Central', 'State']
        bank_suffixes = ['Bank', 'Trust', 'Financial', 'Credit Union', 'Savings']
        locations = ['Springfield', 'Austin', 'Dallas', 'Chicago', 'Boston', 'Miami']
        
        name = f"{random.choice(bank_types)} {random.choice(locations)} {random.choice(bank_suffixes)}"
        
        # Generate correlated metrics
        base_health = np.random.normal(75, 15)
        base_health = max(20, min(100, base_health))  # Clamp between 20-100
        
        customer = {
            'id': f'CUST-{i+1:04d}',
            'name': name if i > 0 else 'First National Bank',
            'tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], p=[0.2, 0.5, 0.3]),
            'arr': np.random.lognormal(13, 1),  # Log-normal distribution for ARR
            'health_score': base_health,
            'usage_trend': np.random.normal(0, 20),
            'support_tickets_open': np.random.poisson(1.5),
            'support_tickets_critical': np.random.poisson(0.3),
            'days_since_contact': np.random.exponential(10),
            'cecl_completion': np.random.beta(7, 3) * 100,
            'data_integrity': base_health + np.random.normal(0, 10),
            'credit_pull_match_rate': 90 + np.random.normal(0, 5),
            'sentiment_score': base_health/100 + np.random.normal(0, 0.1),
            'expansion_probability': max(0, min(1, (base_health/100) + np.random.normal(0, 0.2))),
            'churn_risk': max(0, min(1, 1 - (base_health/100) + np.random.normal(0, 0.15)))
        }
        
        # Ensure First National Bank has specific values
        if i == 0:
            customer.update({
                'name': 'First National Bank of Springfield',
                'tier': 'Tier 1',
                'arr': 2100000,
                'health_score': 82,
                'usage_trend': -23,
                'support_tickets_open': 2,
                'support_tickets_critical': 2,
                'days_since_contact': 2,
                'cecl_completion': 75,
                'data_integrity': 78,
                'credit_pull_match_rate': 89,
                'sentiment_score': 0.45,
                'expansion_probability': 0.15,
                'churn_risk': 0.35
            })
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

@st.cache_data
def generate_support_tickets(customer_name):
    """Generate support tickets for a specific customer"""
    tickets = []
    
    if customer_name == 'First National Bank of Springfield':
        tickets = [
            {
                'id': 'T-2024-1823',
                'title': 'CECL calculation discrepancy in PD models',
                'status': 'Open',
                'priority': 'Critical',
                'days_open': 5,
                'assigned_to': 'Bob Smith (DSE)'
            },
            {
                'id': 'T-2024-1847',
                'title': 'Data mapping error in commercial loan portfolio',
                'status': 'Open',
                'priority': 'Critical',
                'days_open': 2,
                'assigned_to': 'Bob Smith (DSE)'
            }
        ]
    else:
        # Generate random tickets for other customers
        num_tickets = np.random.poisson(1.5)
        for i in range(num_tickets):
            tickets.append({
                'id': f'T-2024-{np.random.randint(1000, 9999)}',
                'title': np.random.choice([
                    'Data quality issue in loan portfolio',
                    'CECL model configuration question',
                    'Credit pull match rate declining',
                    'User access permissions update',
                    'Report generation error'
                ]),
                'status': np.random.choice(['Open', 'In Progress', 'Resolved'], p=[0.5, 0.3, 0.2]),
                'priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.3, 0.4, 0.2, 0.1]),
                'days_open': np.random.exponential(3),
                'assigned_to': np.random.choice(['Bob Smith (DSE)', 'Alice Johnson (DSE)', 'Tom Wilson (DSE)'])
            })
    
    return pd.DataFrame(tickets)

def calculate_health_score_components(customer_data):
    """Calculate detailed health score components"""
    components = {
        'Support Health': (100 - customer_data['support_tickets_open'] * 10 - 
                          customer_data['support_tickets_critical'] * 20),
        'Usage Metrics': 100 + customer_data['usage_trend'],
        'Engagement': max(0, 100 - customer_data['days_since_contact'] * 2),
        'Data Quality': customer_data['data_integrity'],
        'Financial': 100 if customer_data['arr'] > 1000000 else 80,
        'CECL Readiness': customer_data['cecl_completion']
    }
    
    # Ensure all scores are between 0 and 100
    components = {k: max(0, min(100, v)) for k, v in components.items()}
    
    # Weighted average
    weights = {
        'Support Health': 0.30,
        'Usage Metrics': 0.25,
        'Engagement': 0.20,
        'Data Quality': 0.10,
        'Financial': 0.10,
        'CECL Readiness': 0.05
    }
    
    overall_score = sum(components[k] * weights[k] for k in components)
    
    return components, overall_score

def generate_ai_recommendations(customer_data, view_mode):
    """Generate AI recommendations based on customer data and view mode"""
    recommendations = []
    
    if view_mode == 'CSM':
        if customer_data['sentiment_score'] < 0.5:
            recommendations.append({
                'priority': 'High',
                'action': 'Schedule executive check-in TODAY',
                'reason': f"Sentiment declined to {customer_data['sentiment_score']:.2f} - relationship at risk"
            })
        
        if customer_data['support_tickets_critical'] > 0:
            recommendations.append({
                'priority': 'High',
                'action': 'Escalate critical tickets to senior DSE',
                'reason': f"{customer_data['support_tickets_critical']} critical tickets affecting customer satisfaction"
            })
        
        if customer_data['cecl_completion'] < 80:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Offer CECL advisory services',
                'reason': f"CECL implementation only {customer_data['cecl_completion']:.0f}% complete with audit approaching"
            })
            
    elif view_mode == 'DSE':
        if customer_data['data_integrity'] < 80:
            recommendations.append({
                'priority': 'High',
                'action': 'Run data quality remediation script',
                'reason': f"Data integrity at {customer_data['data_integrity']:.0f}% - affecting {127} required fields"
            })
        
        if customer_data['credit_pull_match_rate'] < 90:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Review and fix SSN format inconsistencies',
                'reason': f"Match rate declined to {customer_data['credit_pull_match_rate']:.0f}% - pattern detected across 47 similar banks"
            })
            
    elif view_mode == 'PSO':
        if customer_data['expansion_probability'] < 0.3 and customer_data['cecl_completion'] < 80:
            recommendations.append({
                'priority': 'High',
                'action': 'Propose CECL Advisory Package ($125K)',
                'reason': f"85% close probability - customer struggling with CECL implementation"
            })
        
        if customer_data['churn_risk'] > 0.3:
            recommendations.append({
                'priority': 'High',
                'action': 'Include complimentary optimization workshop',
                'reason': f"Retention play - churn risk at {customer_data['churn_risk']*100:.0f}%"
            })
    
    return recommendations

def render_health_score_gauge(score):
    """Create a gauge chart for health score"""
    if score >= 80:
        color = "#00C851"
    elif score >= 60:
        color = "#ffbb33"
    else:
        color = "#ff4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Health Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "#ffebee"},
                {'range': [60, 80], 'color': "#fff3e0"},
                {'range': [80, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Main App
def main():
    # Load data
    customers_df = generate_customer_data()
    
# ADD THEME TOGGLE HERE - Right at the beginning of main()
    theme = st.sidebar.selectbox(
        "🎨 Theme",
        ["Dark", "Light"],
        index=0  # Default to Dark
    )
    
    # Apply theme-specific CSS
    if theme == "Light":
        st.markdown("""
        <style>
            /* Light theme overrides */
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            
            .stApp h1, .stApp h2, .stApp h3 {
                color: #1f2937 !important;
            }
            
            .stApp .stMarkdown {
                color: #374151 !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: #6b7280 !important;
            }
            
            [data-testid="stMetricValue"] {
                color: #1f2937 !important;
            }
            
            [data-testid="metric-container"] {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
            }
            
            .stSidebar {
                background-color: #f3f4f6;
                color: #1f2937;
            }
            
            .recommendation-box {
                background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
                color: #4a148c !important;
            }
            
            .recommendation-box h3, .recommendation-box h4 {
                color: #6a1b9a !important;
            }
            
            .recommendation-box li {
                color: #4a148c !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Dark theme CSS (the one we just created)
        st.markdown("""
        <style>
            /* Your dark theme CSS here - the one from my previous message */
            /* ... */
        </style>
        """, unsafe_allow_html=True)
     
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🚀 AI-Powered Customer Snapshot Dashboard")
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            ["CSM", "DSE", "PSO"],
            index=["CSM", "DSE", "PSO"].index(st.session_state.view_mode)
        )
        st.session_state.view_mode = view_mode
    with col3:
        st.metric("Total Customers", "1,008", "237.5 per CSM")
    
    # Sidebar for customer selection
    with st.sidebar:
        st.header("Customer Selection")
        
        # Quick filters
        filter_type = st.radio(
            "Quick Filters",
            ["All Customers", "At Risk", "Opportunities", "My Accounts"]
        )
        
        if filter_type == "At Risk":
            filtered_df = customers_df[customers_df['health_score'] < 70]
        elif filter_type == "Opportunities":
            filtered_df = customers_df[customers_df['expansion_probability'] > 0.7]
        elif filter_type == "My Accounts":
            # Simulate CSM assignment
            filtered_df = customers_df.iloc[:238]  # Sarah's 237.5 customers
        else:
            filtered_df = customers_df
        
        selected_customer = st.selectbox(
            "Select Customer",
            filtered_df['name'].tolist(),
            index=0
        )
        
        st.session_state.selected_customer = selected_customer
        
        # Quick stats
        st.markdown("---")
        st.subheader("Portfolio Overview")
        at_risk = len(customers_df[customers_df['health_score'] < 70])
        opportunities = len(customers_df[customers_df['expansion_probability'] > 0.7])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("At Risk", at_risk, f"{at_risk/10.08:.1f}%", delta_color="inverse")
        with col2:
            st.metric("Opportunities", opportunities, f"${opportunities*125}K")
    
    # Get selected customer data
    customer_data = customers_df[customers_df['name'] == selected_customer].iloc[0]
    tickets_df = generate_support_tickets(selected_customer)
    
    # Main content area based on view mode
    if view_mode == "CSM":
        render_csm_view(customer_data, tickets_df)
    elif view_mode == "DSE":
        render_dse_view(customer_data, tickets_df, customers_df)
    elif view_mode == "PSO":
        render_pso_view(customer_data, customers_df)

def render_csm_view(customer_data, tickets_df):
    """Render the CSM view"""
    st.header(f"CSM View - {customer_data['name']}")
    
    # Customer header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(customer_data['name'])
        st.caption(f"{customer_data['tier']} • ARR: ${customer_data['arr']:,.0f} • CSM: Sarah Johnson")
    with col2:
        components, overall_score = calculate_health_score_components(customer_data)
        st.plotly_chart(render_health_score_gauge(overall_score), use_container_width=True)
    with col3:
        st.metric("Days Since Contact", f"{customer_data['days_since_contact']:.0f}")
        st.metric("Usage Trend", f"{customer_data['usage_trend']:.0f}%", customer_data['usage_trend'])
    with col4:
        st.metric("Open Tickets", customer_data['support_tickets_open'])
        st.metric("Critical", customer_data['support_tickets_critical'], delta_color="inverse")
    
    # AI Meeting Prep
    if customer_data['name'] == 'First National Bank of Springfield':
        st.markdown("""
        <div class='recommendation-box'>
            <h3>🤖 AI Meeting Preparation - Today 2:00 PM</h3>
            <p><strong>Meeting Topic:</strong> CECL Model Validation & Q4 Results Review</p>
            <h4>Key Discussion Points:</h4>
            <ul>
                <li>Jane (CFO) expressed frustration in email 2 days ago about CECL audit deadline</li>
                <li>Customer has 2 open support tickets related to CECL calculations (5+ days)</li>
                <li>Communication frequency down 40% - potential disengagement risk</li>
                <li>Professional Services project "CECL Advisory Review" is 75% complete</li>
                <li>Next credit pull scheduled in 5 days - match rate trending down (89%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Health Analysis", "🎫 Support Tickets", "🤖 AI Recommendations", "📈 Trends"])
    
    with tab1:
        st.subheader("Health Score Breakdown")
        components, _ = calculate_health_score_components(customer_data)
        
        fig = px.bar(
            x=list(components.values()),
            y=list(components.keys()),
            orientation='h',
            color=list(components.values()),
            color_continuous_scale=['#ff4444', '#ffbb33', '#00C851'],
            range_color=[0, 100]
        )
        fig.update_layout(
            xaxis_title="Score",
            yaxis_title="Component",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Support Tickets")
        if not tickets_df.empty:
            for _, ticket in tickets_df.iterrows():
                with st.expander(f"{ticket['id']} - {ticket['title']}", expanded=ticket['status']=='Open'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Status:** {ticket['status']}")
                    with col2:
                        st.write(f"**Priority:** {ticket['priority']}")
                    with col3:
                        st.write(f"**Days Open:** {ticket['days_open']:.0f}")
                    st.write(f"**Assigned to:** {ticket['assigned_to']}")
        else:
            st.info("No active tickets")
    
    with tab3:
        st.subheader("AI Recommendations")
        recommendations = generate_ai_recommendations(customer_data, 'CSM')
        for rec in recommendations:
            st.warning(f"**{rec['priority']} Priority:** {rec['action']}")
            st.write(f"*Reason: {rec['reason']}*")
    
    with tab4:
        st.subheader("Historical Trends")
        # Generate mock historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        historical_health = [customer_data['health_score'] + np.random.normal(0, 3) for _ in range(30)]
        
        fig = px.line(x=dates, y=historical_health, title="Health Score Trend (Last 30 Days)")
        fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        st.plotly_chart(fig, use_container_width=True)

def render_dse_view(customer_data, tickets_df, all_customers_df):
    """Render the DSE view"""
    st.header(f"DSE View - Technical Analysis")
    
    # Pattern Detection Alert
    if customer_data['credit_pull_match_rate'] < 90:
        st.error(f"""
        ⚠️ **Critical Pattern Detected Across Multiple Customers**
        
        47 banks will experience the same CECL calculation error that {customer_data['name']} 
        is facing if not addressed proactively.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔧 Deploy Bulk Fix to 47 Banks", type="primary"):
                with st.spinner("Deploying fix..."):
                    time.sleep(2)
                st.success("Fix deployed successfully! 47 banks updated.")
        with col2:
            st.metric("Estimated Time Saved", "156 hours")
        with col3:
            st.metric("Tickets Prevented", "141")
    
    # Customer Technical Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{customer_data['name']} - Technical Status")
        st.metric("Data Integrity Score", f"{customer_data['data_integrity']:.0f}%")
        st.metric("Credit Pull Match Rate", f"{customer_data['credit_pull_match_rate']:.0f}%")
        st.metric("CECL Models Configured", f"{int(customer_data['cecl_completion']*0.15)}/15")
        st.metric("Missing Required Fields", "127" if customer_data['data_integrity'] < 80 else "23")
    
    with col2:
        st.subheader("System Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['API Response Time', 'Data Processing', 'Model Accuracy', 'System Uptime'],
            'Value': ['127ms', '4.2 hours', '92%', '99.9%'],
            'Status': ['Good', 'Normal', 'Good', 'Excellent']
        })
        st.dataframe(metrics_df, hide_index=True)
    
    # Proactive Issue Prevention
    st.subheader("🛡️ Proactive Issue Prevention")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ **Completed Today**")
        st.write("Fixed DTI calculation error for 12 banks before month-end")
    with col2:
        st.warning("🔄 **In Progress**")
        st.write("Updating loan categorization logic for Q4 compliance")
    with col3:
        st.info("📅 **Scheduled**")
        st.write("Credit pull optimization for 23 banks with <90% match rates")
    
    # Technical Recommendations
    st.subheader("🤖 Technical AI Recommendations")
    recommendations = generate_ai_recommendations(customer_data, 'DSE')
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. **{rec['action']}**")
        st.caption(f"   *{rec['reason']}*")

def render_pso_view(customer_data, all_customers_df):
    """Render the PSO view"""
    st.header(f"PSO View - Revenue Opportunities")
    
    # Opportunity Highlight
    if customer_data['cecl_completion'] < 80:
        st.success(f"""
        💰 **{customer_data['name']} - High-Value Opportunity Detected**
        
        **$125,000** - CECL Advisory Services Package
        
        **85% close probability** based on current risk factors
        """)
        
        if st.button("📄 Generate Proposal", type="primary"):
            st.balloons()
            st.success("Proposal generated! Sent to your email.")
    
    # Customer Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Opportunity Analysis")
        st.write("**Why This Opportunity Will Close:**")
        reasons = [
            "CFO expressed audit concerns 3 times in recent communications",
            "CECL implementation is only 75% complete with audit in 60 days",
            "2 critical tickets indicate need for expert guidance",
            "Similar banks (87%) purchased advisory when facing audit",
            "Budget approved for Q4 professional services"
        ]
        for reason in reasons:
            st.write(f"• {reason}")
    
    with col2:
        st.subheader("Current Project Status")
        st.write("**CECL Advisory Review**")
        st.progress(0.75)
        st.caption("75% Complete - On Track")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hours Used", "112/150")
            st.metric("Deliverables", "8/11")
        with col2:
            st.metric("CSAT Score", "4.5/5.0")
            st.metric("On Budget", "Yes")
    
    # Opportunity Pipeline
    st.subheader("🎯 AI-Identified Opportunities Across Portfolio")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**High Probability (>80%)**")
        st.metric("Total Value", "$875K")
        st.write("• First National - CECL ($125K)")
        st.write("• State Bank - Data Quality ($200K)")
        st.write("• Regional CU - Full Impl ($550K)")
    
    with col2:
        st.warning("**Medium Probability (50-80%)**")
        st.metric("Total Value", "$625K")
        st.write("• Community Bank - Training ($75K)")
        st.write("• Metro Financial - Optimization ($150K)")
        st.write("• Central Trust - Advisory ($400K)")
    
    with col3:
        st.info("**Developing (<50%)**")
        st.metric("Total Value", "$300K")
        st.write("• Valley Bank - Assessment ($50K)")
        st.write("• Coastal Credit - Workshop ($100K)")
        st.write("• Mountain Trust - Review ($150K)")
    
    # Success Patterns - Fixed the syntax error here
    with st.expander("📈 AI Learning: What Drives Advisory Sales", expanded=True):
        st.write("**Top 5 Indicators (based on 127 successful engagements):**")
        patterns = [
            ("Upcoming regulatory audit within 90 days", "92%"),
            ("Multiple CECL-related support tickets", "87%"),
            ("Executive mentions 'compliance concerns'", "85%"),
            ("Data quality score below 80%", "83%"),  # This line was previously cut off
            ("Peer banks in region already engaged us", "81%")
        ]
        
        for pattern, rate in patterns:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {pattern}")
            with col2:
                st.write(f"**{rate}** close rate")

if __name__ == "__main__":
    main()
