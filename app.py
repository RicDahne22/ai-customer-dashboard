# AI-Powered Customer Snapshot Dashboard POC - Updated Version
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

# Custom CSS for better styling - handles both light and dark themes
st.markdown("""
<style>
    /* Theme-aware metric cards */
    [data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Remove forced colors - let Streamlit handle theme colors */
    [data-testid="metric-container"] label {
        opacity: 0.8;
    }
    
    /* Health score styles */
    .health-score-good { background-color: #00C851; }
    .health-score-warning { background-color: #ffbb33; }
    .health-score-critical { background-color: #ff4444; }
    
    /* Recommendation box with better contrast */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(148, 0, 211, 0.1) 0%, rgba(148, 0, 211, 0.2) 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(148, 0, 211, 0.3);
        margin: 10px 0;
    }
    
    /* Alert boxes with better contrast */
    .alert-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 8px;
    }
    
    .alert-box.critical {
        background: rgba(220, 53, 69, 0.1);
        border-left-color: #dc3545;
    }
    
    .pattern-alert {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 152, 0, 0.2));
        border: 1px solid rgba(255, 152, 0, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .opportunity-highlight {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.2));
        border: 1px solid rgba(76, 175, 80, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Remove any forced text colors */
    .stMarkdown, .stText {
        color: inherit;
    }
    
    /* Ensure plotly charts work in both themes */
    .js-plotly-plot {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = 'First National Bank of Springfield'
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'CSM'

# Sample data generation
@st.cache_data
def generate_customer_data():
    """Generate sample customer data for 1,008 customers"""
    np.random.seed(42)
    
    customers = []
    for i in range(1008):
        # Generate base metrics
        health_score = np.random.normal(75, 15)
        health_score = max(0, min(100, health_score))
        
        # Special case for First National Bank
        if i == 0:
            customer = {
                'name': 'First National Bank of Springfield',
                'tier': 'Tier 1',
                'csm': 'Sarah Johnson',
                'customer_since': 'Jan 2021',
                'health_score': 82,
                'churn_risk': 0.15,
                'expansion_probability': 0.85,
                'monthly_value': 45000,
                'cecl_completion': 75,
                'support_tickets_open': 2,
                'support_tickets_critical': 2,
                'days_since_contact': 2,
                'usage_trend': -12,
                'data_quality_score': 78,
                'bureau_match_rate': 67,
                'last_bureau_submission': '5 days',
                'missing_fields': 127,
                'cecl_model_status': '12/15 Configured',
                'ps_project_active': True,
                'ps_project_name': 'CECL Advisory Review',
                'ps_project_completion': 75,
                'ps_hours_used': 112,
                'ps_hours_total': 150,
                'opportunity_value': 125000,
                'opportunity_type': 'CECL Advisory Services Package',
                'database_records': 45287,
                'bureau_cost': 4250,
                'data_issues': 'SSN format (24%), Address missing (18%), Name standardization (12%)'
            }
        else:
            # Generate random customer
            customer = {
                'name': f'Customer Bank {i+1}',
                'tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], p=[0.2, 0.5, 0.3]),
                'csm': np.random.choice(['Sarah Johnson', 'Mike Chen', 'Lisa Brown', 'Tom Wilson']),
                'customer_since': f"{np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {np.random.randint(2019, 2024)}",
                'health_score': health_score,
                'churn_risk': (100 - health_score) / 100 * np.random.uniform(0.8, 1.2),
                'expansion_probability': health_score / 100 * np.random.uniform(0.5, 1.0),
                'monthly_value': np.random.uniform(5000, 75000),
                'cecl_completion': np.random.uniform(40, 100),
                'support_tickets_open': np.random.poisson(1),
                'support_tickets_critical': np.random.binomial(2, 0.1),
                'days_since_contact': np.random.exponential(15),
                'usage_trend': np.random.normal(0, 20),
                'data_quality_score': np.random.uniform(65, 98),
                'bureau_match_rate': np.random.uniform(60, 98),
                'last_bureau_submission': f"{np.random.randint(1, 30)} days",
                'missing_fields': np.random.randint(0, 200),
                'cecl_model_status': f"{np.random.randint(8, 15)}/15 Configured",
                'ps_project_active': np.random.choice([True, False], p=[0.3, 0.7]),
                'ps_project_name': np.random.choice(['CECL Advisory', 'Data Quality', 'Implementation', 'Optimization']),
                'ps_project_completion': np.random.uniform(0, 100),
                'ps_hours_used': np.random.randint(0, 200),
                'ps_hours_total': np.random.randint(100, 300),
                'opportunity_value': np.random.choice([0, 50000, 75000, 100000, 125000, 150000, 200000], p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]),
                'opportunity_type': np.random.choice(['CECL Advisory', 'Data Quality Package', 'Full Implementation', 'Optimization Services']),
                'database_records': np.random.randint(10000, 100000),
                'bureau_cost': np.random.randint(1000, 6000),
                'data_issues': 'Various data quality issues'
            }
        
        # Normalize values
        customer['churn_risk'] = max(0, min(1, customer['churn_risk']))
        customer['expansion_probability'] = max(0, min(1, customer['expansion_probability']))
        
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
                'assigned_to': 'Bob Smith (DSE)',
                'root_cause': 'Incorrect date format in loan origination data'
            },
            {
                'id': 'T-2024-1847',
                'title': 'Data mapping error in commercial loan portfolio',
                'status': 'Open',
                'priority': 'Critical',
                'days_open': 2,
                'assigned_to': 'Bob Smith (DSE)',
                'root_cause': 'Impact: 2,341 loans incorrectly categorized'
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
                'assigned_to': np.random.choice(['Bob Smith (DSE)', 'Alice Johnson (DSE)', 'Tom Wilson (DSE)']),
                'root_cause': 'Under investigation'
            })
    
    return pd.DataFrame(tickets)

def calculate_health_score_components(customer_data):
    """Calculate detailed health score components"""
    components = {
        'Support Health': (100 - customer_data['support_tickets_open'] * 10 - 
                          customer_data['support_tickets_critical'] * 20),
        'Usage Metrics': 100 + customer_data['usage_trend'],
        'Engagement': max(0, 100 - customer_data['days_since_contact'] * 2),
        'Data Quality': customer_data['data_quality_score'],
        'Financial': 100 * (1 - customer_data['churn_risk']),
        'CECL Readiness': customer_data['cecl_completion']
    }
    
    # Ensure all scores are within 0-100
    for key in components:
        components[key] = max(0, min(100, components[key]))
    
    # Weighted average
    weights = {
        'Support Health': 0.30,
        'Usage Metrics': 0.25,
        'Engagement': 0.20,
        'Data Quality': 0.10,
        'Financial': 0.10,
        'CECL Readiness': 0.05
    }
    
    weighted_score = sum(components[k] * weights[k] for k in components)
    
    return components, weighted_score

def generate_ai_recommendations(customer_data, view_mode):
    """Generate AI recommendations based on customer data and view mode"""
    recommendations = []
    
    if view_mode == 'CSM':
        if customer_data['support_tickets_critical'] > 0:
            recommendations.append({
                'priority': 'High',
                'action': 'Address critical support tickets before meeting',
                'reason': f"{customer_data['support_tickets_critical']} critical tickets open for {customer_data['name']}"
            })
        
        if customer_data['days_since_contact'] > 30:
            recommendations.append({
                'priority': 'High',
                'action': 'Schedule immediate check-in call',
                'reason': f"No contact in {customer_data['days_since_contact']:.0f} days - engagement risk"
            })
            
        if customer_data['churn_risk'] > 0.3:
            recommendations.append({
                'priority': 'Critical',
                'action': 'Escalate to executive team for retention strategy',
                'reason': f"Churn risk at {customer_data['churn_risk']*100:.0f}% - immediate action required"
            })
            
    elif view_mode == 'DSE':
        if customer_data['data_quality_score'] < 80:
            recommendations.append({
                'priority': 'High',
                'action': 'Run automated data quality cleanup scripts',
                'reason': f"Data quality at {customer_data['data_quality_score']:.0f}% affects credit bureau submissions"
            })
            
        if customer_data['bureau_match_rate'] < 90:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Review and fix SSN format inconsistencies',
                'reason': f"Match rate at {customer_data['bureau_match_rate']:.0f}% - pattern detected across 47 similar banks"
            })
            
    elif view_mode == 'PSO':
        if customer_data['expansion_probability'] > 0.7 and customer_data['cecl_completion'] < 80:
            recommendations.append({
                'priority': 'High',
                'action': f"Propose {customer_data['opportunity_type']} (${customer_data['opportunity_value']:,.0f})",
                'reason': f"{customer_data['expansion_probability']*100:.0f}% close probability - customer struggling with CECL implementation"
            })
        
        if customer_data['churn_risk'] > 0.3:
            recommendations.append({
                'priority': 'High',
                'action': 'Include complimentary optimization workshop',
                'reason': f"Retention play - churn risk at {customer_data['churn_risk']*100:.0f}%"
            })
            
    elif view_mode == 'Credit':
        if customer_data['bureau_match_rate'] < 85:
            recommendations.append({
                'priority': 'Critical',
                'action': 'Run database validation before bureau submission',
                'reason': f"Predicted {customer_data['bureau_match_rate']:.0f}% match rate will result in ${customer_data['bureau_cost']*0.3:,.0f} in rejection fees"
            })
        
        if customer_data['missing_fields'] > 100:
            recommendations.append({
                'priority': 'High',
                'action': 'Execute data cleanup scripts for missing fields',
                'reason': f"{customer_data['missing_fields']} missing fields affecting bureau acceptance"
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
        title = {'text': "Health Score", 'font': {'color': 'inherit'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'inherit'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "rgba(255, 68, 68, 0.2)"},
                {'range': [60, 80], 'color': "rgba(255, 187, 51, 0.2)"},
                {'range': [80, 100], 'color': "rgba(0, 200, 81, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        },
        number={'font': {'color': 'inherit'}}
    ))
    
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'inherit'}
    )
    return fig

# Main App
def main():
    # Load data
    customers_df = generate_customer_data()
    
    # Header with view mode selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🚀 AI-Powered Customer Snapshot Dashboard")
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            ["CSM", "DSE", "PSO", "Credit"],
            index=["CSM", "DSE", "PSO", "Credit"].index(st.session_state.view_mode),
            format_func=lambda x: {
                "CSM": "👥 CSM - Sarah Johnson",
                "DSE": "🔧 DSE - Bob Smith", 
                "PSO": "🎯 PSO - Mary Wilson",
                "Credit": "💳 Credit - Lisa & Mark"
            }[x]
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
            ["All Customers", "At Risk", "Opportunities", "My Accounts", "Poor Data Quality"]
        )
        
        if filter_type == "At Risk":
            filtered_df = customers_df[customers_df['health_score'] < 70]
        elif filter_type == "Opportunities":
            filtered_df = customers_df[customers_df['expansion_probability'] > 0.7]
        elif filter_type == "My Accounts":
            # Simulate CSM assignment for Sarah's customers
            filtered_df = customers_df[customers_df['csm'] == 'Sarah Johnson']
        elif filter_type == "Poor Data Quality":
            filtered_df = customers_df[customers_df['data_quality_score'] < 85]
        else:
            filtered_df = customers_df
        
        selected_customer = st.selectbox(
            "Select Customer",
            filtered_df['name'].tolist(),
            index=0 if 'First National Bank of Springfield' in filtered_df['name'].tolist() else 0
        )
        
        st.session_state.selected_customer = selected_customer
        
        # Quick stats
        st.markdown("---")
        st.subheader("Portfolio Overview")
        at_risk = len(customers_df[customers_df['health_score'] < 70])
        opportunities = len(customers_df[customers_df['expansion_probability'] > 0.7])
        poor_data = len(customers_df[customers_df['data_quality_score'] < 85])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("At Risk", at_risk, f"{at_risk/10.08:.1f}%", delta_color="inverse")
        with col2:
            st.metric("Opportunities", opportunities, f"${opportunities*125}K")
        
        st.metric("Poor Data Quality", poor_data, f"{poor_data/10.08:.1f}%", delta_color="inverse")
    
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
    elif view_mode == "Credit":
        render_credit_view(customer_data, customers_df)

def render_csm_view(customer_data, tickets_df):
    """Render the CSM view"""
    st.header(f"CSM View - {customer_data['name']}")
    st.subheader("Unified view of customer health, support, and engagement for Sarah's 237.5 customers")
    
    # Customer header with health score
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### {customer_data['name']}")
        st.markdown(f"📍 **{customer_data['tier']} Customer** | 📅 **Since:** {customer_data['customer_since']} | 👤 **CSM:** {customer_data['csm']}")
        if customer_data['name'] == 'First National Bank of Springfield':
            st.markdown("📞 **Next Meeting:** Today 2:00 PM")
    
    with col2:
        fig = render_health_score_gauge(customer_data['health_score'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.metric("Churn Risk", f"{customer_data['churn_risk']*100:.0f}%", 
                  "High" if customer_data['churn_risk'] > 0.3 else "Low", 
                  delta_color="inverse")
    
    with col4:
        st.metric("Expansion Prob.", f"{customer_data['expansion_probability']*100:.0f}%",
                  "High" if customer_data['expansion_probability'] > 0.7 else "Low")
    
    # Main dashboard grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # AI Meeting Preparation
        st.markdown("### 🤖 AI Meeting Preparation")
        
        if customer_data['name'] == 'First National Bank of Springfield':
            st.warning("⚠️ **Upcoming Meeting Context:** Meeting Topic: 'CECL Model Validation & Q4 Results Review'")
            
            st.markdown("#### Key Discussion Points:")
            st.markdown("""
            • Jane (CFO) expressed frustration in email 2 days ago about CECL audit deadline
            • Customer has 2 open support tickets related to CECL calculations (5+ days)
            • Communication frequency down 40% - potential disengagement risk
            • Professional Services project "CECL Advisory Review" is 75% complete
            • Next credit pull scheduled in 5 days - match rate trending down (89%)
            """)
            
            st.info("✨ **AI Recommendation**\n\nLead with empathy about audit pressure. Present concrete resolution timeline for CECL tickets. Offer executive escalation path with dedicated resources. Consider bringing PS team lead to demonstrate commitment. Have specific dates for all deliverables ready.")
        
        # Support Tickets
        st.markdown("### 🎫 Support Tickets")
        if len(tickets_df) > 0:
            open_tickets = tickets_df[tickets_df['status'] == 'Open']
            if len(open_tickets) > 0:
                st.error(f"{len(open_tickets)} Open Tickets")
                for _, ticket in open_tickets.iterrows():
                    with st.expander(f"{ticket['id']} - {ticket['title']}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Status:** {ticket['status']} - {ticket['days_open']:.0f} days")
                            st.markdown(f"**Priority:** {ticket['priority']}")
                        with col2:
                            st.markdown(f"**Assigned to:** {ticket['assigned_to']}")
                            if 'root_cause' in ticket and ticket['root_cause']:
                                st.markdown(f"**Root Cause:** {ticket['root_cause']}")
                
                if customer_data['name'] == 'First National Bank of Springfield':
                    st.info("✨ **DSE Alert:** Bob Smith (DSE) is actively working on both tickets. ETA: Tomorrow 10 AM")
        else:
            st.success("No open tickets")
    
    with col2:
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("📞 Schedule Executive Call", use_container_width=True):
            st.success("Executive call scheduled!")
        if st.button("🎫 Escalate Tickets to DSE", use_container_width=True, type="secondary"):
            st.success("Tickets escalated to DSE team!")
        if st.button("📊 Request PS Proposal", use_container_width=True, type="secondary"):
            st.success("PS proposal request sent!")
        if st.button("📝 Generate Meeting Notes", use_container_width=True):
            st.success("Meeting notes generated!")
    
    # AI Risk Assessment
    st.markdown("### 🎯 AI Risk Assessment")
    recommendations = generate_ai_recommendations(customer_data, 'CSM')
    
    if customer_data['churn_risk'] > 0.3:
        st.error(f"🔴 **Escalation Risk: Medium-High** - Multiple indicators suggest customer frustration with CECL implementation timeline")
    
    st.markdown("#### Recommended Actions:")
    for i, rec in enumerate(recommendations, 1):
        if rec['priority'] == 'Critical':
            st.error(f"{i}. **{rec['action']}** - {rec['reason']}")
        elif rec['priority'] == 'High':
            st.warning(f"{i}. **{rec['action']}** - {rec['reason']}")
        else:
            st.info(f"{i}. **{rec['action']}** - {rec['reason']}")
    
    # Additional recommendations for First National Bank
    if customer_data['name'] == 'First National Bank of Springfield':
        st.markdown("""
        **Additional Recommendations:**
        - **Immediate:** Address open CECL calculation tickets before today's meeting
        - **This Week:** Schedule joint session with DSE team to resolve data mapping issues
        - **This Month:** Fast-track remaining CECL Advisory project deliverables
        - **Proactive:** Offer complimentary CECL validation workshop to prepare for audit
        """)

def render_dse_view(customer_data, tickets_df, customers_df):
    """Render the DSE view"""
    st.header("AI-Powered Technical Dashboard - DSE View")
    st.subheader("Pattern detection and proactive issue resolution across 1,008 customers")
    
    # Critical Pattern Alert
    if customer_data['name'] == 'First National Bank of Springfield':
        with st.container():
            st.warning("### ⚠️ Critical Pattern Detected Across Multiple Customers")
            st.markdown("**47 banks will experience the same CECL calculation error** that First National Bank is facing if not addressed proactively.")
            if st.button("Deploy Bulk Fix to 47 Banks", type="primary"):
                st.success("Bulk fix deployed successfully! 47 banks protected from CECL calculation errors.")
    
    # Main dashboard grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Customer Technical Overview
        st.markdown(f"### 🏦 {customer_data['name']} - Technical Overview")
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Data Integrity Score", f"{customer_data['data_quality_score']:.0f}%",
                     "Poor" if customer_data['data_quality_score'] < 80 else "Good")
            st.metric("Last Successful Credit Pull", customer_data['last_bureau_submission'],
                     f"{customer_data['bureau_match_rate']:.0f}% match")
        with col1b:
            st.metric("Missing Required Fields", customer_data['missing_fields'],
                     "High" if customer_data['missing_fields'] > 100 else "Low", delta_color="inverse")
            st.metric("CECL Model Status", customer_data['cecl_model_status'])
        
        # Active Tickets
        if len(tickets_df) > 0:
            st.markdown("#### Active Tickets:")
            for _, ticket in tickets_df.iterrows():
                with st.expander(f"{ticket['id']} - {ticket['title']}", expanded=True):
                    st.markdown(f"**Status:** {ticket['status']} | **Priority:** {ticket['priority']}")
                    if 'root_cause' in ticket:
                        st.markdown(f"**Root Cause:** {ticket['root_cause']}")
    
    with col2:
        # System Performance
        st.markdown("### 📊 System Performance")
        st.metric("Credit Pull Success Rate", "97.5%")
        st.metric("Avg Processing Time", "4.2 hours")
        st.metric("Data Quality Grade", "B+")
        st.metric("API Response Time", "127ms")
    
    # Proactive Issue Prevention
    st.markdown("### 🛡️ Proactive Issue Prevention")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ **Completed**")
        st.markdown("Fixed DTI calculation error for 12 banks before month-end")
    
    with col2:
        st.warning("🔄 **In Progress**")
        st.markdown("Updating loan categorization logic for Q4 compliance")
    
    with col3:
        st.info("📅 **Scheduled**")
        st.markdown("Credit pull optimization for 23 banks with <90% match rates")
    
    # AI Insights
    st.markdown("### ✨ Technical AI Recommendations")
    recommendations = generate_ai_recommendations(customer_data, 'DSE')
    
    st.markdown("""
    1. Deploy date format standardization script to prevent 80% of CECL calculation errors
    2. Implement automated data quality checks 48 hours before each credit pull
    3. Create bulk update tool for missing DTI fields affecting 23% of commercial loans
    4. Schedule maintenance window for Q4 compliance updates (affects 156 banks)
    """)
    
    for rec in recommendations:
        if rec['priority'] == 'High':
            st.warning(f"**{rec['action']}** - {rec['reason']}")
        else:
            st.info(f"**{rec['action']}** - {rec['reason']}")

def render_pso_view(customer_data, customers_df):
    """Render the PSO view"""
    st.header("AI-Powered Opportunity Dashboard - PSO View")
    st.subheader("Revenue opportunities and project insights across all 1,008 customers")
    
    # Opportunity Highlight
    if customer_data['expansion_probability'] > 0.7:
        with st.container():
            st.success(f"### 💰 {customer_data['name']} - High-Value Opportunity Detected")
            st.markdown(f"# ${customer_data['opportunity_value']:,.0f}")
            st.markdown(f"{customer_data['opportunity_type']} - **{customer_data['expansion_probability']*100:.0f}% close probability** based on current risk factors")
            if st.button("Generate Proposal", type="primary"):
                st.success("Proposal generated and sent to sales team!")
    
    # Main dashboard grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Opportunity Analysis
        st.markdown(f"### 🏦 {customer_data['name']} - Opportunity Analysis")
        
        if customer_data['expansion_probability'] > 0.7:
            st.success(f"High Potential - {customer_data['expansion_probability']*100:.0f}% close probability")
            
            st.markdown("#### Why This Opportunity Will Close:")
            if customer_data['name'] == 'First National Bank of Springfield':
                st.markdown("""
                • CFO expressed audit concerns 3 times in recent communications
                • CECL implementation is only 75% complete with audit in 60 days
                • 2 critical tickets indicate need for expert guidance
                • Similar banks (87%) purchased advisory when facing audit
                • Budget approved for Q4 professional services
                """)
                
                st.info("✨ **Competitive Intelligence:** Community Bank of Texas (similar size/complexity) achieved successful CECL audit after our 6-week advisory engagement. Use as reference.")
    
    with col2:
        # Active Projects
        if customer_data['ps_project_active']:
            st.markdown("### 📊 Current Project Status")
            st.markdown(f"**{customer_data['ps_project_name']}**")
            st.markdown(f"Started: Sep 1, 2024")
            st.progress(customer_data['ps_project_completion'] / 100)
            st.markdown(f"{customer_data['ps_project_completion']:.0f}% Complete - On Track")
            
            st.metric("Hours Consumed", f"{customer_data['ps_hours_used']} / {customer_data['ps_hours_total']}")
            st.metric("Customer Satisfaction", "4.5 / 5")
    
    # Opportunity Pipeline
    st.markdown("### 🎯 AI-Identified Opportunities Across Portfolio")
    
    # Calculate pipeline metrics
    high_prob = customers_df[customers_df['expansion_probability'] > 0.8]
    med_prob = customers_df[(customers_df['expansion_probability'] > 0.5) & (customers_df['expansion_probability'] <= 0.8)]
    low_prob = customers_df[customers_df['expansion_probability'] <= 0.5]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**High Probability (>80%)**")
        total_value = high_prob['opportunity_value'].sum()
        st.metric("Pipeline Value", f"${total_value:,.0f}")
        st.markdown(f"**{len(high_prob)} opportunities**")
    
    with col2:
        st.warning("**Medium Probability (50-80%)**")
        total_value = med_prob['opportunity_value'].sum()
        st.metric("Pipeline Value", f"${total_value:,.0f}")
        st.markdown(f"**{len(med_prob)} opportunities**")
    
    with col3:
        st.info("**Developing (<50%)**")
        total_value = low_prob['opportunity_value'].sum()
        st.metric("Pipeline Value", f"${total_value:,.0f}")
        st.markdown(f"**{len(low_prob)} opportunities**")
    
    # Success Pattern Analysis
    st.markdown("### 📈 Success Pattern Analysis")
    with st.container():
        st.info("✨ **AI Learning: What Drives Advisory Sales**\n\n**Top 5 Indicators (based on 127 successful engagements):**\n1. Upcoming regulatory audit within 90 days (92% close rate)\n2. Multiple CECL-related support tickets (87% close rate)\n3. Executive mentions \"compliance concerns\" (85% close rate)\n4. Data quality score below 80% (83% close rate)\n5. Peer banks in region already engaged us (81% close rate)")
    
    # AI Recommendations
    recommendations = generate_ai_recommendations(customer_data, 'PSO')
    if recommendations:
        st.markdown("### 🎯 AI Recommendations")
        for rec in recommendations:
            if rec['priority'] == 'High':
                st.warning(f"**{rec['action']}** - {rec['reason']}")

def render_credit_view(customer_data, customers_df):
    """Render the Credit Team view"""
    st.header("AI-Powered Credit Bureau Submission Dashboard - Credit Team View")
    st.subheader("Database query optimization and bureau submission management across all 1,008 customers")
    
    # Critical Alert Banner
    if customer_data['bureau_match_rate'] < 70:
        with st.container():
            st.error(f"### 🚨 Bureau Submission Alert: {customer_data['name']}")
            st.markdown(f"**Next submission scheduled in {customer_data['last_bureau_submission']}** with predicted {customer_data['bureau_match_rate']:.0f}% bureau match rate (target: 95%). Database validation recommended to avoid ${customer_data['bureau_cost']*0.3:,.0f} in bureau rejection fees.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Database Validation", type="primary"):
                    st.success("Database validation completed! Issues identified and ready for fixing.")
            with col2:
                if st.button("Fix Data Issues", type="primary"):
                    st.success("Data issues fixed! Bureau match rate improved to 94%")
    
    # Today's Bureau Submissions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Today's Bureau Submissions")
        
        # Show current customer submission
        severity = "High Risk" if customer_data['bureau_match_rate'] < 70 else "Medium Risk" if customer_data['bureau_match_rate'] < 85 else "Ready"
        severity_color = "🔴" if severity == "High Risk" else "🟡" if severity == "Medium Risk" else "🟢"
        
        with st.expander(f"{severity_color} {customer_data['name']} → Experian", expanded=True):
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Predicted bureau match rate", f"{customer_data['bureau_match_rate']:.0f}%")
                st.metric("Database records", f"{customer_data['database_records']:,}")
            with col1b:
                st.metric("Bureau cost", f"${customer_data['bureau_cost']:,}")
                st.metric("Query status", "Ready for extraction")
            
            if customer_data['name'] == 'First National Bank of Springfield':
                st.error(f"⚠️ Issues: {customer_data['data_issues']}")
            
            if severity == "High Risk":
                if st.button("Fix Before Submission", type="secondary"):
                    st.success("Data issues fixed!")
            elif severity == "Ready":
                if st.button("Submit to Bureau", type="primary"):
                    st.success("Submitted to bureau successfully!")
    
    with col2:
        # Performance Metrics
        st.markdown("### 📈 Bureau Submission Metrics")
        st.metric("This Month Match Rate", "89.2%", "-2.5%", delta_color="inverse")
        st.metric("YTD Average", "91.7%")
        st.metric("Bureau Cost per Submission", "$4,250")
        st.metric("Rejections This Month", "23", "+5", delta_color="inverse")
        st.metric("Database Query Time", "4.2 hours")
        st.metric("Records Submitted YTD", "2.1M")
    
    # Database Quality Analysis
    st.markdown("### 🗄️ Database Quality Analysis for Bureau Submissions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.error("**SSN**")
        st.metric("Quality", "76%", "10,889 records need cleanup")
        st.caption("Dash/space format issues")
    
    with col2:
        st.warning("**Names**")
        st.metric("Quality", "88%", "5,432 records affected")
        st.caption("Special chars, case issues")
    
    with col3:
        st.warning("**Address**")
        st.metric("Quality", "82%", "8,123 records affected")
        st.caption("Missing apt, standardization")
    
    with col4:
        st.success("**DOB**")
        st.metric("Quality", "96%", "1,234 records affected")
        st.caption("Date format variations")
    
    with col5:
        st.success("**Phone**")
        st.metric("Quality", "94%", "2,567 records affected")
        st.caption("Extensions, formatting")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Database Cleanup Scripts", use_container_width=True):
            st.success("Database cleanup completed! Quality improved by 15%")
    with col2:
        if st.button("Export Quality Report", use_container_width=True):
            st.success("Quality report exported!")
    
    # AI Optimization Recommendations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤖 AI Database Optimization")
        with st.container():
            st.info("✨ **Pre-Submission Database Improvements**\n\n1. **SSN Standardization:** Auto-format 10,889 SSNs to XXX-XX-XXXX format (+18% bureau acceptance)\n2. **Address USPS Validation:** Standardize 8,123 addresses before bureau submission (+12% improvement)\n3. **Name Field Cleanup:** Remove special characters, standardize case (+8% improvement)\n4. **Database Query Optimization:** Pre-validate fields during extraction (+25% faster processing)")
    
    with col2:
        st.markdown("### 💰 Bureau Cost Savings")
        st.success("**Potential Savings**")
        st.metric("Current monthly rejection cost", "$97,750")
        st.metric("Potential savings", "$68,425", "70% reduction")
        st.metric("Annual savings", "$821,100")
        st.metric("Query time reduction", "2.1 hours", "per submission")
    
    # Portfolio Impact Analysis
    st.markdown("### 🏦 Database Quality by Customer Portfolio")
    
    # Calculate quality distribution
    poor_quality = len(customers_df[customers_df['data_quality_score'] < 85])
    moderate_quality = len(customers_df[(customers_df['data_quality_score'] >= 85) & (customers_df['data_quality_score'] < 92)])
    excellent_quality = len(customers_df[customers_df['data_quality_score'] >= 92])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error(f"**Poor Data Quality (<85% clean)**")
        st.metric("Banks", poor_quality)
        if customer_data['data_quality_score'] < 85:
            st.markdown(f"Including: **{customer_data['name']}** ({customer_data['data_quality_score']:.0f}% clean)")
    
    with col2:
        st.warning(f"**Moderate Quality (85-92% clean)**")
        st.metric("Banks", moderate_quality)
    
    with col3:
        st.success(f"**Excellent Quality (>92% clean)**")
        st.metric("Banks", excellent_quality)
    
    # AI Recommendations
    recommendations = generate_ai_recommendations(customer_data, 'Credit')
    if recommendations:
        st.markdown("### 🎯 AI Recommendations for This Customer")
        for rec in recommendations:
            if rec['priority'] == 'Critical':
                st.error(f"**{rec['action']}** - {rec['reason']}")
            elif rec['priority'] == 'High':
                st.warning(f"**{rec['action']}** - {rec['reason']}")

if __name__ == "__main__":
    main()
