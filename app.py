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
    /* Theme-aware styles that work in both light and dark modes */
    
    /* Metric containers with better theme support */
    [data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background-color: rgba(128, 128, 128, 0.12);
        border-color: rgba(128, 128, 128, 0.25);
    }
    
    /* Remove forced colors - let Streamlit handle theme colors */
    [data-testid="metric-container"] label {
        opacity: 0.7;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-weight: 600;
    }
    
    /* Custom containers with semi-transparent backgrounds */
    .custom-container {
        background: rgba(128, 128, 128, 0.05);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Success/Warning/Error styles using transparency */
    .success-container {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    .warning-container {
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    .error-container {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Remove any forced text colors */
    .stMarkdown, .stText {
        color: inherit;
    }
    
    /* Better button styling for both themes */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Ensure plotly charts work in both themes */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Better expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 0.5rem;
    }
    
    /* Table styling improvements */
    .stDataFrame {
        background-color: rgba(128, 128, 128, 0.02);
    }
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.03);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: rgba(128, 128, 128, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = 'First National Bank of Springfield'
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'CSM'

# Sample data generation with realistic patterns
@st.cache_data
def generate_customer_data():
    """Generate realistic synthetic customer data for 1,008 customers"""
    np.random.seed(42)
    
    # Realistic bank name components based on common patterns
    prefixes = ['First', 'Community', 'Regional', 'Citizens', 'Heritage', 'Liberty', 'Union', 'Pioneer', 
                'Founders', 'Summit', 'Valley', 'Coastal', 'Mountain', 'Prairie', 'River', 'Lake',
                'Central', 'National', 'State', 'Federal', 'American', 'United', 'Trust', 'Mutual']
    
    locations = ['Springfield', 'Riverside', 'Oakville', 'Maplewood', 'Pinehurst', 'Fairview', 
                 'Greenfield', 'Brookside', 'Hillcrest', 'Lakewood', 'Westfield', 'Eastgate',
                 'Northpoint', 'Southbridge', 'Midtown', 'Downtown', 'Uptown', 'Crossroads',
                 'Grandview', 'Pleasant Valley', 'Crystal Lake', 'Stone Mountain', 'Gold Coast',
                 'Silver Springs', 'Copper Hills', 'Iron Ridge', 'Diamond Valley', 'Pearl Harbor']
    
    suffixes = ['Bank', 'Credit Union', 'Financial', 'Trust', 'Savings', 'Bank & Trust', 
                'Federal Credit Union', 'Community Bank', 'Savings Bank', 'Trust Company']
    
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 
              'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI', 'CO', 'MN', 'SC', 'AL',
              'LA', 'KY', 'OR', 'OK', 'CT', 'UT', 'IA', 'NV', 'AR', 'MS', 'KS', 'NM',
              'NE', 'WV', 'ID', 'HI', 'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
    
    # CSM distribution based on real data
    csms = ['Eric Dahn'] * 265 + ['Austin Rayfield'] * 237 + ['Bryan Burton'] * 230 + ['Melissa Stark'] * 218
    remaining = 1008 - len(csms)
    csms.extend(['Other CSM'] * remaining)
    np.random.shuffle(csms)
    
    # Territory mapping
    territory_map = {
        'Eric Dahn': ['CA', 'TX', 'NM', 'AZ', 'OK', 'NV', 'KS', 'MO', 'IA', 'HI'],
        'Bryan Burton': ['MI', 'OH', 'PA', 'NY', 'NJ', 'CT', 'MA', 'VT', 'NH', 'ME', 'RI', 'MD', 'DE'],
        'Austin Rayfield': ['FL', 'GA', 'NC', 'SC', 'TN', 'AL', 'MS', 'LA', 'AR', 'KY', 'VA', 'WV'],
        'Melissa Stark': ['WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'ND', 'SD', 'NE', 'MN', 'WI', 'IL', 'IN']
    }
    
    # Health score distribution (1-5 scale, matching real data)
    # Real distribution shows most are healthy (4-5), some at risk (2-3), few critical (1)
    health_weights = [0.03, 0.05, 0.10, 0.35, 0.47]  # for scores 1-5
    
    customers = []
    used_names = set()
    
    for i in range(1008):
        # Generate unique bank name
        while True:
            if i == 0:
                # Keep our demo customer
                name = 'First National Bank of Springfield'
                state = 'MO'
                break
            else:
                prefix = np.random.choice(prefixes)
                location = np.random.choice(locations)
                suffix = np.random.choice(suffixes)
                name = f"{prefix} {location} {suffix}"
                
                # Ensure uniqueness
                if name not in used_names:
                    used_names.add(name)
                    # Assign state based on CSM territory
                    csm = csms[i]
                    if csm in territory_map:
                        state = np.random.choice(territory_map[csm])
                    else:
                        state = np.random.choice(states)
                    break
        
        # Generate health score with realistic distribution
        health_score = np.random.choice([1, 2, 3, 4, 5], p=health_weights)
        
        # Convert to 0-100 scale for display
        health_score_display = {1: 20, 2: 40, 3: 60, 4: 80, 5: 95}[health_score]
        
        # Realistic correlation - poor health = more tickets, higher churn risk
        if health_score == 1:
            ticket_avg = 150
            churn_risk = 0.75
            expansion_prob = 0.05
        elif health_score == 2:
            ticket_avg = 80
            churn_risk = 0.45
            expansion_prob = 0.15
        elif health_score == 3:
            ticket_avg = 40
            churn_risk = 0.25
            expansion_prob = 0.35
        elif health_score == 4:
            ticket_avg = 20
            churn_risk = 0.10
            expansion_prob = 0.55
        else:  # health_score == 5
            ticket_avg = 8
            churn_risk = 0.02
            expansion_prob = 0.75
        
        # Add realistic variance
        support_tickets = max(0, int(np.random.normal(ticket_avg, ticket_avg * 0.3)))
        
        # Critical tickets correlate with poor health
        critical_tickets = 0
        if health_score <= 2:
            critical_tickets = np.random.binomial(min(5, support_tickets), 0.3)
        elif health_score == 3:
            critical_tickets = np.random.binomial(min(3, support_tickets), 0.1)
        
        # Special case for First National Bank (our demo)
        if i == 0:
            customer = {
                'name': name,
                'state': state,
                'tier': 'Tier 1',
                'csm': 'Bryan Burton',
                'customer_since': 'Jan 2021',
                'health_score': 82,
                'health_score_raw': 4,
                'churn_risk': 0.15,
                'expansion_probability': 0.85,
                'monthly_value': 45000,
                'annual_revenue': 540000,
                'cecl_completion': 75,
                'support_tickets_open': 2,
                'support_tickets_critical': 2,
                'support_tickets_ytd': 47,
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
                'data_issues': 'SSN format (24%), Address missing (18%), Name standardization (12%)',
                'health_tagline': 'Struggling with CECL implementation - needs immediate attention'
            }
        else:
            # Realistic tier distribution
            tier = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], p=[0.15, 0.45, 0.40])
            
            # Revenue based on tier
            if tier == 'Tier 1':
                annual_revenue = np.random.uniform(500000, 2000000)
            elif tier == 'Tier 2':
                annual_revenue = np.random.uniform(100000, 500000)
            else:
                annual_revenue = np.random.uniform(25000, 100000)
            
            # Project likelihood based on health and tier
            ps_project_active = np.random.random() < (0.4 if health_score <= 3 else 0.2)
            
            # Realistic health taglines based on score
            if health_score == 1:
                taglines = [
                    'At risk of cancellation - needs executive intervention',
                    'Severe data quality issues blocking progress',
                    'No engagement in 90+ days - ghosting risk',
                    'Multiple escalations - relationship deteriorating',
                    'Invoice disputes - considering alternatives'
                ]
            elif health_score == 2:
                taglines = [
                    'Struggling with implementation - needs hands-on support',
                    'High ticket volume indicating platform challenges',
                    'Engagement declining - at risk',
                    'Data quality issues impacting credit pulls',
                    'Project delays causing frustration'
                ]
            elif health_score == 3:
                taglines = [
                    'Making progress but needs guidance',
                    'Stable but not fully utilizing platform',
                    'Some technical challenges to address',
                    'Moderate engagement - room for growth',
                    'Project on track with minor issues'
                ]
            elif health_score == 4:
                taglines = [
                    'Strong adoption with expansion potential',
                    'Good platform usage - ready for advanced features',
                    'Stable and growing relationship',
                    'Minor support needs only',
                    'Project successful - exploring next phase'
                ]
            else:  # health_score == 5
                taglines = [
                    'Power user - potential reference customer',
                    'Excellent adoption across all modules',
                    'Strategic partner - innovation opportunities',
                    'Minimal support needs - self-sufficient',
                    'Strong advocate - bringing referrals'
                ]
            
            customer = {
                'name': name,
                'state': state,
                'tier': tier,
                'csm': csms[i],
                'customer_since': f"{np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {np.random.randint(2019, 2024)}",
                'health_score': health_score_display,
                'health_score_raw': health_score,
                'churn_risk': churn_risk + np.random.uniform(-0.1, 0.1),
                'expansion_probability': expansion_prob + np.random.uniform(-0.1, 0.1),
                'monthly_value': annual_revenue / 12,
                'annual_revenue': annual_revenue,
                'cecl_completion': np.random.uniform(0, 100) if health_score < 4 else np.random.uniform(60, 100),
                'support_tickets_open': np.random.binomial(5, 0.3) if support_tickets > 0 else 0,
                'support_tickets_critical': critical_tickets,
                'support_tickets_ytd': support_tickets,
                'days_since_contact': np.random.exponential(15) if health_score >= 3 else np.random.exponential(30),
                'usage_trend': np.random.normal(0, 20) if health_score >= 3 else np.random.normal(-10, 20),
                'data_quality_score': np.random.uniform(60, 98) if health_score >= 3 else np.random.uniform(40, 80),
                'bureau_match_rate': np.random.uniform(85, 98) if health_score >= 4 else np.random.uniform(60, 85),
                'last_bureau_submission': f"{np.random.randint(1, 30)} days",
                'missing_fields': np.random.randint(0, 50) if health_score >= 4 else np.random.randint(50, 200),
                'cecl_model_status': f"{np.random.randint(8, 15)}/15 Configured" if health_score >= 3 else f"{np.random.randint(0, 8)}/15 Configured",
                'ps_project_active': ps_project_active,
                'ps_project_name': np.random.choice(['CECL Advisory', 'Data Quality Remediation', 'Implementation Services', 'Optimization Engagement']),
                'ps_project_completion': np.random.uniform(0, 100) if ps_project_active else 0,
                'ps_hours_used': np.random.randint(0, 200) if ps_project_active else 0,
                'ps_hours_total': np.random.randint(100, 300) if ps_project_active else 0,
                'opportunity_value': np.random.choice([0, 50000, 75000, 100000, 125000, 150000, 200000], 
                                                    p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]) if expansion_prob > 0.5 else 0,
                'opportunity_type': np.random.choice(['CECL Advisory', 'Data Quality Package', 'Bureau Optimization', 'Advanced Analytics']),
                'database_records': np.random.randint(10000, 100000),
                'bureau_cost': np.random.randint(1000, 6000),
                'data_issues': 'Various data quality issues',
                'health_tagline': np.random.choice(taglines)
            }
        
        # Normalize values
        customer['churn_risk'] = max(0, min(1, customer['churn_risk']))
        customer['expansion_probability'] = max(0, min(1, customer['expansion_probability']))
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

@st.cache_data
def generate_support_tickets(customer_name):
    """Generate realistic support tickets for a specific customer"""
    tickets = []
    
    # Get customer data to generate appropriate tickets
    customer = customers_df[customers_df['name'] == customer_name].iloc[0] if customer_name in customers_df['name'].values else None
    
    if customer_name == 'First National Bank of Springfield':
        tickets = [
            {
                'id': 'INC-2024-18234',
                'title': 'CECL calculation discrepancy in PD models',
                'status': 'Open',
                'priority': 'Critical',
                'days_open': 5,
                'assigned_to': 'Bob Smith (DSE)',
                'root_cause': 'Incorrect date format in loan origination data',
                'category': 'CECL/ALLL'
            },
            {
                'id': 'INC-2024-18471',
                'title': 'Data mapping error in commercial loan portfolio',
                'status': 'Open',
                'priority': 'Critical',
                'days_open': 2,
                'assigned_to': 'Bob Smith (DSE)',
                'root_cause': 'Impact: 2,341 loans incorrectly categorized',
                'category': 'Data Quality'
            }
        ]
    elif customer is not None:
        # Generate tickets based on customer health
        health_score = customer['health_score_raw']
        
        # Realistic ticket categories based on real data
        ticket_categories = {
            'File processing': 0.327,
            'CECL/ALLL': 0.184,
            'Data Quality': 0.156,
            'Credit Reports': 0.098,
            'User Access': 0.067,
            'Configuration': 0.054,
            'Training Request': 0.042,
            'Integration': 0.038,
            'Performance': 0.034
        }
        
        # More tickets for unhealthy customers
        if health_score == 1:
            num_open_tickets = np.random.poisson(3)
            num_recent_tickets = np.random.poisson(8)
        elif health_score == 2:
            num_open_tickets = np.random.poisson(2)
            num_recent_tickets = np.random.poisson(5)
        elif health_score == 3:
            num_open_tickets = np.random.poisson(1)
            num_recent_tickets = np.random.poisson(3)
        else:
            num_open_tickets = np.random.binomial(2, 0.2)
            num_recent_tickets = np.random.poisson(1)
        
        # Generate realistic ticket titles by category
        ticket_titles = {
            'File processing': [
                'Credit file upload failed - invalid format',
                'Batch processing stuck at validation',
                'Large file timeout during processing',
                'CSV parsing error on row 10,234',
                'File rejection - missing required columns'
            ],
            'CECL/ALLL': [
                'Q-factor calculation not matching expected results',
                'PD model convergence issues',
                'Scenario analysis configuration questions',
                'Historical loss data import problems',
                'Forecast period settings clarification needed'
            ],
            'Data Quality': [
                'SSN format validation rejecting valid entries',
                'Address standardization not working correctly',
                'Duplicate detection flagging false positives',
                'Missing data in required fields report',
                'Data mapping configuration lost after update'
            ],
            'Credit Reports': [
                'Bureau match rate below 80%',
                'Credit pull timeout errors',
                'Tradeline data not populating',
                'Score discrepancies between bureaus',
                'Bulk credit pull scheduling issues'
            ],
            'User Access': [
                'New user cannot access CECL module',
                'Password reset not working',
                'Permission error when running reports',
                'Admin access needed for configuration',
                'User deactivation request'
            ]
        }
        
        # Generate open tickets
        for i in range(num_open_tickets):
            category = np.random.choice(list(ticket_categories.keys()), p=list(ticket_categories.values()))
            priority = 'Critical' if health_score <= 2 and i == 0 else np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.3, 0.4, 0.2, 0.1])
            
            tickets.append({
                'id': f'INC-2024-{np.random.randint(10000, 99999)}',
                'title': np.random.choice(ticket_titles.get(category, ['General platform inquiry'])),
                'status': np.random.choice(['Open', 'In Progress'], p=[0.6, 0.4]),
                'priority': priority,
                'days_open': np.random.exponential(3),
                'assigned_to': np.random.choice(['Bob Smith (DSE)', 'Alice Johnson (DSE)', 'Tom Wilson (DSE)', 'Sarah Martinez (DSE)']),
                'root_cause': 'Under investigation',
                'category': category
            })
        
        # Add some recently closed tickets for context
        for i in range(num_recent_tickets):
            category = np.random.choice(list(ticket_categories.keys()), p=list(ticket_categories.values()))
            tickets.append({
                'id': f'INC-2024-{np.random.randint(10000, 99999)}',
                'title': np.random.choice(ticket_titles.get(category, ['General platform inquiry'])),
                'status': 'Resolved',
                'priority': np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2]),
                'days_open': 0,
                'assigned_to': np.random.choice(['Bob Smith (DSE)', 'Alice Johnson (DSE)', 'Tom Wilson (DSE)', 'Sarah Martinez (DSE)']),
                'root_cause': 'Resolved',
                'category': category
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
        title = {'text': "Health Score"},
        gauge = {
            'axis': {'range': [None, 100]},
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
        }
    ))
    
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Main App
def main():
    # Load data - make it globally accessible
    global customers_df
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
            ["All Customers", "Critical (Score 1)", "At Risk (Score 2-3)", "Healthy (Score 4-5)", "My Accounts", "High Ticket Volume"]
        )
        
        if filter_type == "Critical (Score 1)":
            filtered_df = customers_df[customers_df['health_score_raw'] == 1]
        elif filter_type == "At Risk (Score 2-3)":
            filtered_df = customers_df[customers_df['health_score_raw'].isin([2, 3])]
        elif filter_type == "Healthy (Score 4-5)":
            filtered_df = customers_df[customers_df['health_score_raw'].isin([4, 5])]
        elif filter_type == "My Accounts":
            # Filter by current view mode CSM
            if st.session_state.view_mode == 'CSM':
                filtered_df = customers_df[customers_df['csm'] == 'Bryan Burton']
            else:
                filtered_df = customers_df
        elif filter_type == "High Ticket Volume":
            filtered_df = customers_df[customers_df['support_tickets_ytd'] > 100]
        else:
            filtered_df = customers_df
        
        # Sort by health score (worst first) for easier selection
        filtered_df = filtered_df.sort_values('health_score_raw', ascending=True)
        
        selected_customer = st.selectbox(
            f"Select Customer ({len(filtered_df)} matches)",
            filtered_df['name'].tolist(),
            index=0 if len(filtered_df) > 0 else 0,
            format_func=lambda x: f"{x} ({filtered_df[filtered_df['name']==x]['state'].iloc[0]})"
        )
        
        st.session_state.selected_customer = selected_customer
        
        # Quick stats
        st.markdown("---")
        st.subheader("Portfolio Overview")
        
        # Calculate real statistics
        critical = len(customers_df[customers_df['health_score_raw'] == 1])
        at_risk = len(customers_df[customers_df['health_score_raw'].isin([2, 3])])
        healthy = len(customers_df[customers_df['health_score_raw'].isin([4, 5])])
        high_tickets = len(customers_df[customers_df['support_tickets_ytd'] > 100])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Critical", critical, f"{critical/10.08:.1f}%", delta_color="inverse")
            st.metric("High Tickets", high_tickets, ">100 YTD", delta_color="inverse")
        with col2:
            st.metric("At Risk", at_risk, f"{at_risk/10.08:.1f}%", delta_color="inverse")
            st.metric("Healthy", healthy, f"{healthy/10.08:.1f}%")
        
        # CSM breakdown
        st.markdown("---")
        st.subheader("CSM Breakdown")
        csm_counts = customers_df['csm'].value_counts()
        for csm in csm_counts.head(4).index:  # Show top 4 CSMs
            count = csm_counts[csm]
            health_avg = customers_df[customers_df['csm'] == csm]['health_score'].mean()
            st.markdown(f"**{csm}**: {count} accounts (avg: {health_avg:.0f})")
    
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
    st.subheader(f"Managing {len(customers_df[customers_df['csm'] == customer_data['csm']])} customers in {customer_data['state']}")
    
    # Customer header with health score
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### {customer_data['name']}")
        st.markdown(f"📍 **{customer_data['tier']} Customer** | 🌎 **{customer_data['state']}** | 📅 **Since:** {customer_data['customer_since']} | 💰 **ARR:** ${customer_data['annual_revenue']:,.0f}")
        if customer_data.get('health_tagline'):
            if customer_data['health_score_raw'] <= 2:
                st.error(f"⚠️ {customer_data['health_tagline']}")
            elif customer_data['health_score_raw'] == 3:
                st.warning(f"📊 {customer_data['health_tagline']}")
            else:
                st.success(f"✅ {customer_data['health_tagline']}")
    
    with col2:
        fig = render_health_score_gauge(customer_data['health_score'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.metric("Churn Risk", f"{customer_data['churn_risk']*100:.0f}%", 
                  "High" if customer_data['churn_risk'] > 0.3 else "Low", 
                  delta_color="inverse")
        st.metric("Support Tickets YTD", customer_data['support_tickets_ytd'],
                  "High Volume" if customer_data['support_tickets_ytd'] > 100 else "Normal")
    
    with col4:
        st.metric("Expansion Prob.", f"{customer_data['expansion_probability']*100:.0f}%",
                  "High" if customer_data['expansion_probability'] > 0.7 else "Low")
        st.metric("Days Since Contact", f"{customer_data['days_since_contact']:.0f}",
                  "Overdue" if customer_data['days_since_contact'] > 30 else "Recent")
    
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
