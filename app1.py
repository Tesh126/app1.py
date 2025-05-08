import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Insurance Agent Performance Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2563EB;
    }
    .card {
        border-radius: 5px;
        background-color: #f0f2f6;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
    }
    .high-performer {
        color: #059669;
        font-weight: bold;
    }
    .medium-performer {
        color: #D97706;
        font-weight: bold;
    }
    .low-performer {
        color: #DC2626;
        font-weight: bold;
    }
    .improving {
        color: #10B981;
    }
    .declining {
        color: #EF4444;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 10px 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Insurance Agent Performance Tracker & Advisor</h1>", unsafe_allow_html=True)

with st.expander("About this app", expanded=False):
    st.markdown("""
    This application helps insurance agents and managers track performance metrics, view historical trends, 
    and receive personalized advice based on performance categories. The system uses advanced analytics 
    to classify agents into performance tiers and provide targeted interventions for improvement.
    
    **Features:**
    - Agent performance dashboard with key metrics
    - Historical trend analysis
    - Performance categorization (High/Medium/Low)
    - Personalized recommendations based on performance category
    - Progress tracking over time
    - Intervention strategies tailored to specific needs
    """)

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)

# Function to load data - in production, this would connect to your database
@st.cache_data
def load_data():
    # For demo purposes, we'll generate some sample data based on the schema mentioned
    # In production, you'd load your actual CSV or database connection
    
    # Sample data with 20 agents over 12 months
    np.random.seed(42)
    data = []
    
    agent_codes = [f"AG{i:03d}" for i in range(1, 21)]
    months = pd.date_range(start='2023-01-01', end='2024-01-01', freq='MS')
    
    for agent_code in agent_codes:
        # Fixed attributes for each agent
        agent_age = np.random.randint(25, 55)
        join_date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        agent_join_month = join_date.strftime('%Y-%m')
        
        # First policy sold is after joining but within 3 months
        days_to_first_policy = np.random.randint(1, 90)
        first_policy_date = join_date + pd.Timedelta(days=days_to_first_policy)
        first_policy_sold_month = first_policy_date.strftime('%Y-%m')
        
        # Cumulative metrics that grow over time
        cumulative_policy_holders = 0
        cumulative_cash_payment_policies = 0
        
        # Agent baseline performance (different for each agent)
        performance_base = np.random.normal(1, 0.3)
        
        for month in months:
            # Skip months before agent joined
            if month < join_date:
                continue
                
            month_str = month.strftime('%Y-%m')
            
            # Seasonality factor (Q4 is usually better for insurance)
            seasonality = 1.0
            if month.month in [10, 11, 12]:
                seasonality = 1.2
            
            # Experience factor (agents improve over time)
            months_experience = (month - join_date).days / 30
            experience_factor = min(1 + (months_experience * 0.01), 1.5)
            
            # Random monthly fluctuation
            monthly_factor = np.random.normal(1, 0.15)
            
            # Calculate base performance for this month
            month_performance = performance_base * seasonality * experience_factor * monthly_factor
            
            # Generate metrics based on performance
            unique_proposals_last_7_days = int(np.random.poisson(5 * month_performance))
            unique_proposals_last_15_days = int(np.random.poisson(4 * month_performance))
            unique_proposals_last_21_days = int(np.random.poisson(3 * month_performance))
            unique_proposal = unique_proposals_last_7_days + unique_proposals_last_15_days + unique_proposals_last_21_days
            
            quotation_rate = 0.7 + (np.random.random() * 0.2)  # 70-90% of proposals become quotations
            unique_quotations_last_7_days = int(unique_proposals_last_7_days * quotation_rate)
            unique_quotations_last_15_days = int(unique_proposals_last_15_days * quotation_rate)
            unique_quotations_last_21_days = int(unique_proposals_last_21_days * quotation_rate)
            unique_quotations = unique_quotations_last_7_days + unique_quotations_last_15_days + unique_quotations_last_21_days
            
            customer_ratio = 0.8 + (np.random.random() * 0.2)  # Each customer may have multiple proposals
            unique_customers_last_7_days = int(unique_proposals_last_7_days * customer_ratio)
            unique_customers_last_15_days = int(unique_proposals_last_15_days * customer_ratio)
            unique_customers_last_21_days = int(unique_proposals_last_21_days * customer_ratio)
            unique_customers = unique_customers_last_7_days + unique_customers_last_15_days + unique_customers_last_21_days
            
            policy_conversion = 0.3 + (np.random.random() * 0.3)  # 30-60% of quotations become policies
            new_policy_count = int(unique_quotations * policy_conversion)
            
            # ANBP value depends on policy count and agent skill
            avg_policy_value = np.random.normal(1000, 200) * month_performance
            ANBP_value = new_policy_count * avg_policy_value
            
            # Net income is a percentage of ANBP
            commission_rate = 0.1 + (np.random.random() * 0.05)  # 10-15% commission
            net_income = ANBP_value * commission_rate
            
            # Update cumulative metrics
            cumulative_policy_holders += new_policy_count
            new_cash_policies = int(new_policy_count * (0.3 + np.random.random() * 0.2))  # 30-50% cash payment
            cumulative_cash_payment_policies += new_cash_policies
            
            # Calculate ratios
            proposal_to_quotation_ratio = unique_quotations / unique_proposal if unique_proposal > 0 else 0
            quotation_to_policy_ratio = new_policy_count / unique_quotations if unique_quotations > 0 else 0
            proposal_to_policy_ratio = new_policy_count / unique_proposal if unique_proposal > 0 else 0
            customer_to_policy_ratio = new_policy_count / unique_customers if unique_customers > 0 else 0
            avg_ANBP_per_policy = ANBP_value / new_policy_count if new_policy_count > 0 else 0
            
            # Add row to dataset
            data.append({
                'agent_code': agent_code,
                'agent_age': agent_age,
                'agent_join_month': agent_join_month,
                'first_policy_sold_month': first_policy_sold_month,
                'year_month': month_str,
                'unique_proposals_last_7_days': unique_proposals_last_7_days,
                'unique_proposals_last_15_days': unique_proposals_last_15_days,
                'unique_proposals_last_21_days': unique_proposals_last_21_days,
                'unique_proposal': unique_proposal,
                'unique_quotations_last_7_days': unique_quotations_last_7_days,
                'unique_quotations_last_15_days': unique_quotations_last_15_days,
                'unique_quotations_last_21_days': unique_quotations_last_21_days,
                'unique_quotations': unique_quotations,
                'unique_customers_last_7_days': unique_customers_last_7_days,
                'unique_customers_last_15_days': unique_customers_last_15_days,
                'unique_customers_last_21_days': unique_customers_last_21_days,
                'unique_customers': unique_customers,
                'new_policy_count': new_policy_count,
                'ANBP_value': ANBP_value,
                'net_income': net_income,
                'number_of_policy_holders': cumulative_policy_holders,
                'number_of_cash_payment_policies': cumulative_cash_payment_policies,
                'proposal_to_quotation_ratio': proposal_to_quotation_ratio,
                'quotation_to_policy_ratio': quotation_to_policy_ratio,
                'proposal_to_policy_ratio': proposal_to_policy_ratio,
                'customer_to_policy_ratio': customer_to_policy_ratio,
                'avg_ANBP_per_policy': avg_ANBP_per_policy,
                'tenure_months': (month - pd.Timestamp(agent_join_month)).days / 30
            })
    
    df = pd.DataFrame(data)
    
    # Convert year_month to datetime for easier sorting/filtering
    df['year_month_dt'] = pd.to_datetime(df['year_month'])
    
    # Sort data
    df = df.sort_values(['agent_code', 'year_month_dt'])
    
    # Calculate month-over-month changes
    df = calculate_mom_changes(df)
    
    return df

def calculate_mom_changes(df):
    """Calculate month-over-month changes for key metrics"""
    # Group by agent and sort by date
    df_sorted = df.sort_values(['agent_code', 'year_month_dt'])
    
    # Calculate absolute changes
    for metric in ['new_policy_count', 'ANBP_value', 'net_income']:
        df_sorted[f'{metric}_mom_change'] = df_sorted.groupby('agent_code')[metric].diff()
    
    # Calculate percentage changes
    for metric in ['new_policy_count', 'ANBP_value', 'net_income']:
        df_sorted[f'{metric}_mom_pct_change'] = df_sorted.groupby('agent_code')[metric].pct_change() * 100
    
    return df_sorted

def categorize_agents(df, selected_month):
    """Categorize agents based on performance metrics"""
    # Filter to the selected month
    latest_month_per_agent = df[df['year_month'] == selected_month].copy()
    
    # Define core performance metrics with weights
    performance_metrics = {
        'new_policy_count': 0.25,
        'ANBP_value': 0.25,
        'net_income': 0.20,
        'proposal_to_policy_ratio': 0.15,
        'customer_to_policy_ratio': 0.15
    }
    
    # Verify all metrics are available
    available_performance_metrics = {}
    for metric, weight in performance_metrics.items():
        if metric in latest_month_per_agent.columns:
            non_nan_pct = (1 - latest_month_per_agent[metric].isna().mean()) * 100
            if non_nan_pct >= 50:  # Require at least 50% of values to be non-NaN
                available_performance_metrics[metric] = weight
    
    # Normalize weights to sum to 1
    if available_performance_metrics:
        weight_sum = sum(available_performance_metrics.values())
        available_performance_metrics = {k: v/weight_sum for k, v in available_performance_metrics.items()}
    
    # Calculate performance scores if we have metrics
    if available_performance_metrics:
        # Fill any NaN values with median for performance calculation
        for metric in available_performance_metrics:
            median_value = latest_month_per_agent[metric].median()
            latest_month_per_agent[metric] = latest_month_per_agent[metric].fillna(median_value)
        
        # Scale the metrics to be comparable
        scaler = StandardScaler()
        scaled_metrics = scaler.fit_transform(latest_month_per_agent[list(available_performance_metrics.keys())])
        scaled_df = pd.DataFrame(
            scaled_metrics, 
            columns=list(available_performance_metrics.keys()),
            index=latest_month_per_agent.index
        )
        
        # Calculate weighted performance score
        latest_month_per_agent['performance_score'] = 0
        for metric, weight in available_performance_metrics.items():
            latest_month_per_agent['performance_score'] += scaled_df[metric] * weight
    else:
        latest_month_per_agent['performance_score'] = 0
    
    # Improvement score calculation
    improvement_metrics = [
        'new_policy_count_mom_pct_change',
        'ANBP_value_mom_pct_change',
        'net_income_mom_pct_change'
    ]
    
    # Add absolute change metrics
    absolute_change_metrics = [
        'new_policy_count_mom_change',
        'ANBP_value_mom_change',
        'net_income_mom_change'
    ]
    
    # Check availability of improvement metrics
    available_improvement_metrics = []
    available_absolute_metrics = []
    
    for metric in improvement_metrics:
        if metric in latest_month_per_agent.columns:
            non_nan_pct = (1 - latest_month_per_agent[metric].isna().mean()) * 100
            if non_nan_pct >= 30:  # Lower threshold for derived metrics
                available_improvement_metrics.append(metric)
                
    for metric in absolute_change_metrics:
        if metric in latest_month_per_agent.columns:
            non_nan_pct = (1 - latest_month_per_agent[metric].isna().mean()) * 100
            if non_nan_pct >= 30:
                available_absolute_metrics.append(metric)
    
    # Calculate improvement score
    if available_improvement_metrics or available_absolute_metrics:
        # Fill remaining NaN values with 0 (no change)
        for metric in available_improvement_metrics + available_absolute_metrics:
            latest_month_per_agent[metric] = latest_month_per_agent[metric].fillna(0)
        
        # Standardize the metrics
        if len(available_improvement_metrics + available_absolute_metrics) > 0:
            # Create a new scaler for improvement metrics
            imp_scaler = StandardScaler()
            improvement_data = imp_scaler.fit_transform(
                latest_month_per_agent[available_improvement_metrics + available_absolute_metrics]
            )
            improvement_df = pd.DataFrame(
                improvement_data,
                columns=available_improvement_metrics + available_absolute_metrics,
                index=latest_month_per_agent.index
            )
            
            # Calculate improvement score with weighted components
            latest_month_per_agent['improvement_score'] = 0
            
            # Add percentage change components (with higher weights)
            pct_weight = 0.7 / max(len(available_improvement_metrics), 1) if available_improvement_metrics else 0
            for metric in available_improvement_metrics:
                latest_month_per_agent['improvement_score'] += improvement_df[metric] * pct_weight
                
            # Add absolute change components (with lower weights)
            abs_weight = 0.3 / max(len(available_absolute_metrics), 1) if available_absolute_metrics else 0
            for metric in available_absolute_metrics:
                latest_month_per_agent['improvement_score'] += improvement_df[metric] * abs_weight
        else:
            latest_month_per_agent['improvement_score'] = 0
    else:
        latest_month_per_agent['improvement_score'] = 0
    
    # Agent categorization using K-means
    if latest_month_per_agent['performance_score'].nunique() > 1:
        # Reshape for K-means
        perf_scores = latest_month_per_agent[['performance_score']].values
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        latest_month_per_agent['cluster'] = kmeans.fit_predict(perf_scores)
    
        # Map clusters to performance categories based on cluster centers
        cluster_centers = kmeans.cluster_centers_.flatten()
        cluster_performance_map = {
            np.argmax(cluster_centers): 'High',
            np.argsort(cluster_centers)[1]: 'Medium',
            np.argmin(cluster_centers): 'Low'
        }
        latest_month_per_agent['performance_category'] = latest_month_per_agent['cluster'].map(cluster_performance_map)
    else:
        latest_month_per_agent['performance_category'] = 'Medium'
    
    # Identify most improved and declining agents
    if 'improvement_score' in latest_month_per_agent.columns and latest_month_per_agent['improvement_score'].nunique() > 1:
        # Top 10% are most improved
        improvement_threshold = np.percentile(latest_month_per_agent['improvement_score'], 90)
        latest_month_per_agent['most_improved'] = latest_month_per_agent['improvement_score'] > improvement_threshold
    
        # Bottom 10% are declining
        decline_threshold = np.percentile(latest_month_per_agent['improvement_score'], 10)
        latest_month_per_agent['declining'] = latest_month_per_agent['improvement_score'] < decline_threshold
    else:
        latest_month_per_agent['most_improved'] = False
        latest_month_per_agent['declining'] = False
    
    return latest_month_per_agent

def get_personalized_advice(agent_data):
    """Generate personalized advice based on agent's performance category and metrics"""
    
    performance_category = agent_data['performance_category']
    is_improving = agent_data['most_improved']
    is_declining = agent_data['declining']
    
    # Base advice structure
    advice = {
        'summary': '',
        'strengths': [],
        'areas_to_improve': [],
        'action_items': [],
        'resources': []
    }
    
    # Performance category specific advice
    if performance_category == 'High':
        advice['summary'] = "You're among our top performers! Your excellent results demonstrate your strong skills in customer acquisition and policy conversion."
        
        # Add strengths
        if agent_data['new_policy_count'] > 0:
            advice['strengths'].append(f"Strong sales performance with {agent_data['new_policy_count']} new policies this month")
        if agent_data['ANBP_value'] > 0:
            advice['strengths'].append(f"Excellent premium generation totaling ${agent_data['ANBP_value']:,.2f}")
        if agent_data['proposal_to_policy_ratio'] > 0.4:
            advice['strengths'].append(f"High proposal-to-policy conversion rate of {agent_data['proposal_to_policy_ratio']:.1%}")
        if agent_data['customer_to_policy_ratio'] > 0.8:
            advice['strengths'].append("Strong customer relationship management skills")
        
        # Areas to improve and action items depend on specific metrics
        if agent_data['avg_ANBP_per_policy'] < agent_data['avg_ANBP_per_policy_median']:
            advice['areas_to_improve'].append("Policy value optimization")
            advice['action_items'].append("Focus on upselling premium plans to increase your average policy value")
            advice['resources'].append("Premium Product Training Module")
        
        if agent_data['quotation_to_policy_ratio'] < 0.5:
            advice['areas_to_improve'].append("Closing sales")
            advice['action_items'].append("Review your closing techniques to convert more quotations into policies")
            advice['resources'].append("Sales Closing Masterclass")
        
        advice['action_items'].append("Mentor a lower-performing agent to share your successful strategies")
        advice['resources'].append("Leadership Development Workshop")
        
    elif performance_category == 'Medium':
        advice['summary'] = "You're performing at a steady level with good potential for growth. With targeted improvements, you could move into our top performer category."
        
        # Add strengths
        if agent_data['proposal_to_quotation_ratio'] > 0.7:
            advice['strengths'].append("Good proposal-to-quotation conversion")
        if agent_data['tenure_months'] < 6 and agent_data['new_policy_count'] > 3:
            advice['strengths'].append("Promising early performance for a new agent")
        if agent_data['most_improved']:
            advice['strengths'].append("Showing solid improvement month-over-month")
        
        # Areas to improve
        if agent_data['new_policy_count'] < agent_data['new_policy_count_median']:
            advice['areas_to_improve'].append("Sales volume")
            advice['action_items'].append("Increase your customer outreach by 20% to generate more leads")
            advice['resources'].append("Lead Generation Strategies Guide")
        
        if agent_data['proposal_to_policy_ratio'] < 0.3:
            advice['areas_to_improve'].append("Sales conversion")
            advice['action_items'].append("Practice objection handling to improve your proposal-to-policy conversion")
            advice['resources'].append("Objection Handling Workshop")
        
        if agent_data['avg_ANBP_per_policy'] < agent_data['avg_ANBP_per_policy_median']:
            advice['areas_to_improve'].append("Policy value")
            advice['action_items'].append("Focus on highlighting premium benefits to increase your average policy value")
            advice['resources'].append("Value Selling Training")
        
        advice['action_items'].append("Set a goal to increase your performance score by 10% next month")
        advice['resources'].append("Performance Goal Setting Workshop")
        
    else:  # Low performer
        advice['summary'] = "You're currently below target performance levels. With the right focus and support, you can significantly improve your results."
        
        # Find any strengths to build confidence
        if agent_data['quotation_to_policy_ratio'] > 0.2:
            advice['strengths'].append("You show potential in converting quotations to policies")
        if agent_data['most_improved']:
            advice['strengths'].append("You're showing improvement from last month - keep up the momentum!")
        if agent_data['unique_customers'] > agent_data['unique_customers_median']:
            advice['strengths'].append("Good customer outreach numbers")
        
        # Key areas to improve
        advice['areas_to_improve'].append("Overall sales performance")
        advice['areas_to_improve'].append("Lead generation and conversion")
        
        if agent_data['new_policy_count'] < 3:
            advice['action_items'].append("Set a minimum target of 3 new policies next month")
        
        if agent_data['proposal_to_quotation_ratio'] < 0.5:
            advice['action_items'].append("Improve your proposal quality to increase quotation rates")
            advice['resources'].append("Proposal Writing Masterclass")
        
        if agent_data['quotation_to_policy_ratio'] < 0.3:
            advice['action_items'].append("Practice closing techniques to convert more quotations to policies")
            advice['resources'].append("Sales Closing Bootcamp")
        
        advice['action_items'].append("Schedule weekly check-ins with your manager for personalized coaching")
        advice['resources'].append("One-on-One Coaching Program")
        advice['resources'].append("Core Sales Skills Training Module")
    
    # Add improvement or decline specific advice
    if is_improving:
        advice['summary'] += " You've shown great improvement recently - whatever you're doing is working!"
        advice['action_items'].append("Document what changes you've made that led to your recent improvement")
    elif is_declining:
        advice['summary'] += " Your recent metrics show a decline that needs attention."
        advice['action_items'].append("Schedule a meeting with your manager to discuss challenges you're facing")
        advice['resources'].append("Performance Recovery Plan Template")
    
    # If no strengths found, add a generic one
    if not advice['strengths']:
        advice['strengths'].append("Commitment to the job and potential for growth")
    
    return advice

def create_agent_history_plot(df, agent_code, metrics):
    """Create a historical trend plot for an agent's key metrics"""
    agent_data = df[df['agent_code'] == agent_code].sort_values('year_month_dt')
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=agent_data['year_month_dt'], 
            y=agent_data[metric],
            mode='lines+markers',
            name=metric
        ))
    
    fig.update_layout(
        title=f'Historical Trends - Agent {agent_code}',
        xaxis_title='Month',
        yaxis_title='Value',
        legend_title='Metrics',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def create_comparison_plot(df, latest_month, agent_code, metric):
    """Create a comparison plot of the agent vs peers in their performance category"""
    
    # Get latest month data with categories
    latest_data = df[df['year_month'] == latest_month].copy()
    
    # Get the agent's performance category
    agent_category = latest_data[latest_data['agent_code'] == agent_code]['performance_category'].iloc[0]
    
    # Filter peers (same category)
    peers = latest_data[latest_data['performance_category'] == agent_category]
    
    # Create a color map
    colors = ['lightgrey'] * len(peers)
    agent_idx = peers[peers['agent_code'] == agent_code].index[0]
    colors[peers.index.get_loc(agent_idx)] = '#3B82F6'  # Highlight the agent
    
    # Sort by the metric
    peers = peers.sort_values(metric, ascending=False)
    
    # Create the comparison plot
    fig = px.bar(
        peers,
        x='agent_code',
        y=metric,
        title=f'Comparison with {agent_category} Performers - {metric}',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        xaxis_title='Agent',
        yaxis_title=metric,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Add a line for the average
    avg_value = peers[metric].mean()
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=avg_value,
        x1=len(peers)-0.5,
        y1=avg_value,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=len(peers)/2,
        y=avg_value*1.05,
        text=f"Category Average: {avg_value:.2f}",
        showarrow=False,
        font=dict(color="red")
    )
    
    return fig

def create_radar_chart(agent_data, metrics_dict):
    """Create a radar chart showing agent's strengths and weaknesses"""
    
    categories = list(metrics_dict.keys())
    values = []
    
    for metric, display_name in metrics_dict.items():
        # Normalize the value between 0 and 1 based on min/max in the dataset
        if metric in agent_data and f"{metric}_min" in agent_data and f"{metric}_max" in agent_data:
            min_val = agent_data[f"{metric}_min"]
            max_val = agent_data[f"{metric}_max"]
            
            if max_val > min_val:  # Avoid division by zero
                value = (agent_data[metric] - min_val) / (max_val - min_val)
                values.append(max(0, min(1, value)))  # Clamp between 0 and 1
            else:
                values.append(0.5)  # Default to middle if no range
        else:
            values.append(0)  # Default if data not available
    
    # Close the loop for the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Performance Strengths & Weaknesses",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    
    return fig

# Load data
df = load_data()

# Get unique months for selection
months = sorted(df['year_month'].unique())
latest_month = months[-1]

# Sidebar - Agent selection
st.sidebar.markdown("### Select Agent")
agent_codes = sorted(df['agent_code'].unique())
selected_agent = st.sidebar.selectbox("Choose Agent", agent_codes)

# Month selection
st.sidebar.markdown("### Select Month")
selected_month = st.sidebar.selectbox("Choose Month", months, index=len(months)-1)

# Process data based on selection
all_agents_current_month = categorize_agents(df, selected_month)

# Get data for the selected agent
agent_data = all_agents_current_month[all_agents_current_month['agent_code'] == selected_agent].iloc[0].copy()

# Add percentile information and median values for comparison
for metric in ['new_policy_count', 'ANBP_value', 'net_income', 'proposal_to_policy_ratio', 'avg_ANBP_per_policy', 'unique_customers']:
    if metric in all_agents_current_month.columns:
        agent_data[f"{metric}_min"] = all_agents_current_month[metric].min()
        agent_data[f"{metric}_max"] = all_agents_current_month[metric].max()
        agent_data[f"{metric}_median"] = all_agents_current_month[metric].median()
        agent_data[f"{metric}_percentile"] = stats = 100 * (all_agents_current_month[metric] <= agent_data[metric]).mean()

# Sidebar - Additional resources
with st.sidebar.expander("Training Resources"):
    st.markdown("""
    - [Sales Skills Training](https://example.com)
    - [Product Knowledge Base](https://example.com)
    - [Customer Relationship Management](https://example.com)
    - [Objection Handling Techniques](https://example.com)
    - [Premium Product Training](https://example.com)
    """)

# Main content
st.markdown(f"<h2 class='sub-header'>Agent Dashboard: {selected_agent}</h2>", unsafe_allow_html=True)

# Agent Performance Summary
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Add a performance status indicator
    performance_category = agent_data['performance_category']
    performance_class = f"{performance_category.lower()}-performer"
    
    st.markdown(f"""
    <div class="card">
        <h3>Performance Summary</h3>
        <p>Performance Category: <span class="{performance_class}">{performance_category} Performer</span></p>
        <p>Performance Score: {agent_data['performance_score']:.2f}</p>
    """, unsafe_allow_html=True)
    
    # Add trend indicator
    if agent_data['most_improved']:
        st.markdown('<p>Trend: <span class="improving">↑ Improving</span></p>', unsafe_allow_html=True)
    elif agent_data['declining']:
        st.markdown('<p>Trend: <span class="declining">↓ Declining</span></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p>Trend: → Stable</p>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Agent tenure info
    tenure_months = agent_data['tenure_months']
    years = int(tenure_months // 12)
    months = int(tenure_months % 12)
    tenure_str = f"{years}y {months}m" if years > 0 else f"{months} months"
    
    st.markdown(f"""
    <div class="card">
        <h3>Agent Info</h3>
        <p>Age: {agent_data['agent_age']}</p>
        <p>Tenure: {tenure_str}</p>
        <p>Joined: {agent_data['agent_join_month']}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Month info
    month_date = datetime.strptime(selected_month, '%Y-%m')
    month_name = month_date.strftime('%B %Y')
    
    st.markdown(f"""
    <div class="card">
        <h3>Current Month</h3>
        <p>{month_name}</p>
        <p>Report Date: {datetime.now().strftime('%d %b %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

# Key Performance Metrics
st.markdown("<h3 class='sub-header'>Key Performance Metrics</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>New Policies</h4>
        <h2>{agent_data['new_policy_count']}</h2>
        <p>{agent_data['new_policy_count_percentile']:.0f}th percentile</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>ANBP Value</h4>
        <h2>${agent_data['ANBP_value']:,.2f}</h2>
        <p>{agent_data['ANBP_value_percentile']:.0f}th percentile</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Net Income</h4>
        <h2>${agent_data['net_income']:,.2f}</h2>
        <p>{agent_data['net_income_percentile']:.0f}th percentile</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    proposal_to_policy = agent_data['proposal_to_policy_ratio'] * 100
    st.markdown(f"""
    <div class="metric-card">
        <h4>Conversion Rate</h4>
        <h2>{proposal_to_policy:.1f}%</h2>
        <p>{agent_data['proposal_to_policy_ratio_percentile']:.0f}th percentile</p>
    </div>
    """, unsafe_allow_html=True)

# Month-over-Month Changes
st.markdown("<h3 class='sub-header'>Month-over-Month Changes</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if 'new_policy_count_mom_pct_change' in agent_data:
        change = agent_data['new_policy_count_mom_pct_change']
        color = "improving" if change > 0 else "declining" if change < 0 else ""
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Policies Change</h4>
            <h2 class="{color}">{arrow} {change:.1f}%</h2>
            <p>From previous month</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>Policies Change</h4>
            <h2>N/A</h2>
            <p>No data available</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if 'ANBP_value_mom_pct_change' in agent_data:
        change = agent_data['ANBP_value_mom_pct_change']
        color = "improving" if change > 0 else "declining" if change < 0 else ""
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        st.markdown(f"""
        <div class="metric-card">
            <h4>ANBP Change</h4>
            <h2 class="{color}">{arrow} {change:.1f}%</h2>
            <p>From previous month</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>ANBP Change</h4>
            <h2>N/A</h2>
            <p>No data available</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if 'net_income_mom_pct_change' in agent_data:
        change = agent_data['net_income_mom_pct_change']
        color = "improving" if change > 0 else "declining" if change < 0 else ""
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Income Change</h4>
            <h2 class="{color}">{arrow} {change:.1f}%</h2>
            <p>From previous month</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>Income Change</h4>
            <h2>N/A</h2>
            <p>No data available</p>
        </div>
        """, unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Personalized Advice", "Historical Trends", "Peer Comparison", "Performance Analysis"])

with tab1:
    # Generate personalized advice
    advice = get_personalized_advice(agent_data)
    
    st.markdown(f"""
    <div class="info-box">
        <h3>Performance Summary</h3>
        <p>{advice['summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strengths")
        for strength in advice['strengths']:
            st.markdown(f"✅ {strength}")
        
        st.markdown("### Areas to Improve")
        for area in advice['areas_to_improve']:
            st.markdown(f"🔍 {area}")
    
    with col2:
        st.markdown("### Recommended Actions")
        for action in advice['action_items']:
            st.markdown(f"📋 {action}")
        
        st.markdown("### Recommended Resources")
        for resource in advice['resources']:
            st.markdown(f"📚 {resource}")
    
    # Goal setting section
    st.markdown("### Set Performance Goals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("New Policy Goal for Next Month", min_value=1, value=max(5, int(agent_data['new_policy_count'] * 1.1)))
    
    with col2:
        st.number_input("ANBP Value Goal for Next Month", min_value=1000, value=int(agent_data['ANBP_value'] * 1.1), step=1000)
    
    activity_goal = st.slider("Weekly Customer Contact Goal", min_value=5, max_value=50, value=20)
    
    if st.button("Save Goals"):
        st.success("Goals saved successfully! Your manager will be notified.")

with tab2:
    # Historical trends
    st.markdown("### Historical Performance Trends")
    
    # Select metrics to display
    trend_metrics = st.multiselect(
        "Select metrics to display",
        options=['new_policy_count', 'ANBP_value', 'net_income', 'proposal_to_policy_ratio'],
        default=['new_policy_count', 'ANBP_value']
    )
    
    if trend_metrics:
        trend_fig = create_agent_history_plot(df, selected_agent, trend_metrics)
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Please select at least one metric to display the trend.")
    
    # Show month-over-month percentage changes
    st.markdown("### Month-over-Month Growth")
    
    agent_history = df[df['agent_code'] == selected_agent].sort_values('year_month_dt')
    
    if len(agent_history) > 1:
        # Filter metrics for MoM changes
        mom_metrics = [col for col in agent_history.columns if '_mom_pct_change' in col]
        
        if mom_metrics:
            # Select last few months
            recent_months = agent_history.tail(6)
            
            # Plot MoM changes
            mom_fig = go.Figure()
            
            for metric in mom_metrics:
                display_name = metric.replace('_mom_pct_change', '').replace('_', ' ').title()
                mom_fig.add_trace(go.Bar(
                    x=recent_months['year_month'],
                    y=recent_months[metric],
                    name=display_name
                ))
            
            mom_fig.update_layout(
                title='Month-over-Month Percentage Changes',
                xaxis_title='Month',
                yaxis_title='% Change',
                height=400
            )
            
            st.plotly_chart(mom_fig, use_container_width=True)
        else:
            st.info("Month-over-month change data is not available.")
    else:
        st.info("Not enough historical data to calculate month-over-month changes.")

with tab3:
    # Peer comparison
    st.markdown("### Compare with Peers")
    
    comparison_metric = st.selectbox(
        "Select metric for comparison",
        options=['new_policy_count', 'ANBP_value', 'net_income', 'proposal_to_policy_ratio', 'customer_to_policy_ratio'],
        index=0
    )
    
    comparison_fig = create_comparison_plot(df, selected_month, selected_agent, comparison_metric)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Performance distribution
    st.markdown("### Performance Distribution")
    
    # Create histogram of performance scores
    fig = px.histogram(
        all_agents_current_month, 
        x='performance_score',
        nbins=20,
        title='Distribution of Performance Scores'
    )
    
    # Add a line for the selected agent
    fig.add_vline(
        x=agent_data['performance_score'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Your Score: {agent_data['performance_score']:.2f}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Performance analysis
    st.markdown("### Detailed Performance Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Radar chart of key metrics
        radar_metrics = {
            'new_policy_count': 'New Policies',
            'ANBP_value': 'Premium Value',
            'net_income': 'Net Income',
            'proposal_to_policy_ratio': 'Conversion Rate',
            'customer_to_policy_ratio': 'Customer Efficiency'
        }
        
        radar_fig = create_radar_chart(agent_data, radar_metrics)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        # Conversion funnel
        st.markdown("#### Sales Conversion Funnel")
        
        funnel_data = {
            'stage': ['Customers', 'Proposals', 'Quotations', 'Policies'],
            'count': [
                agent_data['unique_customers'],
                agent_data['unique_proposal'],
                agent_data['unique_quotations'],
                agent_data['new_policy_count']
            ]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        funnel_fig = px.funnel(
            funnel_df,
            x='count',
            y='stage',
            title='Sales Conversion Funnel'
        )
        
        st.plotly_chart(funnel_fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Metrics")
    
    detailed_metrics = {
        'Metric': [
            'New Policy Count', 'ANBP Value', 'Net Income',
            'Unique Proposals', 'Unique Quotations', 'Unique Customers',
            'Proposal to Policy Ratio', 'Quotation to Policy Ratio', 'Customer to Policy Ratio',
            'Avg ANBP per Policy', 'Performance Score', 'Improvement Score'
        ],
        'Value': [
            agent_data['new_policy_count'],
            f"${agent_data['ANBP_value']:,.2f}",
            f"${agent_data['net_income']:,.2f}",
            agent_data['unique_proposal'],
            agent_data['unique_quotations'],
            agent_data['unique_customers'],
            f"{agent_data['proposal_to_policy_ratio']:.2f}",
            f"{agent_data['quotation_to_policy_ratio']:.2f}",
            f"{agent_data['customer_to_policy_ratio']:.2f}",
            f"${agent_data['avg_ANBP_per_policy']:,.2f}" if 'avg_ANBP_per_policy' in agent_data else "N/A",
            f"{agent_data['performance_score']:.2f}",
            f"{agent_data['improvement_score']:.2f}" if 'improvement_score' in agent_data else "N/A"
        ],
        'Category Median': [
            f"{agent_data['new_policy_count_median']:.0f}",
            f"${agent_data['ANBP_value_median']:,.2f}",
            f"${agent_data['net_income_median']:,.2f}",
            "N/A", "N/A", "N/A",
            f"{agent_data['proposal_to_policy_ratio_median']:.2f}",
            "N/A",
            f"{agent_data['customer_to_policy_ratio_median']:.2f}",
            f"${agent_data['avg_ANBP_per_policy_median']:,.2f}" if 'avg_ANBP_per_policy_median' in agent_data else "N/A",
            "N/A", "N/A"
        ],
        'Percentile': [
            f"{agent_data['new_policy_count_percentile']:.0f}%",
            f"{agent_data['ANBP_value_percentile']:.0f}%",
            f"{agent_data['net_income_percentile']:.0f}%",
            "N/A", "N/A", "N/A",
            f"{agent_data['proposal_to_policy_ratio_percentile']:.0f}%",
            "N/A",
            f"{agent_data['customer_to_policy_ratio_percentile']:.0f}%",
            f"{agent_data['avg_ANBP_per_policy_percentile']:.0f}%" if 'avg_ANBP_per_policy_percentile' in agent_data else "N/A",
            "N/A", "N/A"
        ]
    }
    
    detailed_df = pd.DataFrame(detailed_metrics)
    st.dataframe(detailed_df, use_container_width=True, hide_index=True)

# Footer with navigation links
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Need Help?")
    st.markdown("Contact your manager or the support team at support@insurancecompany.com")

with col2:
    st.markdown("### Training Calendar")
    st.markdown("Next training session: Sales Masterclass on May 15, 2025")

with col3:
    st.markdown("### Quick Links")
    st.markdown("""
    - [Company Dashboard](https://example.com)
    - [Knowledge Base](https://example.com)
    - [Submit Feedback](https://example.com)
    """)

# Run the app with: streamlit run app.py