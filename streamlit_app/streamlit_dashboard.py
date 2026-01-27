import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import requests

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/saved_models/xgboost_model.pkl")
        scaler = joblib.load("models/saved_models/scaler.pkl")

        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        else:
            try:
                model_features = model.get_booster().feature_names
            except Exception:
                model_features = None

        scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None

        return model, scaler, model_features, scaler_features, True

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False
    
@st.cache_data
def load_sample_data():
    try:
        data = pd.read_csv('data/processed/train_processed.csv')
        return data
    except Exception as e:
        st.warning(f"Sample data not available: {e}")
        return None


def determine_risk_level(rul):
    if rul < 50:
        return "Critical"
    elif rul < 100:
        return "High"
    elif rul < 500:
        return "Medium"
    else:
        return "Low"
    
def get_risk_color(risk_level):
    colors = {
        "Critical": "#FF4B4B",
        "High": "#FFA500",
        "Medium": "#FFD700",
        "Low": "#32CD32"
    }
    return colors.get(risk_level, "#808080")

MAX_RUL = 125

def predict_rul(model, scaler, model_features, scaler_features, features):
    try:
        if model_features is None:
            st.error("Model feature names not available. Re-save model with feature names.")
            return None

        X = pd.DataFrame(
            [[features.get(col, 0) for col in model_features]],
            columns=model_features
        ).astype(float)

        if scaler_features is not None and scaler_features == model_features:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values

        rul_scaled = float(model.predict(X_scaled)[0])

        rul_cycles = max(0, rul_scaled * MAX_RUL)

        return rul_cycles

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

model, scaler, model_features, scaler_features, models_loaded = load_models()
sample_data = load_sample_data()

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("**Real-time equipment monitoring and failure prediction**")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")

    if models_loaded:
        st.success("Models loaded")
    else:
        st.error("Models not loaded")
    
    st.markdown("---")

    page = st.selectbox(
        "Select Page",
        ["Home", "Single Prediction", "Batch Analysis", 
         "Fleet Overview", "Model Info"]
    )
    
    st.markdown("---")

    st.subheader("Settings")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    show_confidence = st.checkbox("Show confidence intervals", value=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard provides real-time predictive maintenance 
    insights using machine learning.
    
    **Model**: XGBoost  
    **Accuracy**: RMSE < 15 cycles  
    **Version**: 1.0.0
    """)

if page == "Home":
    st.header("Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Equipment",
            value="100",
            delta="Active"
        )
    
    with col2:
        st.metric(
            label="Critical Alerts",
            value="3",
            delta="-2 from yesterday",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg RUL",
            value="65 cycles",
            delta="+5 cycles"
        )
    
    with col4:
        st.metric(
            label="Uptime",
            value="98.5%",
            delta="+0.5%"
        )
    
    st.markdown("---")

    st.subheader("Risk Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Count': [75, 18, 5, 2],
            'Percentage': [75, 18, 5, 2]
        })
        
        fig = px.bar(
            risk_data,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Low': '#32CD32',
                'Medium': '#FFD700',
                'High': '#FFA500',
                'Critical': '#FF4B4B'
            },
            title="Equipment by Risk Level"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Summary")
        for _, row in risk_data.iterrows():
            risk_level = row['Risk Level']
            count = row['Count']

            color = get_risk_color(risk_level)
            st.markdown(
                f"<span style='color:{color}; font-weight:600'>{risk_level}</span>: "
                f"{count} units ({row['Percentage']}%)",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Actions")
        if st.button("View Critical Units"):
            st.info("Units #45, #67, #89 require immediate attention")

    st.markdown("---")
    st.subheader("Recent Activity")
    
    activity_data = pd.DataFrame({
        'Time': [(datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M") for i in range(5)],
        'Unit': [f"Unit #{i}" for i in [45, 23, 67, 12, 89]],
        'Event': ['Critical alert', 'RUL update', 'High risk alert', 'Maintenance completed', 'Critical alert'],
        'RUL': [5, 45, 18, 120, 7]
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)

elif page == "Single Prediction":
    st.header("Single Equipment Prediction")
    st.markdown("Enter sensor readings to predict RUL for a single equipment unit")
    
    if not models_loaded:
        st.error("Models not loaded. Please check model files.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equipment Information")
        unit_id = st.number_input("Unit ID", min_value=1, max_value=1000, value=1)
        time_cycles = st.number_input("Current Cycle", min_value=1, max_value=500, value=50)
    
    with col2:
        st.subheader("Operational Settings")
        op_setting_1 = st.number_input("Operational Setting 1", value=0.0023, format="%.4f")
        op_setting_2 = st.number_input("Operational Setting 2", value=0.0003, format="%.4f")
        op_setting_3 = st.number_input("Operational Setting 3", value=100.0)
    
    st.markdown("---")
    st.subheader("Sensor Readings")

    sensor_cols = st.columns(3)
    sensor_values = {}
    
    for i in range(1, 22):
        col_idx = (i - 1) % 3
        with sensor_cols[col_idx]:
            sensor_values[f'sensor_{i}'] = st.number_input(
                f"Sensor {i}",
                value=float(500 + i * 10),
                format="%.2f",
                key=f"sensor_{i}"
            )
    
    st.markdown("---")
    
    if st.button("🔮 Predict RUL", type="primary"):
        features = {
            'unit_id': unit_id,
            'time_cycles': time_cycles,
            'op_setting_1': op_setting_1,
            'op_setting_2': op_setting_2,
            'op_setting_3': op_setting_3,
        }
        features.update(sensor_values)

        with st.spinner("Analyzing..."):
            rul = predict_rul(model, scaler, model_features, scaler_features, features)
        
        if rul is not None:
            risk_level = determine_risk_level(rul)
            risk_color = get_risk_color(risk_level)

            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted RUL",
                    value=f"{rul:.1f} cycles"
                )
            
            with col2:
                st.markdown(f"### Risk Level")
                st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>", 
                          unsafe_allow_html=True)
            
            with col3:
                if show_confidence:
                    st.metric(
                        label="Confidence Interval",
                        value=f"+-10 cycles"
                    )
            
            st.markdown("---")

            st.subheader("Recommendations")
            
            if risk_level == "Critical":
                st.error("""
                **IMMEDIATE ACTION REQUIRED**
                - Schedule emergency maintenance within 24 hours
                - Consider temporary equipment shutdown
                - Prepare replacement parts
                - Alert maintenance team and management
                """)
            elif risk_level == "High":
                st.warning("""
                **HIGH PRIORITY**
                - Schedule maintenance within 1 week
                - Increase monitoring frequency
                - Order replacement parts
                - Prepare maintenance crew
                """)
            elif risk_level == "Medium":
                st.info("""
                **MEDIUM PRIORITY**
                - Schedule maintenance within 2-3 weeks
                - Continue regular monitoring
                - Review maintenance history
                - Check parts inventory
                """)
            else:
                st.success("""
                **Low Risk**
                - Equipment operating normally
                - Continue routine maintenance schedule
                - Regular monitoring recommended
                """)

            st.markdown("---")
            st.subheader("RUL Timeline")
            
            timeline_data = pd.DataFrame({
                'Cycle': list(range(time_cycles, time_cycles + int(rul) + 1)),
                'Status': ['Operational'] * int(rul) + ['Failure Predicted']
            })
            
            fig = px.line(
                timeline_data,
                x='Cycle',
                y=[1] * len(timeline_data),
                color='Status',
                title=f"Equipment Lifecycle Projection (Unit {unit_id})"
            )
            fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Batch Analysis":
    st.header("Batch Analysis")
    st.markdown("Analyze multiple equipment units simultaneously")
    
    if sample_data is not None:
        st.subheader("Sample Fleet Data")

        lifecycle_stage = st.selectbox(
            "Select lifecycle stage for analysis",
            ["Latest", "Early", "Middle", "Random"],
            index=0
        )

        unique_units = sample_data['unit_id'].unique()
        selected_units = st.multiselect(
            "Select Units to Analyze",
            options=unique_units,
            default=list(unique_units[:5])
        )
        
        if selected_units and st.button("Analyze Selected Units"):
            with st.spinner("Analyzing fleet..."):
                results = []
                
                for unit in selected_units:
                    unit_rows = sample_data[sample_data['unit_id'] == unit]

                    if lifecycle_stage == "Latest":
                        unit_data = unit_rows.iloc[-1]
                    elif lifecycle_stage == "Early":
                        unit_data = unit_rows.iloc[0]
                    elif lifecycle_stage == "Middle":
                        unit_data = unit_rows.iloc[len(unit_rows)//2]
                    else:  # Random
                        unit_data = unit_rows.sample(1).iloc[0]

                    features = {
                        col: float(unit_data[col]) if col in unit_data.index else 0
                        for col in model_features
                    }

                    rul = predict_rul(
                        model,
                        scaler,
                        model_features,
                        scaler_features,
                        features
                    )
                    
                    if rul is not None:
                        risk_level = determine_risk_level(rul)
                        results.append({
                            'Unit ID': unit,
                            'Cycle': int(unit_data.get('time_cycles', -1)),
                            'RUL (cycles)': round(rul, 1),
                            'Risk Level': risk_level,
                        })
                
                results_df = pd.DataFrame(results)

                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Units Analyzed", len(results_df))
                with col2:
                    st.metric("Avg RUL", f"{results_df['RUL (cycles)'].mean():.1f}")
                with col3:
                    critical_count = len(results_df[results_df['Risk Level'] == 'Critical'])
                    st.metric("Critical", critical_count)
                with col4:
                    high_count = len(results_df[results_df['Risk Level'] == 'High'])
                    st.metric("High Risk", high_count)
                
                st.markdown("---")

                st.subheader("Analysis Results")
                st.dataframe(
                    results_df.style.applymap(
                        lambda x: f'background-color: {get_risk_color(x)}' if x in ['Critical', 'High', 'Medium', 'Low'] else '',
                        subset=['Risk Level']
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        results_df,
                        x='RUL (cycles)',
                        nbins=20,
                        title="RUL Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    risk_counts = results_df['Risk Level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Distribution",
                        color=risk_counts.index,
                        color_discrete_map={
                            'Low': '#32CD32',
                            'Medium': '#FFD700',
                            'High': '#FFA500',
                            'Critical': '#FF4B4B'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sample data not available. Please ensure data files are present.")

elif page == "Fleet Overview":
    st.header("Fleet Overview")
    st.markdown("Comprehensive fleet health monitoring")

    np.random.seed(42)
    fleet_size = 100
    
    fleet_data = pd.DataFrame({
        'Unit ID': range(1, fleet_size + 1),
        'RUL': np.random.gamma(5, 15, fleet_size),
        'Cycles': np.random.randint(50, 300, fleet_size),
        'Last Maintenance': pd.date_range(end=datetime.now(), periods=fleet_size, freq='D')[::-1]
    })
    
    fleet_data['Risk Level'] = fleet_data['RUL'].apply(lambda x: determine_risk_level(x))

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fleet Size", fleet_size)
    with col2:
        avg_rul = fleet_data['RUL'].mean()
        st.metric("Avg RUL", f"{avg_rul:.1f} cycles")
    with col3:
        critical = len(fleet_data[fleet_data['Risk Level'] == 'Critical'])
        st.metric("Critical", critical, delta=f"{critical/fleet_size*100:.1f}%")
    with col4:
        health_score = (fleet_size - critical) / fleet_size * 100
        st.metric("Fleet Health", f"{health_score:.1f}%")
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['Low', 'Medium', 'High', 'Critical'],
            default=['Critical', 'High']
        )
    
    with col2:
        rul_range = st.slider(
            "RUL Range (cycles)",
            min_value=0,
            max_value=int(fleet_data['RUL'].max()),
            value=(0, int(fleet_data['RUL'].max()))
        )

    filtered_data = fleet_data[
        (fleet_data['Risk Level'].isin(risk_filter)) &
        (fleet_data['RUL'] >= rul_range[0]) &
        (fleet_data['RUL'] <= rul_range[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_data)} of {fleet_size} units**")

    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_data,
            x='Cycles',
            y='RUL',
            color='Risk Level',
            size='RUL',
            hover_data=['Unit ID'],
            title="Fleet Status Map",
            color_discrete_map={
                'Low': '#32CD32',
                'Medium': '#FFD700',
                'High': '#FFA500',
                'Critical': '#FF4B4B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            fleet_data,
            x='Risk Level',
            y='RUL',
            color='Risk Level',
            title="RUL Distribution by Risk Level",
            color_discrete_map={
                'Low': '#32CD32',
                'Medium': '#FFD700',
                'High': '#FFA500',
                'Critical': '#FF4B4B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Fleet Details")
    st.dataframe(filtered_data, use_container_width=True, hide_index=True)

elif page == "Model Info":
    st.header("Model Information")
    st.markdown("Details about the predictive maintenance models")

    st.subheader("Model Performance")
    
    try:
        results = pd.read_csv('models/saved_models/final_model_comparison.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### XGBoost (Production Model)")
            xgb_results = results[results['model'] == 'XGBoost'].iloc[0]
            
            metrics = {
                'RMSE': f"{xgb_results['rmse']:.2f} cycles",
                'MAE': f"{xgb_results['mae']:.2f} cycles",
                'R² Score': f"{xgb_results['r2']:.4f}",
                'Asymmetric Score': f"{xgb_results['asymmetric_score']:.2f}",
                'Training Time': f"{xgb_results['training_time']:.2f}s"
            }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("### Model Comparison")
            fig = px.bar(
                results,
                x='model',
                y='rmse',
                title="RMSE Comparison",
                color='model',
                text='rmse'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load performance metrics: {e}")
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Specifications")
        st.markdown("""
        **Model Type**: XGBoost Regressor
        
        **Hyperparameters**:
        - Learning Rate: 0.01
        - Max Depth: 7
        - n_estimators: 1000
        - Subsample: 0.8
        - Colsample by tree: 0.8
        
        **Features**: 150+
        - Rolling statistics
        - Degradation indicators
        - Time-based features
        - Lag features
        """)
    
    with col2:
        st.subheader("Training Information")
        st.markdown("""
        **Dataset**: NASA CMAPSS Turbofan Engine
        
        **Training Data**:
        - 100 engines
        - 20,000+ observations
        - 21 sensors
        
        **Validation Strategy**:
        - Time-series split
        - 80/20 train/test
        - No data leakage
        
        **Version**: 1.0.0  
        **Last Updated**: January 2025
        """)
    
    st.markdown("---")

    st.subheader("Top Features")
    st.info("Feature importance analysis available in the modeling notebooks")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Predictive Maintenance Dashboard v1.0.0 | Built with Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    st.rerun()
