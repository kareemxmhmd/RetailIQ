import sys
from pathlib import Path

# Ensure project root is on sys.path so Streamlit can find the src module
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any

from src.data.data_loader import load_config

# Set page config
st.set_page_config(page_title='RetailIQ', layout='wide')

@st.cache_resource
def get_config() -> Dict[str, Any]:
    """Load configuration dictionary."""
    return load_config()

@st.cache_resource
def get_artifacts(_config: dict):
    """Load ML artifacts."""
    from src.models.inference import load_artifacts
    return load_artifacts(_config)

def main():
    """Main UI application."""
    st.title('RetailIQ, Customer Segmentation')
    st.markdown("Predict customer segments based on their Recency, Frequency, and Monetary (RFM) metrics.")
    
    config = get_config()
    
    st.sidebar.header('Customer Features')
    recency = st.sidebar.number_input('Recency (days since last purchase)', min_value=0, max_value=1000, value=30)
    frequency = st.sidebar.number_input('Frequency (number of orders)', min_value=1, max_value=10000, value=5)
    monetary = st.sidebar.number_input('Monetary (total spend $)', min_value=0.01, max_value=1000000.0, value=500.0, format='%.2f')
    
    predict_btn = st.sidebar.button('Predict Segment', type='primary', use_container_width=True)

    with st.sidebar.expander('About RetailIQ', expanded=False):
        st.markdown(
            """
            **RetailIQ** is an end-to-end customer segmentation engine powered by RFM analysis and KMeans clustering.
            
            - **Author**: [kareemxmhmd](https://github.com/kareemxmhmd)
            - **Source Code**: [GitHub Repository](https://github.com/kareemxmhmd/RetailIQ)
            - **Dataset**: UCI Online Retail (541K+ records)
            - **Serving**: FastAPI REST API & Streamlit
            """
        )
    
    if predict_btn:
        rfm_input = {
            'Recency': float(recency),
            'Frequency': float(frequency),
            'Monetary': float(monetary)
        }
        
        segment_name = None
        cluster_id = None
        description = None
        
        try:
            if config.get("ui", {}).get("mode", "direct") == "direct":
                from src.models.inference import predict_segment
                artifacts = get_artifacts(config)
                result = predict_segment(rfm_input, artifacts)
                cluster_id = result['cluster']
                segment_name = result['segment']
                description = config.get("segment_labels", {}).get("descriptions", {}).get(segment_name, "")
            elif config.get("ui", {}).get("mode") == "api":
                api_url = config.get("ui", {}).get("api_url", "http://localhost:8000")
                response = requests.post(f"{api_url}/predict", json=rfm_input)
                response.raise_for_status()
                data = response.json()
                cluster_id = data.get("cluster")
                segment_name = data.get("segment")
                description = data.get("description")
                
            if segment_name:
                st.metric(label="Predicted Segment", value=segment_name)
                
                if segment_name in ["VIP", "Loyal"]:
                    st.success(f"**{segment_name}**: {description}")
                elif segment_name in ["Regular", "New"]:
                    st.info(f"**{segment_name}**: {description}")
                elif segment_name in ["At-Risk", "Dormant", "Occasional"]:
                    st.warning(f"**{segment_name}**: {description}")
                else:
                    st.error(f"**{segment_name}**: {description}")
                    
                with st.expander("View Input Features"):
                    st.json(rfm_input)
                    
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    st.markdown("---")
    st.subheader("Segment Descriptions")
    
    segments = config.get("segment_labels", {}).get("names", [])
    descriptions = config.get("segment_labels", {}).get("descriptions", {})
    
    if segments and descriptions:
        df = pd.DataFrame({
            "Segment": segments,
            "Description": [descriptions.get(s, "") for s in segments]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
