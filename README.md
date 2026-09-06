# RetailIQ — Customer Segmentation Engine

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-14%20Passing-brightgreen.svg)](https://docs.pytest.org/)

RetailIQ is an enterprise-grade machine learning system for customer segmentation built on retail transaction data. Combining behavioral **Recency, Frequency, Monetary (RFM)** analysis with unsupervised **KMeans clustering**, it converts raw e-commerce purchases into actionable business personas served via a high-performance **FastAPI** REST engine and an interactive **Streamlit** dashboard.

---

## About The Project

In e-commerce and modern retail, one-size-fits-all marketing results in low engagement, higher churn rates, and wasted ad spend. **RetailIQ** solves this problem by delivering an automated, reproducible segmentation framework that categorizes customers by their real behavioral footprint.

### Key Highlights
- **RFM Behavioral Modeling**: Quantifies purchase timeliness, transaction cadence, and lifetime monetary volume.
- **Unsupervised KMeans Clustering**: Segments customers into distinct groups using log-transformed features and normalized distances, validated by Silhouette Analysis and the Elbow Method.
- **Dual Serving Modes**: Features a low-latency FastAPI inference service (`/predict`) as well as an interactive Streamlit UI with live scoring.
- **Drift Monitoring**: Incorporates real-time Population Stability Index (PSI) tracking to detect when customer distributions shift over time.
- **Production Architecture**: Packaged with Docker Compose, automated Pytest test suites, and strict configuration management.

---

## Architecture & Data Flow

RetailIQ follows a modular, reproducible data science architecture:

1. **Load & Clean**: Loads transaction records, drops cancellations (`InvoiceNo` starting with `C`), negative quantities, and missing customer IDs, while casting types cleanly.
2. **Exploratory Data Analysis (EDA)**: Analyzes distributions, revenues by geography, and top products; exports summary visualizations to `reports/`.
3. **RFM Feature Engineering**: Computes Recency (days since last purchase), Frequency (distinct invoice count), and Monetary (total spend) per customer.
4. **Feature Preprocessing**: Handles skewness via log/power transformations and scales metrics using `StandardScaler`.
5. **Clustering & Profiling**: Evaluates optimal $k$ via Elbow method and Silhouette scores, fits KMeans, and profiles clusters into interpretable business segments.
6. **Inference & Monitoring**: Exposes a low-latency REST API and dashboard with real-time PSI (Population Stability Index) drift tracking.

---

## Segment Descriptions & Actionable Strategies

| Segment | Business Description | Recommended Marketing Action |
| :--- | :--- | :--- |
| **VIP** | Highest-value customers with recent, frequent, and high-spend purchases. | Prime candidates for VIP loyalty perks, concierge support, and early access. |
| **Loyal** | Consistent repeat buyers with solid spend and engagement. | Responsive to cross-sell and early-access campaigns; reward brand advocacy. |
| **Regular** | Steady mid-tier customers who make routine purchases. | Target with volume discounts, product bundles, and personalized recommendations. |
| **At-Risk** | Previously active customers showing declining recency and engagement. | Send automated win-back offers, satisfaction surveys, and special discounts. |
| **Dormant** | Customers who haven't purchased in a significant duration. | Re-engage via low-cost automated re-engagement email sequences. |
| **New** | Recently acquired customers with few transactions. | Optimize onboarding flows, deliver welcome discounts, and nurture repeat purchase. |
| **Occasional** | Infrequent, low-spend shoppers. | Capitalize on seasonal promotions, holiday offers, and flash sales. |
| **Churned** | Inactive customers with no recent purchases and low lifetime value. | Lowest priority for paid remarketing spend; focus ad budgets elsewhere. |

---

## Quick Start

### 1. Installation

Ensure Python 3.11+ is installed, then set up your environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Execute the full training, evaluation, and artifact generation sequence:

```bash
python src/run_all.py
```

### 3. Start the API Server

Launch the FastAPI inference engine:

```bash
uvicorn src.api.server:app --reload
```

The API will be accessible at `http://localhost:8000` (interactive documentation at `http://localhost:8000/docs`).

### 4. Start the Streamlit Dashboard

In a new terminal window, launch the interactive UI:

```bash
streamlit run src/ui/app.py
```

Open `http://localhost:8501` to view customer distributions, RFM metrics, and test live predictions.

---

## Technical Methodology

1. **RFM Derivation**:
   - **Recency ($R$)**: Days elapsed between the reference cutoff date and the customer's most recent transaction.
   - **Frequency ($F$)**: Count of distinct purchase invoices per customer.
   - **Monetary ($M$)**: Cumulative spend calculated as $\sum (\text{Quantity} \times \text{UnitPrice})$.
2. **Transformations & Scaling**:
   - Features exhibit substantial right-skewness. A logarithmic transformation ($\log_{1p}$) stabilizes variance, followed by `StandardScaler` normalization.
3. **Cluster Evaluation**:
   - Inertia curves (Elbow method) and Silhouette scores evaluate cluster separation across $k \in [2, 10]$ to select optimal partitions.
4. **Drift Detection**:
   - Baseline distribution comparison utilizing Population Stability Index (PSI):
     $$\text{PSI} = \sum \left( (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right) \right)$$
     Alerts flag when $\text{PSI} > 0.2$, signaling the need for model recalibration.

---

## Dataset

Built on the renowned **UCI Online Retail Dataset** comprising transatlantic e-commerce transactions:
- **Timeframe**: 01/12/2010 to 09/12/2011
- **Transactions**: 541,909 records across 38 countries
- **Attributes**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

## Repository Structure

```
RetailIQ/
├── artifacts/              # Serialized model, scaler, and segment mappings
├── config/
│   └── config.yaml         # Hyperparameters, paths, and segment labels
├── data/
│   └── Online Retail.csv   # Raw transaction records
├── notebook/
│   └── eda.ipynb           # Exploratory data analysis notebook
├── reports/                # EDA distribution plots and summaries
├── src/
│   ├── api/                # FastAPI server, schemas, and endpoints
│   ├── data/               # Ingestion, validation, and cleaning logic
│   ├── features/           # RFM feature extraction and preprocessing
│   ├── models/             # KMeans training, evaluation, and inference
│   ├── monitoring/         # PSI drift detector and JSONL audit logging
│   ├── ui/                 # Streamlit interactive application
│   └── run_all.py          # End-to-end execution orchestrator
├── tests/                  # Unit test suite
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service container orchestration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Docker Deployment

To build and run both the API and Streamlit UI via Docker Compose:

```bash
docker-compose up --build
```

- **API Service**: `http://localhost:8000`
- **Streamlit UI**: `http://localhost:8501`

---

## Testing

Run the automated test suite with coverage and verbose logging:

```bash
pytest tests/ -v
```

---

## Author & Attribution

- **Author**: [Kareem Mohamed](https://github.com/kareemxmhmd)
- **GitHub**: [@kareemxmhmd](https://github.com/kareemxmhmd)
- **Repository**: [https://github.com/kareemxmhmd/RetailIQ](https://github.com/kareemxmhmd/RetailIQ)

