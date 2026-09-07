# RetailIQ-Customer Segmentation Engine

> **Live Demo:** Explore the deployed interactive application at [https://retailiq-n.streamlit.app/](https://retailiq-n.streamlit.app/)

RetailIQ is an enterprise-grade machine learning system for customer segmentation built on retail transaction data. Combining behavioral **Recency, Frequency, Monetary (RFM)** analysis with unsupervised **KMeans clustering**, it converts raw e-commerce purchases into actionable business personas served via a high-performance **FastAPI** REST engine and an interactive **Streamlit** dashboard.

---

## About The Project

In e-commerce and modern retail, one-size-fits-all marketing results in low engagement, higher churn rates, and wasted ad spend. **RetailIQ** solves this problem by delivering an automated, reproducible segmentation framework that categorizes customers by their real behavioral footprint.

### Key Highlights
- **RFM Behavioral Modeling**: Quantifies purchase timeliness, transaction cadence, and lifetime monetary volume.
- **Unsupervised KMeans Clustering**: Segments customers into 8 distinct personas using log-transformed features (`np.log1p`), outlier percentile clipping, and `StandardScaler` normalized distances, validated by Silhouette Analysis, Calinski-Harabasz Index, Davies-Bouldin Index, and bootstrap stability (mean ARI > 0.90).
- **Archetype Persona Mapping**: Uses Hungarian matching between cluster RFM centroids and business definitions to deterministically assign personas without collapsing into degenerate whale-only clusters.
- **Dual Serving Modes**: Features a low-latency FastAPI inference service (`/predict` and `/health`) with bounded Pydantic validation as well as an interactive Streamlit UI with live scoring.
- **Drift Monitoring**: Incorporates real-time Population Stability Index (PSI) tracking to detect when customer distributions shift over time.
- **Production Architecture**: Packaged with Docker Compose, automated GitHub Actions CI workflow (`.github/workflows/ci.yml`), thread-safe audit logging, and centralized YAML configuration.

---

## Architecture & Data Flow

RetailIQ follows a modular, reproducible data science architecture:

1. **Load & Clean**: Loads transaction records, drops cancellations (`InvoiceNo` starting with `C`), negative quantities, and missing customer IDs, while casting types cleanly.
2. **Exploratory Data Analysis (EDA)**: Analyzes distributions, revenues by geography, and top products; exports summary visualizations to `reports/`.
3. **RFM Feature Engineering**: Computes Recency (days since last purchase), Frequency (distinct invoice count), and Monetary (total spend) per customer, along with behavioral metrics (`AvgBasketSize`, `AvgBasketValue`, `Tenure`, `UniqueProducts`).
4. **Feature Preprocessing**: Clips extreme wholesale outliers at the 99.9th percentile, applies `np.log1p` transformation to compress right-skewed Monetary and Frequency distributions, and scales metrics using `StandardScaler`.
5. **Clustering & Profiling**: Evaluates $k$ across multiple metrics (Elbow Inertia, Silhouette, Calinski-Harabasz, Davies-Bouldin), trains KMeans ($k=8$), validates cluster stability via bootstrap resampling (mean ARI ~0.92), and maps clusters to business personas via multi-dimensional RFM archetype matching.
6. **Inference & Monitoring**: Exposes a FastAPI REST API and dashboard with real-time PSI (Population Stability Index) drift tracking and thread-safe audit logging.

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
Alternatively, access the hosted cloud demo directly at [https://retailiq-n.streamlit.app/](https://retailiq-n.streamlit.app/).

---

## Docker Deployment

To build and run both the API and Streamlit UI via Docker Compose:

```bash
docker-compose up --build
```

- **API Service**: `http://localhost:8000`
- **Streamlit UI**: `http://localhost:8501`

---

## Testing & Continuous Integration

Run the automated test suite locally:

```bash
pytest tests/ -v
```

Automated testing is integrated with **GitHub Actions** via `.github/workflows/ci.yml`, running all unit and integration tests on push and pull requests to `main`.

---

## Production & Deployment Considerations

- **Artifact Versioning**: For local demonstration, production model weights and preprocessors are packaged in `artifacts/`. In enterprise production environments, models and scalers should be versioned through a dedicated model registry (e.g. MLflow, AWS S3, or Google Cloud Storage).
- **Data Governance**: Raw datasets are excluded from git version control via `.gitignore` to prevent repository bloat.
- **Audit Logging**: Local audit logs append predictions synchronously with thread-safe locking. Under heavy concurrent multi-worker load, delegate log writes to an asynchronous message broker (e.g. Celery, RabbitMQ, or Kafka).

---
