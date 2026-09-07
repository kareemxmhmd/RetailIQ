"""
Audit logger for tracking model predictions in RetailIQ.
Includes thread-safe local file appending for development and single-process servers.
Note: For distributed multi-worker production deployments, an asynchronous queue or
dedicated log aggregation service (e.g., Kafka, fluentd, or CloudWatch) is recommended.
"""
import json
import os
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
_audit_lock = threading.Lock()

def log_prediction(input_data: dict, output_data: dict, config: dict) -> None:
    """
    Log a prediction with its input and output data. Thread-safe via internal mutex.

    Args:
        input_data (dict): The input features for the prediction.
        output_data (dict): The prediction output/result.
        config (dict): The configuration dictionary containing paths.
    """
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'input': input_data,
        'output': output_data
    }
    
    audit_log_path = config['paths']['audit_log']
    os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
    
    try:
        with _audit_lock:
            with open(audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Prediction audited and saved to {audit_log_path}")
    except Exception as e:
        logger.error(f"Failed to write prediction to audit log: {e}")
        raise

def read_audit_log(config: dict) -> list[dict]:
    """
    Read all JSON lines from the audit log file.

    Args:
        config (dict): The configuration dictionary containing paths.

    Returns:
        list[dict]: A list of records read from the audit log. Returns empty list if file not found.
    """
    audit_log_path = config['paths']['audit_log']
    records = []
    
    try:
        with open(audit_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    except FileNotFoundError:
        logger.warning(f"Audit log file not found at {audit_log_path}. Returning empty list.")
        return []
