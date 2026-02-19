"""
Utility functions for the Credit Card Fraud Detection System.
"""
import logging
import json
import sys
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    """
    Sets up logging configuration.
    Logs are saved to a file in the 'logs' directory and printed to stdout.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    logit = logging.getLogger("CreditCardFraud")
    logit.setLevel(logging.INFO)

    # JSON Formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_record)

    # File Handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(JsonFormatter())
    logit.addHandler(file_handler)

    # Console Handler (Human readable for dev)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logit.addHandler(console_handler)

    return logit

logger = setup_logging()
