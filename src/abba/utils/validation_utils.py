"""Validation utility functions for ABBA system."""

import structlog
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

logger = structlog.get_logger()


class ValidationUtils:
    """Utility class for data validation operations."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None, 
                          numeric_columns: List[str] = None) -> Dict[str, Any]:
        """Validate a dataframe structure and content."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check if dataframe is empty
            if df.empty:
                validation_result["valid"] = False
                validation_result["errors"].append("DataFrame is empty")
                return validation_result
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required columns: {missing_columns}")
            
            # Check numeric columns
            if numeric_columns:
                non_numeric_columns = []
                for col in numeric_columns:
                    if col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            non_numeric_columns.append(col)
                
                if non_numeric_columns:
                    validation_result["warnings"].append(f"Non-numeric columns expected to be numeric: {non_numeric_columns}")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                validation_result["warnings"].append(f"Missing values found: {missing_counts.sum()} total")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                validation_result["warnings"].append(f"Duplicate rows found: {duplicates}")
            
            logger.info(f"DataFrame validation completed: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating DataFrame: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    @staticmethod
    def validate_odds(odds: Union[float, int]) -> bool:
        """Validate that odds are reasonable."""
        try:
            if not isinstance(odds, (int, float)):
                return False
            
            # Odds should be greater than 1.0
            if odds <= 1.0:
                return False
            
            # Odds should not be unreasonably high (e.g., > 1000)
            if odds > 1000:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating odds: {e}")
            return False
    
    @staticmethod
    def validate_probability(probability: float) -> bool:
        """Validate that probability is between 0 and 1."""
        try:
            if not isinstance(probability, (int, float)):
                return False
            
            if probability < 0.0 or probability > 1.0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating probability: {e}")
            return False
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """Validate date range."""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt >= end_dt:
                return False
            
            # Check if dates are not too far in the future
            if end_dt > datetime.now() + pd.Timedelta(days=365):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating date range: {e}")
            return False
    
    @staticmethod
    def validate_sport(sport: str) -> bool:
        """Validate sport name."""
        try:
            valid_sports = ["MLB", "NHL", "NBA", "NFL", "SOCCER"]
            return sport.upper() in valid_sports
            
        except Exception as e:
            logger.error(f"Error validating sport: {e}")
            return False
    
    @staticmethod
    def validate_bet_amount(amount: float, min_amount: float = 1.0, max_amount: float = 10000.0) -> bool:
        """Validate bet amount."""
        try:
            if not isinstance(amount, (int, float)):
                return False
            
            if amount < min_amount or amount > max_amount:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating bet amount: {e}")
            return False
    
    @staticmethod
    def validate_model_prediction(prediction: np.ndarray) -> bool:
        """Validate model prediction output."""
        try:
            if not isinstance(prediction, np.ndarray):
                return False
            
            if prediction.size == 0:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                return False
            
            # Check if probabilities sum to 1 (for classification)
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                row_sums = np.sum(prediction, axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model prediction: {e}")
            return False
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            if not isinstance(config, dict):
                validation_result["valid"] = False
                validation_result["errors"].append("Config must be a dictionary")
                return validation_result
            
            # Check required keys
            if required_keys:
                missing_keys = [key for key in required_keys if key not in config]
                if missing_keys:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required config keys: {missing_keys}")
            
            # Validate specific config values
            if "risk_threshold" in config:
                if not ValidationUtils.validate_probability(config["risk_threshold"]):
                    validation_result["errors"].append("Invalid risk_threshold")
            
            if "max_position_size" in config:
                if not ValidationUtils.validate_probability(config["max_position_size"]):
                    validation_result["errors"].append("Invalid max_position_size")
            
            logger.info(f"Config validation completed: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return {
                "valid": False,
                "errors": [f"Config validation error: {str(e)}"],
                "warnings": []
            } 