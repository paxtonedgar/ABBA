"""Data utility functions for ABBA system."""

import structlog
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = structlog.get_logger()


class DataUtils:
    """Utility class for data processing operations."""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a dataframe by removing duplicates and handling missing values."""
        try:
            # Remove duplicates
            df_clean = df.drop_duplicates()
            
            # Handle missing values in numeric columns
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
            # Handle missing values in categorical columns
            categorical_columns = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
            
            logger.info(f"DataFrame cleaned: {len(df)} -> {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {e}")
            return df
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        try:
            df_normalized = df.copy()
            
            for col in columns:
                if col in df_normalized.columns:
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
            
            logger.info(f"Features normalized: {columns}")
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return df
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create time-based features from date column."""
        try:
            df_features = df.copy()
            
            if date_column in df_features.columns:
                df_features[date_column] = pd.to_datetime(df_features[date_column])
                
                # Extract time features
                df_features['year'] = df_features[date_column].dt.year
                df_features['month'] = df_features[date_column].dt.month
                df_features['day'] = df_features[date_column].dt.day
                df_features['day_of_week'] = df_features[date_column].dt.dayofweek
                df_features['quarter'] = df_features[date_column].dt.quarter
                
                # Create season features for sports
                df_features['is_playoff_season'] = df_features['month'].isin([10, 11, 12, 1, 2, 3])
                df_features['is_regular_season'] = df_features['month'].isin([4, 5, 6, 7, 8, 9])
            
            logger.info(f"Time features created from {date_column}")
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return df
    
    @staticmethod
    def calculate_rolling_stats(df: pd.DataFrame, group_column: str, value_column: str, 
                               window: int = 10) -> pd.DataFrame:
        """Calculate rolling statistics for grouped data."""
        try:
            df_rolling = df.copy()
            
            # Calculate rolling mean and std
            rolling_mean = df_rolling.groupby(group_column)[value_column].rolling(
                window=window, min_periods=1
            ).mean().reset_index()
            
            rolling_std = df_rolling.groupby(group_column)[value_column].rolling(
                window=window, min_periods=1
            ).std().reset_index()
            
            # Merge back to original dataframe
            df_rolling[f'{value_column}_rolling_mean'] = rolling_mean[value_column]
            df_rolling[f'{value_column}_rolling_std'] = rolling_std[value_column]
            
            logger.info(f"Rolling stats calculated for {value_column} with window {window}")
            return df_rolling
            
        except Exception as e:
            logger.error(f"Error calculating rolling stats: {e}")
            return df
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        try:
            df_encoded = df.copy()
            
            for col in columns:
                if col in df_encoded.columns:
                    # One-hot encode
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
            
            logger.info(f"Categorical variables encoded: {columns}")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {e}")
            return df
    
    @staticmethod
    def split_train_test(df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
                        random_state: int = 42) -> tuple:
        """Split data into training and testing sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            # Remove target column from features
            features = df.drop(columns=[target_column])
            target = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None, None, None, None 