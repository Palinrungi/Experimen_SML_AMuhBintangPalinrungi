"""
Automated Preprocessing Script for Credit Card Fraud Detection
Author: Bintang Palinrungi
Description: Script untuk melakukan preprocessing otomatis pada dataset credit card fraud
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class CreditCardFraudPreprocessor:
    """
    Class untuk preprocessing data credit card fraud detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_merchant = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """
        Load dataset dari file CSV
        
        Args:
            filepath (str): Path ke file CSV
            
        Returns:
            pd.DataFrame: Dataset yang sudah dimuat
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_duplicates(self, df):
        """
        Menghapus data duplikat
        
        Args:
            df (pd.DataFrame): Dataset input
            
        Returns:
            pd.DataFrame: Dataset tanpa duplikat
        """
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        print(f"Removed {removed} duplicate rows")
        return df
    
    def feature_engineering(self, df):
        """
        Membuat fitur-fitur baru dari data yang ada
        
        Args:
            df (pd.DataFrame): Dataset input
            
        Returns:
            pd.DataFrame: Dataset dengan fitur baru
        """
        print("Creating new features...")
        df = df.copy()
        
        # Transaction hour categories
        df['hour_category'] = pd.cut(df['transaction_hour'], 
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                     include_lowest=True)
        
        # Amount categories
        df['amount_category'] = pd.cut(df['amount'],
                                       bins=[0, 50, 150, 300, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Age groups
        df['age_group'] = pd.cut(df['cardholder_age'],
                                bins=[0, 25, 35, 50, 100],
                                labels=['Young', 'Adult', 'Middle Age', 'Senior'])
        
        # Risk score calculation
        df['risk_score'] = (
            df['foreign_transaction'] * 2 +
            df['location_mismatch'] * 3 +
            (100 - df['device_trust_score']) / 20 +
            df['velocity_last_24h'] * 1.5
        )
        
        print("New features created: hour_category, amount_category, age_group, risk_score")
        return df
    
    def encode_features(self, df, is_training=True):
        """
        Encode categorical features
        
        Args:
            df (pd.DataFrame): Dataset input
            is_training (bool): Flag untuk training atau inference
            
        Returns:
            pd.DataFrame: Dataset dengan fitur ter-encode
        """
        print("Encoding categorical features...")
        df = df.copy()
        
        # Label encoding untuk merchant_category
        if is_training:
            df['merchant_category_encoded'] = self.le_merchant.fit_transform(df['merchant_category'])
        else:
            df['merchant_category_encoded'] = self.le_merchant.transform(df['merchant_category'])
        
        # One-hot encoding untuk fitur kategorikal baru
        df = pd.get_dummies(df, columns=['hour_category', 'amount_category', 'age_group'], 
                           drop_first=True)
        
        print("Categorical features encoded")
        return df
    
    def prepare_features(self, df):
        """
        Mempersiapkan fitur dengan menghapus kolom yang tidak diperlukan
        
        Args:
            df (pd.DataFrame): Dataset input
            
        Returns:
            tuple: (X, y) features dan target
        """
        print("Preparing features and target...")
        df = df.copy()
        
        # Drop kolom yang tidak diperlukan
        columns_to_drop = ['transaction_id', 'merchant_category']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Pisahkan features dan target
        if 'is_fraud' in df.columns:
            X = df.drop('is_fraud', axis=1)
            y = df['is_fraud']
        else:
            X = df
            y = None
        
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
        
        return X, y
    
    def balance_data(self, X, y, random_state=42):
        """
        Balance data menggunakan SMOTE
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            random_state (int): Random state untuk reproducibility
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        print("Balancing data with SMOTE...")
        print(f"Before SMOTE - Fraud: {y.sum()}, Non-fraud: {(y == 0).sum()}")
        
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"After SMOTE - Fraud: {y_balanced.sum()}, Non-fraud: {(y_balanced == 0).sum()}")
        return X_balanced, y_balanced
    
    def scale_features(self, X, is_training=True):
        """
        Scale features menggunakan StandardScaler
        
        Args:
            X (pd.DataFrame): Features
            is_training (bool): Flag untuk training atau inference
            
        Returns:
            pd.DataFrame: Scaled features
        """
        print("Scaling features...")
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        print("Features scaled")
        return X_scaled
    
    def preprocess(self, filepath, test_size=0.2, random_state=42, 
                   balance=True, save_preprocessor=True, output_dir='data/processed'):
        """
        Fungsi utama untuk melakukan preprocessing lengkap
        
        Args:
            filepath (str): Path ke file CSV
            test_size (float): Proporsi data test
            random_state (int): Random state
            balance (bool): Apakah perlu balance data
            save_preprocessor (bool): Apakah perlu save preprocessor objects
            output_dir (str): Directory untuk menyimpan output
            
        Returns:
            dict: Dictionary berisi data yang sudah diproses
        """
        print("="*80)
        print("STARTING AUTOMATED PREPROCESSING")
        print("="*80)
        
        # Step 1: Load data
        df = self.load_data(filepath)
        
        # Step 2: Handle duplicates
        df = self.handle_duplicates(df)
        
        # Step 3: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 4: Encode features
        df = self.encode_features(df, is_training=True)
        
        # Step 5: Prepare features
        X, y = self.prepare_features(df)
        
        # Simpan feature columns
        self.feature_columns = X.columns.tolist()
        
        # Step 6: Split data
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Step 7: Balance data (optional)
        if balance:
            X_train, y_train = self.balance_data(X_train, y_train, random_state)
        
        # Step 8: Scale features
        X_train_scaled = self.scale_features(X_train, is_training=True)
        X_test_scaled = self.scale_features(X_test, is_training=False)
        
        # Save preprocessor objects
        if save_preprocessor:
            os.makedirs(output_dir, exist_ok=True)
            
            preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'le_merchant': self.le_merchant,
                    'feature_columns': self.feature_columns
                }, f)
            print(f"\nPreprocessor saved to {preprocessor_path}")
            
            # Save processed data
            train_data_path = os.path.join(output_dir, 'train_data.csv')
            test_data_path = os.path.join(output_dir, 'test_data.csv')
            
            train_df = X_train_scaled.copy()
            train_df['is_fraud'] = y_train.values
            train_df.to_csv(train_data_path, index=False)
            
            test_df = X_test_scaled.copy()
            test_df['is_fraud'] = y_test.values
            test_df.to_csv(test_data_path, index=False)
            
            print(f"Processed data saved to {output_dir}")
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nFinal Summary:")
        print(f"  - Training samples: {X_train_scaled.shape[0]}")
        print(f"  - Test samples: {X_test_scaled.shape[0]}")
        print(f"  - Number of features: {X_train_scaled.shape[1]}")
        print(f"  - Training set balanced: {balance}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def load_preprocessor(self, preprocessor_path):
        """
        Load preprocessor objects dari file
        
        Args:
            preprocessor_path (str): Path ke file preprocessor
        """
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.le_merchant = preprocessor_data['le_merchant']
        self.feature_columns = preprocessor_data['feature_columns']
        
        print(f"Preprocessor loaded from {preprocessor_path}")
    
    def transform_new_data(self, df):
        """
        Transform data baru menggunakan preprocessor yang sudah di-fit
        
        Args:
            df (pd.DataFrame): Data baru
            
        Returns:
            pd.DataFrame: Data yang sudah ditransform
        """
        print("Transforming new data...")
        
        # Apply same transformations
        df = self.handle_duplicates(df)
        df = self.feature_engineering(df)
        df = self.encode_features(df, is_training=False)
        X, y = self.prepare_features(df)
        
        # Ensure same columns as training
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns]
        
        # Scale
        X_scaled = self.scale_features(X, is_training=False)
        
        print("New data transformed successfully")
        return X_scaled, y


def main():
    """
    Fungsi main untuk menjalankan preprocessing
    """
    # Initialize preprocessor
    preprocessor = CreditCardFraudPreprocessor()
    
    # Path ke dataset
    data_path = 'Credit Card Dataset/credit_card_fraud_10k.csv'
    
    # Jalankan preprocessing
    processed_data = preprocessor.preprocess(
        filepath=data_path,
        test_size=0.2,
        random_state=42,
        balance=True,
        save_preprocessor=True,
        output_dir='data/processed'
    )
    
    print("\nPreprocessing selesai! Data siap untuk training.")
    print(f"X_train shape: {processed_data['X_train'].shape}")
    print(f"X_test shape: {processed_data['X_test'].shape}")
    print(f"Number of features: {len(processed_data['feature_columns'])}")


if __name__ == "__main__":
    main()
