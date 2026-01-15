
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from data import load_config, load_data

def select_features():
    print("Loading data...")
    config = load_config()
    train_df, _ = load_data(config)
    
    # Target (Binary is usually sufficient for general feature importance, 
    # but we can check Multiclass too. Let's use Multiclass 'attack_cat' 
    # to be sensitive to specific attack types corresponding to Janarthanan 2017)
    target_col = config["data"]["target_multiclass"]
    target_bin = config["data"]["target_binary"]
    drop_cols = config["data"].get("drop_columns", [])
    
    # Drop targets and identifiers
    X = train_df.drop(columns=[target_col, target_bin] + drop_cols, errors='ignore')
    # Use Multiclass target for selection to capture nuances of all classes
    y = train_df[target_col]
    
    # Preprocessing for MI calculation (MI needs numeric inputs)
    # 1. Numerics: Impute
    # 2. Categoricals: Ordinal Encode (MI works well with discrete encoded cats)
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Simple Encoder for Selection Only
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])
    
    print("Preprocessing for Feature Selection...")
    X_encoded = preprocessor.fit_transform(X)
    
    # Get feature names back
    feature_names = num_cols + cat_cols
    
    print("Calculating Mutual Information (Information Gain)...")
    # mutual_info_classif is discrete=False by default for X, but we have mixed.
    # It handles continuous appropriately.
    mi_scores = mutual_info_classif(X_encoded, y, discrete_features='auto', random_state=42)
    
    # Create DataFrame
    mi_df = pd.DataFrame({'feature': feature_names, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values(by='mi_score', ascending=False)
    
    print("\nTop Features by Information Gain:")
    print(mi_df.head(35))
    
    # Select Top 30
    top_30_features = mi_df['feature'].head(30).tolist()
    
    # Save to new config snippet
    selection_config = {
        "selected_features_top_30": top_30_features
    }
    
    with open("configs/selected_features.yaml", "w") as f:
        yaml.dump(selection_config, f)
        
    print("\nSaved Top 30 features to configs/selected_features.yaml")
    return top_30_features

if __name__ == "__main__":
    select_features()
