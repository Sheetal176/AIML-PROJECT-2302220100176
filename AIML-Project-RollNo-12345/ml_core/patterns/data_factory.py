import pandas as pd

class DataProcessorFactory:
    @staticmethod
    def get_processor(processor_type: str):
        if processor_type == "standard":
            return StandardDataProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

class StandardDataProcessor:
    def __init__(self):
        self.expected_columns = [
            'gender_M', 'hsc_s_Commerce', 'hsc_s_Science',
            'degree_t_Others', 'degree_t_Sci&Tech', 'workex_Yes',
            'specialisation_Mkt&HR'
        ]

    def preprocess(self, df: pd.DataFrame, is_training: bool = True):
        # Drop salary and sl_no if they exist
        cols_to_drop = ['salary', 'sl_no', 'ssc_b', 'hsc_b']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            
        # Clip percentages
        perc_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
        for col in perc_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0, upper=100)
                
        # Standardize categorical cols
        cat_cols = ['gender', 'workex', 'specialisation']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].str.title()
                
        # Feature Engineering: academic average
        if all(c in df.columns for c in ['ssc_p', 'hsc_p', 'degree_p']):
            df['academic_average'] = df[['ssc_p', 'hsc_p', 'degree_p']].mean(axis=1)
            
        # If target 'status' exists, map it
        if 'status' in df.columns:
            df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
            y = df['status']
            df = df.drop(columns=['status'])
        else:
            y = None
            
        # One-hot encode
        cols_to_encode = ['gender', 'hsc_s', 'degree_t', 'workex', 'specialisation']
        
        if is_training:
            cols_present = [c for c in cols_to_encode if c in df.columns]
            df_encoded = pd.get_dummies(df, columns=cols_present, drop_first=True, dtype=int)
        else:
            df_encoded = df.copy()
            for col in self.expected_columns:
                if col in perc_cols or col == 'academic_average':
                    continue
                # For categorical columns, parse feature and value
                for feature in cols_to_encode:
                    if col.startswith(feature + '_'):
                        val = col[len(feature)+1:]
                        if feature in df.columns:
                            df_encoded[col] = (df[feature].astype(str) == val).astype(int)
                        else:
                            df_encoded[col] = 0
            
            # Keep only the exact features the model was trained on, in the exact order
            df_encoded = df_encoded[self.expected_columns]
            
        return df_encoded, y
