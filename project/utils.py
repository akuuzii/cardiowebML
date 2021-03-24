import pandas as pd
from sklearn.model_selection import train_test_split



col_names = [
    'id', 'gender', 'age', 'smoking_ever', 'smoking_never', 'smoking_current', 'diabetes',
    'dyslipidaemia', 'hypertension', 'family_history', 'medical_history', 'mi_or_pci',
    'myocardial_infarction', 'pci', 'copd', 'pad', 'cva', 'length', 'weight', 'bmi',
    'pet_sss', 'pet_sds', 'ef_stress', 'ef_rest', 'lad_stress', 'lad_rest', 'lad_reserve',
    'lcx_stress', 'lcx_rest', 'lcx_reserve', 'rca_stress', 'rca_rest', 'rca_reserve',
    'total_stress', 'total_rest', 'total_mfr', '1year_obs_event'
]

col_names_cac = [
    'cac_total', 'cac_lm', 'cac_lad', 'cac_lcx', 'cac_rca'
]


def load_data(data_dir,dataset_type):
    if dataset_type == 'blanco_cac':
        cols = col_names_cac + col_names
    else:
        cols = col_names
    df = pd.read_excel(data_dir,engine='openpyxl',sheet_name=dataset_type,header=0,usecols=cols)
    x_train, x_test = train_test_split(df, test_size=0.2, random_state=123, stratify=df['1year_obs_event'])  # 42

    y_train = x_train['1year_obs_event'].copy()
    key_train = x_train['id'].copy()
    x_train = x_train.drop(['1year_obs_event','id'], axis=1)

    y_test = x_test['1year_obs_event'].copy()
    key_test = x_test['id'].copy()
    x_test = x_test.drop(['1year_obs_event','id'], axis=1)

    return x_train, y_train, x_test, y_test, key_train, key_test








