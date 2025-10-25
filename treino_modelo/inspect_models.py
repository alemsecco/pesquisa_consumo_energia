import joblib
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(base, 'treino_modelo', 'modelo')

candidates = ['rfr.joblib','rfr_tuned.joblib','rfr_global.joblib']
for name in candidates:
    path = os.path.join(base, 'treino_modelo', 'modelo', name)
    if not os.path.exists(path):
        print(name, 'NOT FOUND at', path)
        continue
    print('\nLoading', path)
    obj = joblib.load(path)
    print('Type:', type(obj))
    # If it's a Pipeline, show steps
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(obj, Pipeline):
            print('Pipeline steps:', obj.named_steps.keys())
            # show preprocessor type
            if 'preprocessor' in obj.named_steps:
                print('preprocessor type:', type(obj.named_steps['preprocessor']))
            if 'regressor' in obj.named_steps:
                print('regressor type:', type(obj.named_steps['regressor']))
    except Exception as e:
        print('Could not introspect pipeline:', e)
    # show attributes that can hint expected features
    for attr in ['feature_names_in_', 'n_features_in_']:
        if hasattr(obj, attr):
            print(attr, getattr(obj, attr))
    # print repr summary (short)
    print('repr:', repr(obj)[:500])
