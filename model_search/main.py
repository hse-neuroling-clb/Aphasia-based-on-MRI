import argparse
import numpy as np
import pandas as pd

import os 
import yaml
import random

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from func import weighted_balanced_accuracy,fbeta_macro, remap_labels, create_pipeline, optimize_model_parameters, process_results

# Baseline Models
from sklearn.dummy import DummyClassifier

# Bayesian Models
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Linear Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

# Instance-Based Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

# Kernel-Based Models
from sklearn.svm import SVC

# Tree Models
from sklearn.tree import DecisionTreeClassifier

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Models
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Run model search for one configuration.")
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--scoring", default=None)
    parser.add_argument("--oversampling", choices=["true", "false"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--tabpfn-device", choices=["auto", "cpu", "cuda"], default=None)
    return parser.parse_args()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


CLI_ARGS = parse_args()
CONFIG_PATH = os.path.abspath(CLI_ARGS.config)
CONFIG_DIR = os.path.dirname(CONFIG_PATH)

# Load configurations
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

if CLI_ARGS.data_path is not None:
    config["data_path"] = CLI_ARGS.data_path
if CLI_ARGS.scoring is not None:
    config["scoring"] = CLI_ARGS.scoring
if CLI_ARGS.oversampling is not None:
    config["oversampling"] = parse_bool(CLI_ARGS.oversampling)
if CLI_ARGS.seed is not None:
    config["seed"] = CLI_ARGS.seed
if CLI_ARGS.n_trials is not None:
    config["number_of_trials"] = CLI_ARGS.n_trials
if CLI_ARGS.start is not None:
    config["start"] = CLI_ARGS.start
if CLI_ARGS.end is not None:
    config["end"] = CLI_ARGS.end
if CLI_ARGS.tabpfn_device is not None:
    config["tabpfn_device"] = CLI_ARGS.tabpfn_device


#-------------------------------------------------------------------------------#
#--------------------------------Data Loading-----------------------------------#
#-------------------------------------------------------------------------------#
data_path = config['data_path']
if not os.path.isabs(data_path):
    data_path = os.path.join(CONFIG_DIR, data_path)

data_path = os.path.abspath(data_path)

df = pd.read_csv(data_path)
demographic_features = df[['Gender', 'Age', 'Post onset']]
scan_feature_start = df.columns.get_loc('CinMid - GM')
scan_features = df.iloc[:, scan_feature_start:]
selected_features = list(demographic_features) + list(scan_features)
categorical_features = ['Gender']
numerical_features = ['Age', 'Post onset'] + list(scan_features)
X = df[selected_features].copy()
y = df['Aphasia_severity'].copy()


#-------------------------------------------------------------------------------#
#---------------------------------Base Config-----------------------------------#
#-------------------------------------------------------------------------------# 
combinations = config['combinations'][config['start']:config['end']]
seed = config['seed']
n_trials = config['number_of_trials']
all_models = config['model_selection']['all_models']
selected_models = config['model_selection']['selected_models']
tabpfn_device = config.get('tabpfn_device', 'auto')


#-------------------------------------------------------------------------------#
#-----------------------------Metrics Selection---------------------------------#
#-------------------------------------------------------------------------------#
weighted_ba_scorer = make_scorer(weighted_balanced_accuracy)
f0_5_macro = make_scorer(fbeta_macro, beta=0.5)
f2_macro = make_scorer(fbeta_macro, beta=2)
mcc = make_scorer(matthews_corrcoef)
cohen_kappa = make_scorer(cohen_kappa_score)

scoring_metrics_yaml = config['scoring_metrics']
scoring_metrics_default = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'weighted_balanced_accuracy': weighted_ba_scorer,
    'precision_macro': 'precision_macro',
    'precision_micro': 'precision_micro',
    'precision_weighted': 'precision_weighted',
    'recall_macro': 'recall_macro',
    'recall_micro': 'recall_micro',
    'recall_weighted': 'recall_weighted',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_weighted': 'f1_weighted',
    'f0.5_macro': f0_5_macro,
    'f2_macro': f2_macro,
    'mcc': mcc,  
    'cohen_kappa': cohen_kappa,  
    'roc_auc': 'roc_auc',
    'roc_auc_ovr': 'roc_auc_ovr',
    'roc_auc_ovo': 'roc_auc_ovo',
    'roc_auc_ovr_weighted': 'roc_auc_ovr_weighted',
    'roc_auc_ovo_weighted': 'roc_auc_ovo_weighted',
}
scoring_metrics = {}
for metric in scoring_metrics_yaml:
    if metric in scoring_metrics_default:
        scoring_metrics[metric] = scoring_metrics_default[metric]

scoring_goal = config['scoring']
scoring = scoring_metrics_default.get(scoring_goal)
if scoring is None:
    raise ValueError(f"Scoring metric '{scoring_goal}' not found in the default scoring metrics.")


def resolve_tabpfn_device(device):
    if device != 'auto':
        return device

    try:
        import torch
    except ImportError:
        return 'cpu'

    return 'cuda' if torch.cuda.is_available() else 'cpu'

#-------------------------------------------------------------------------------#
#---------------------------------CV Config-------------------------------------#
#-------------------------------------------------------------------------------#
cv = StratifiedKFold(
    n_splits=config['cross_validation']['n_splits'],
    shuffle=config['cross_validation']['shuffle'],
    random_state=seed
)


#-------------------------------------------------------------------------------#
#--------------------------------Experiment-------------------------------------#
#-------------------------------------------------------------------------------#
output_root = os.path.abspath(CLI_ARGS.output_root or config.get("output_root", BASE_DIR))
os.makedirs(output_root, exist_ok=True)
os.chdir(output_root)

EXP = CLI_ARGS.exp_name or ('Max_' + str(scoring_goal) + '_SMOTE='+ str(config['oversampling']) + '_Trails=' + str(n_trials) + '_Seed=' + str(seed))
os.makedirs(EXP, exist_ok=True)

if __name__ == "__main__":    
    np.random.seed(seed)
    random.seed(seed)

    if all_models or 'RandomLabeling' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RandomLabeling\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            strategy = trial.suggest_categorical('strategy', ['uniform'])

            #-------------------------------Model--------------------------------------#
            model = DummyClassifier(strategy=strategy, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)

            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DummyClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_RandomLabeling.csv')


    if all_models or 'MajorityVote' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing MajorityVote\n")
        print('='*80)
        
        def objective(trial, X, y, cv, random_state=42):

            strategy = trial.suggest_categorical('strategy', ['most_frequent','prior'])

            #-------------------------------Model--------------------------------------#
            model = DummyClassifier(strategy=strategy, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)

            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DummyClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
            
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_MajorityVote.csv')


    if all_models or 'DecisionStump' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing DecisionStump\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            max_depth = trial.suggest_int('max_depth', 1, 1)

            #-------------------------------Model--------------------------------------#
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DecisionTreeClassifier(max_depth=1, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_DecisionStump.csv')


    if all_models or 'NaiveBayes' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing NaiveBayes\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-1, log=True)

            #-------------------------------Model--------------------------------------#
            model = GaussianNB(
                var_smoothing=var_smoothing
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)    
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = GaussianNB(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')

    if all_models or 'QDA' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing QDA\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            reg_param = trial.suggest_float('reg_param', 1e-5, 1.0, log=True)

            #-------------------------------Model--------------------------------------#
            model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = QuadraticDiscriminantAnalysis(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')


    if all_models or 'KNN' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing KNN\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            n_neighbors = trial.suggest_int('n_neighbors', 1, 100, log=True)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 2)
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree','brute'])
            leaf_size = trial.suggest_int('leaf_size', 1, 100, log=True)

            #-------------------------------Model--------------------------------------#
            model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    p=p,
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = KNeighborsClassifier(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')


    if all_models or 'RKNN' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RKNN\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            radius = trial.suggest_float('radius', 1e-3, 10.0, log=True)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            leaf_size = trial.suggest_int('leaf_size', 1, 100, log=True)
            outlier_label=trial.suggest_categorical('outlier_label', ['most_frequent'])
            p = trial.suggest_int('p', 1, 2)

            #-------------------------------Model--------------------------------------#
            model = RadiusNeighborsClassifier(
                radius=radius,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                outlier_label=outlier_label,
                p=p
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RadiusNeighborsClassifier(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)


        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')



    if all_models or 'LR' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing LR\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            penalty = trial.suggest_categorical('penalty', ['l2'])
            C = trial.suggest_float('C', 1e-5, 10, log=True)
            max_iter = trial.suggest_int('max_iter', 1500, 2500)
            tol = trial.suggest_float('tol', 1e-5, 10.0, log=True)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                max_iter=max_iter,
                solver='lbfgs',
                tol=tol,
                class_weight=class_weight,
                multi_class='multinomial',
                random_state=random_state
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = LogisticRegression(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')

    if all_models or 'RR' in selected_models:

        #-------------------------------Model--------------------------------------#
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        class RidgeClassifierWithProba(RidgeClassifier):
            def predict_proba(self, X):
                try:
                    check_is_fitted(self)
                except NotFittedError as exc:
                    print(f"Model is not fitted yet.")
                    raise

                decision = self.decision_function(X)
                #print(f"Decision shape: {decision.shape}")
                #print(f"Decision first few values: {decision[:5]}")

                if decision.ndim == 1:
                    # Binary classification
                    decision = np.column_stack([-decision, decision])
                elif decision.ndim == 2:
                    # Multiclass classification
                    # Ensure decision.shape[1] equals the number of classes
                    if decision.shape[1] != len(self.classes_):
                        raise ValueError(f"Decision function output shape {decision.shape} does not match number of classes {len(self.classes_)}")
                else:
                    raise ValueError(f"Unexpected decision function output shape: {decision.shape}")

                #print(f"Adjusted decision shape: {decision.shape}")
                #print(f"Adjusted decision first few values:\n{decision[:5]}")

                proba = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                proba /= proba.sum(axis=1, keepdims=True)
                
                #print(f"Proba shape: {proba.shape}")
                #print(f"Proba first few values:\n{proba[:5]}")
                #print(f"Proba sum for first few samples: {proba[:5].sum(axis=1)}")

                return proba
        #--------------------------------------------------------------------------#

        print('='*80)
        print(f"\n{EXP} for Optimizing RR\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
            tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = RidgeClassifierWithProba(
                alpha=alpha,
                tol=tol,
                class_weight=class_weight,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#


            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RidgeClassifierWithProba(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')

    if all_models or 'SVC_linear' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_linear\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            gamma = trial.suggest_float('gamma', 1e-5, 10, log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            #shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])


            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    class_weight=class_weight,
                    probability=probability,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)    

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}_{kernel}.csv')

    if all_models or 'SVC_poly' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_poly\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['poly'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            #gamma = trial.suggest_categorical('gamma', ['scale','auto'])
            gamma = trial.suggest_float('gamma', 1e-2, 10.0,log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            #shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    #probability=probability,
                    max_iter=20000,
                    #class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}_{kernel}.csv')


    if all_models or 'SVC_sigmoid' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_sigmoid\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['sigmoid'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            #gamma = trial.suggest_categorical('gamma', ['scale','auto'])
            gamma = trial.suggest_float('gamma', 1e-2, 10.0,log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    probability=probability,
                    class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)    
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}_{kernel}.csv')


    if all_models or 'SVC_rbf' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_rbf\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10,log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf'])
            degree = trial.suggest_int('degree', 1, 5) if kernel == 'poly' else 3 # Only used for 'poly' kernel
            # gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            gamma = trial.suggest_float('gamma', 1e-5, 10, log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
            probability = trial.suggest_categorical('probability', [True])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    probability=probability,
                    #class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}_{kernel}.csv')



    if all_models or 'DT' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing DT\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            splitter = trial.suggest_categorical('splitter', ['best', 'random'])
            max_depth = trial.suggest_int('max_depth', 3, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
            min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 1.0)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])


            #-------------------------------Model--------------------------------------#
            model = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                #class_weight=class_weight,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#
            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DecisionTreeClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')



    if all_models or 'Adaboost' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing Adaboost\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 3.0, log=True)
            estimator = DecisionTreeClassifier(
                max_depth=trial.suggest_int('base_max_depth', 3, 50),
                criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
                splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 30),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 30),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 10, 1000, log=True),
                min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0.0, 1.0),
                class_weight=trial.suggest_categorical('class_weight', [None, 'balanced'])
            )

            #-------------------------------Model--------------------------------------#
            model = AdaBoostClassifier(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            ada_params = {
                'n_estimators': params['n_estimators'],
                'learning_rate': params['learning_rate']
            }
            dt_params = {
                'max_depth': params['base_max_depth'],
                'criterion': params['criterion'],
                'splitter': params['splitter'],
                'min_samples_split': params['min_samples_split'],
                'min_samples_leaf': params['min_samples_leaf'],
                'max_features': params['max_features'],
                'max_leaf_nodes': params['max_leaf_nodes'],
                'min_impurity_decrease': params['min_impurity_decrease'],
                #'class_weight': params['class_weight']
            }

            #-------------------------------------------------------------------------#
            estimator = DecisionTreeClassifier(**dt_params)
            model = AdaBoostClassifier(estimator=estimator, **ada_params, random_state=42)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')


    if all_models or 'RF' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RF\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_samples_split = trial.suggest_float('min_samples_split', 1e-5, 1.0, log=True)
            min_samples_leaf = trial.suggest_float('min_samples_leaf', 1e-5, 1.0, log=True)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000,log=True)

            #-------------------------------Model--------------------------------------#
            model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RandomForestClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')

    if all_models or 'MLP' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing MLP\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            activation = trial.suggest_categorical('activation', ['relu'])
            solver = trial.suggest_categorical('solver', ['adam'])
            alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [256])
            learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
            max_iter = trial.suggest_int('max_iter', 200, 1000)
            learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 2e-2, log=True)

            #-------------------------------Model--------------------------------------#
            model = MLPClassifier(
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    learning_rate_init=learning_rate_init,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = MLPClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')
    

    if all_models or 'XGB' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing XGB\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            param = {
                'objective': 'multi:softmax',  # For multiclass classification
                'eval_metric': 'mlogloss',  # Multiclass logloss
                'use_label_encoder': False,  # To avoid warning for deprecation
                'booster': trial.suggest_categorical('booster', ['gbtree']),
                'lambda': trial.suggest_float('lambda', 1e-5, 1.0, log=True),   
                'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),   
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500,log=True),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),   
             
            }
            
            #-------------------------------Model--------------------------------------#
            model = xgb.XGBClassifier(**param, random_state=random_state,tree_method = "hist", verbosity=1)
            #--------------------------------------------------------------------------#
            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)  

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]
            
            #-------------------------------------------------------------------------#
            model = xgb.XGBClassifier(**params,
                                      objective="multi:softmax",
                                      eval_metric="mlogloss",
                                      use_label_encoder=False , 
                                      random_state=seed,
                                      tree_method = "hist", 
                                      verbosity=1)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv') 


    if all_models or 'LGBM' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing LGBM\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            
            # LightGBM specific hyperparameters
            param = {
                'objective': 'multiclass',   
                'metric': 'multi_logloss',   
                'verbosity': -1,
                #'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                'num_leaves': trial.suggest_int('num_leaves', 100, 3000, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0,log=True),
                'subsample': trial.suggest_float('subsample', 1e-3, 1.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-3, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
            
            #-------------------------------Model--------------------------------------#
            model = lgb.LGBMClassifier(**param, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)  

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = lgb.LGBMClassifier(**params, objective='multiclass', metric='multi_logloss', verbosity=-1, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')
    
    if all_models or 'Tab' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing TabPFN\n")
        print('='*80)

        from tabpfn import TabPFNClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score

        current_tabpfn_device = resolve_tabpfn_device(tabpfn_device)

        def objective(trial, X, y, cv, scoring='accuracy', random_state=42):
            
            N_ensemble_configurations = trial.suggest_int('N_ensemble_configurations', 2, 64)
            
            scores = []
            for train_index, test_index in cv.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                #-------------------------------Model--------------------------------------#
                model = TabPFNClassifier(device=current_tabpfn_device, N_ensemble_configurations=N_ensemble_configurations)
                #--------------------------------------------------------------------------#

                pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif scoring == 'balanced_accuracy':
                    score = balanced_accuracy_score(y_test, y_pred)
                elif scoring == 'mcc':
                    score = matthews_corrcoef(y_test, y_pred)
                elif scoring == 'f1_weighted':
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")
                
                scores.append(score)
            
            return np.mean(scores)
        
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        
        

        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, 
                                recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, 
                                roc_auc_score)

        def compute_scores(pipeline, X, y_remapped, cv, scoring_metrics):
            all_scores = {metric: [] for metric in scoring_metrics.keys()}
            
            for train_index, test_index in cv.split(X, y_remapped):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y_remapped.iloc[train_index], y_remapped.iloc[test_index]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)
                
                for metric, scorer in scoring_metrics.items():
                    if metric in ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                        if metric == 'accuracy':
                            score = accuracy_score(y_test, y_pred)
                        elif metric == 'balanced_accuracy':
                            score = balanced_accuracy_score(y_test, y_pred)
                        elif metric == 'precision_weighted':
                            score = precision_score(y_test, y_pred, average='weighted')
                        elif metric == 'recall_weighted':
                            score = recall_score(y_test, y_pred, average='weighted')
                        elif metric == 'f1_weighted':
                            score = f1_score(y_test, y_pred, average='weighted')
                    elif metric == 'mcc':
                        score = matthews_corrcoef(y_test, y_pred)
                    elif metric == 'cohen_kappa':
                        score = cohen_kappa_score(y_test, y_pred)
                    elif metric in ['roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']:
                        score = roc_auc_score(y_test, y_pred_proba, multi_class=metric.split('_')[-2], average='weighted')
                    else:
                        raise ValueError(f"Unsupported scoring metric: {metric}")
                    
                    all_scores[metric].append(score)
            
            mean_scores = {metric: np.mean(scores) for metric, scores in all_scores.items()}
            std_scores = {metric: np.std(scores) for metric, scores in all_scores.items()}
            formatted_scores = {metric: f"{mean_scores[metric]:.3f} ± {std_scores[metric]:.3f}" for metric in scoring_metrics.keys()}
            
            return formatted_scores

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = TabPFNClassifier(**params, device=current_tabpfn_device)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features, use_smote=config['oversampling'],random_state=seed)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}/{EXP}_{model_name}.csv')





#-------------------------------------------------------------------------------#
#--------------------------------Experiment-------------------------------------#
#-------------------------------------------------------------------------------#
    print('='*80)
    print('='*80)
    print(f'\nExperiment {EXP} with {n_trials} trails is done\n')
    print('='*80)
    print('='*80)



