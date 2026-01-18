"""LASSO Feature Selection Module"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from src.config import get_alpha_range, ALPHA_MIN, ALPHA_MAX, RANDOM_STATE
from src.logger import logger_inst


def run_lasso_aic_bic(X, y, criterion='bic'):
    """
    Run LASSO with automatic AIC/BIC selection.
    Uses LassoLarsIC which finds optimal alpha using information criterion.

    Args:
        X: features DataFrame
        y: target array
        criterion: 'aic' or 'bic'

    Returns:
        dict with optimal features, alpha, and model
    """
    logger_inst.info(f"Running LASSO with {criterion.upper()} criterion...")

    model = LassoLarsIC(criterion=criterion)
    model.fit(X, y)

    nonzero = np.abs(model.coef_) > 1e-10  # nonzero_mask
    selected_features = X.columns[nonzero].tolist()

    logger_inst.info(f"Optimal alpha ({criterion.upper()}): {model.alpha_:.6f}")
    logger_inst.info(f"Selected {len(selected_features)} features: {selected_features}")

    return {
        'features': selected_features,
        'n_features': len(selected_features),
        'alpha': model.alpha_,
        'coefficients': dict(zip(X.columns[nonzero], model.coef_[nonzero])),
        'model': model,
        f'{criterion}_path': model.criterion_  # Full AIC/BIC path
    }


def run_lasso_cv(X, y, n_alphas=100, cv=10):
    """
    Run LASSO with cross-validation to find optimal alpha.

    Args:
        X: features DataFrame
        y: target array
        n_alphas: number of alphas to try
        cv: number of CV folds

    Returns:
        dict with optimal features and CV results
    """
    logger_inst.info(f"Running LassoCV with {cv}-fold CV...")

    model = LassoCV(n_alphas=n_alphas, cv=cv, random_state=RANDOM_STATE, max_iter=100000)
    model.fit(X, y)

    nonzero_mask = np.abs(model.coef_) > 1e-10
    selected_features = X.columns[nonzero_mask].tolist()

    logger_inst.info(f"Optimal alpha (CV): {model.alpha_:.6f}")
    logger_inst.info(f"Selected {len(selected_features)} features: {selected_features}")

    return {
        'features': selected_features,
        'n_features': len(selected_features),
        'alpha': model.alpha_,
        'coefficients': dict(zip(X.columns[nonzero_mask], model.coef_[nonzero_mask])),
        'model': model,
        'alphas': model.alphas_,
        'mse_path': model.mse_path_  # MSE for each alpha
    }


def run_lasso_full_path(X, y, alphas=None):
    """
    Run LASSO across all alpha values to get all possible subsets.
    Args:
        X: features DataFrame
        y: target array
        alphas: array of alpha values to try

    Returns:
        dict mapping n_features -> subset info
    """
    if alphas is None:
        alphas = get_alpha_range()

    logger_inst.info(f"Running LASSO path with {len(alphas)} alpha values...")

    subsets = {}

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=100000, tol=1e-6)
        model.fit(X, y)

        nonzero = np.abs(model.coef_) > 1e-10
        n_feats = np.sum(nonzero)

        if n_feats > 0 and n_feats not in subsets:
            feat_names = X.columns[nonzero].tolist()
            coefs = model.coef_[nonzero]

            subsets[n_feats] = {
                'features': feat_names,
                'coefficients': dict(zip(feat_names, coefs)),
                'alpha': alpha
            }

    logger_inst.info(f"Found {len(subsets)} different subset sizes: {sorted(subsets.keys())}")

    return subsets


def get_coef_path(X, y, n_points=100):
    """Get coefficient path for plotting."""
    alphas = np.logspace(ALPHA_MIN, ALPHA_MAX, n_points)
    n_features = X.shape[1]
    coef_path = np.zeros((len(alphas), n_features))

    for i, a in enumerate(alphas):
        model = Lasso(alpha=a, max_iter=100000, tol=1e-6)
        model.fit(X, y)
        coef_path[i, :] = model.coef_

    return alphas, coef_path


def print_subsets(subsets, name="UPDRS"):
    """Log LASSO subsets."""
    logger_inst.info(f"\nLASSO subsets for {name}:")
    logger_inst.info(f"Found {len(subsets)} different sizes: {sorted(subsets.keys())}")

    for n in sorted(subsets.keys()):
        feats = subsets[n]['features']
        alpha = subsets[n]['alpha']
        logger_inst.info(f"  {n:2d} features (α={alpha:.6f}): {feats}")


def compare_selection_methods(X, y, target_name="UPDRS"):
    """
    Compare all LASSO selection methods and return results.
    """
    logger_inst.info(f"\n{'=' * 60}")
    logger_inst.info(f"COMPARING LASSO METHODS FOR {target_name}")
    logger_inst.info(f"{'=' * 60}")

    # Method 1: AIC
    aic_result = run_lasso_aic_bic(X, y, criterion='aic')

    # Method 2: BIC
    bic_result = run_lasso_aic_bic(X, y, criterion='bic')

    # Method 3: Cross-validation
    cv_result = run_lasso_cv(X, y)

    # Summary
    logger_inst.info(f"\n--- Summary for {target_name} ---")
    logger_inst.info(f"AIC: {aic_result['n_features']} features - {aic_result['features']}")
    logger_inst.info(f"BIC: {bic_result['n_features']} features - {bic_result['features']}")
    logger_inst.info(f"CV:  {cv_result['n_features']} features - {cv_result['features']}")

    return {
        'aic': aic_result,
        'bic': bic_result,
        'cv': cv_result
    }


def run_lasso_pipeline(input_path: Path, output_tables: Path) -> dict:
    """
    LASSO feature selection pipeline.
    Uses sklearn's built-in AIC/BIC (LassoLarsIC).
    """
    logger_inst.info("=" * 60)
    logger_inst.info("LASSO FEATURE SELECTION PIPELINE")
    logger_inst.info("=" * 60)

    df = pd.read_csv(input_path)
    features = [c for c in df.columns if c not in SKIP_COLS]

    X = df[features]
    y_motor = df["motor_UPDRS"].values
    y_total = df["total_UPDRS"].values

    results = {}

    # === Motor UPDRS ===
    logger_inst.info("\n" + "#" * 60)
    logger_inst.info("MOTOR UPDRS")
    logger_inst.info("#" * 60)

    motor_bic = run_lasso_aic_bic(X, y_motor, criterion='bic')
    motor_aic = run_lasso_aic_bic(X, y_motor, criterion='aic')
    motor_subsets = run_lasso_full_path(X, y_motor)
    print_subsets(motor_subsets, "Motor UPDRS")

    results['motor'] = {
        'bic': motor_bic,
        'aic': motor_aic,
        'subsets': motor_subsets
    }

    # === Total UPDRS ===
    logger_inst.info("\n" + "#" * 60)
    logger_inst.info("TOTAL UPDRS")
    logger_inst.info("#" * 60)

    total_bic = run_lasso_aic_bic(X, y_total, criterion='bic')
    total_aic = run_lasso_aic_bic(X, y_total, criterion='aic')
    total_subsets = run_lasso_full_path(X, y_total)
    print_subsets(total_subsets, "Total UPDRS")

    results['total'] = {
        'bic': total_bic,
        'aic': total_aic,
        'subsets': total_subsets
    }

    # === Final Summary ===
    logger_inst.info("\n" + "=" * 60)
    logger_inst.info("FINAL RESULTS (BIC)")
    logger_inst.info("=" * 60)

    logger_inst.info(f"\nMotor UPDRS: {motor_bic['n_features']} features")
    logger_inst.info(f"  Features: {motor_bic['features']}")
    logger_inst.info(f"  Alpha: {motor_bic['alpha']:.6f}")

    logger_inst.info(f"\nTotal UPDRS: {total_bic['n_features']} features")
    logger_inst.info(f"  Features: {total_bic['features']}")
    logger_inst.info(f"  Alpha: {total_bic['alpha']:.6f}")

    # Save to CSV
    summary = pd.DataFrame([
        {'target': 'motor_UPDRS', 'n_features': motor_bic['n_features'],
         'features': str(motor_bic['features']), 'alpha': motor_bic['alpha']},
        {'target': 'total_UPDRS', 'n_features': total_bic['n_features'],
         'features': str(total_bic['features']), 'alpha': total_bic['alpha']}
    ])
    summary.to_csv(output_tables / "lasso_results.csv", index=False)
    logger_inst.info(f"\nResults saved to {output_tables / 'lasso_results.csv'}")

    return results
