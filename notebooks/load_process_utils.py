#                        IMPORTS
import dsi #De Simone Inference module
import itertools
import time
import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


#                        CONSTANTS
RANDOM_STATE = int('8'*8)
DARK_MODE = True
VAL_SIZE = 0.15
TEST_SIZE = 15/85
LINES_SEPARATOR = "_.0"+"-<<O>>-"*15+"._"


def load_dataset_from_task(task_id):
    """
    Load a CTR23 dataset using its OpenML task ID.
    Returns df and target column name.
    """

    # Load the task
    task = openml.tasks.get_task(task_id)

    # Extract the dataset associated with the task
    dataset = task.get_dataset()
    
    # Name retrival
    task_name = dataset.name
    
    # Identify the target column
    target_name = dataset.default_target_attribute

    # Load data as pandas DataFrame
    X, y, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    # Build final DataFrame
    X[target_name] = y

    # Dimentions
    n_rows, n_cols = X.shape
    
    dataset_info = {
        "task_name" : task_name,
        "dataset_size" : (n_rows, n_cols)
    }
    
    print(f"Task: {task_name}")
    print(f"Dataset size: {n_rows} rows × {n_cols} columns")
    
    return X, target_name, dataset_info


def split_convert_dataset(df, shuffle=True):
    """
    Splits the dataset and converts to array-like float64.
    Returns train, validation and test arrays.
    """
    
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        shuffle=shuffle,
        random_state=RANDOM_STATE,
    )

    df_train, df_val = train_test_split(
        df_train,
        test_size=VAL_SIZE,
        shuffle=shuffle,
        random_state=RANDOM_STATE,
    )
    
    # Convert to numpy float64
    train_arr = df_train.to_numpy(dtype=np.float64)
    val_arr   = df_val.to_numpy(dtype=np.float64)
    test_arr  = df_test.to_numpy(dtype=np.float64)

    return train_arr, val_arr, test_arr


def handle_numeric_nan(df):
    """
    Checks numeric columns for NaN values and asks the user how to handle them.
    Options:
        A: Remove numeric columns containing NaN
        B: Remove rows containing NaN
        C: Abort the regression algorithm
    Returns:
        A cleaned DataFrame or raises an exception.
    """

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Find numeric columns containing NaN
    nan_cols = [col for col in numeric_cols if df[col].isna().any()]

    # If no NaN, return immediately
    if not nan_cols:
        return df

    # Inform the user
    print("\nWarning: Numeric columns contain NaN values:")
    print(f"   → {nan_cols}\n")

    print("Choose how to handle NaN values:")
    print("A: Remove numeric columns containing NaN")
    print("B: Remove rows containing NaN")
    print("C: Abort the regression algorithm")

    choice = input("\nEnter your choice (A/B/C): ").strip().upper()

    # Option A: drop columns
    if choice == "A":
        print(f"\nRemoving columns: {nan_cols}")
        return df.drop(columns=nan_cols)

    # Option B: drop rows
    elif choice == "B":
        print("\nRemoving rows containing NaN in numeric columns")
        return df.dropna(subset=numeric_cols)

    # Option C: abort
    elif choice == "C":
        raise ValueError(
            f"Regression aborted because numeric columns contain NaN: {nan_cols}"
        )

    else:
        raise ValueError("Invalid choice. Expected A, B, or C.")
    

def collapse_rare_categories(df, max_levels=1000):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns #Prende le colonne

    for col in cat_cols:

        # Ensure "__MISSING__" is a valid category if column is categorical
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "__MISSING__" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["__MISSING__"])
                    
        # Introduce a special category where data is missing simil to ctr23 pipeline
        df[col] = df[col].fillna("__MISSING__")
        
        counts = df[col].value_counts()

        if len(counts) <= max_levels:
            continue

        keep = counts.nlargest(max_levels).index
        df[col] = df[col].where(df[col].isin(keep), "__OTHER__")

        df[col] = df[col].astype("category")

    return df


def one_hot_encode(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(cat_cols) == 0:
        return df

    df[cat_cols] = df[cat_cols].fillna("__MISSING__")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cat_cols])

    encoded_df = (
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        .astype("int8")
        .reset_index(drop=True)
    )

    df = df.drop(columns=cat_cols).reset_index(drop=True)

    return pd.concat([df, encoded_df], axis=1)


def preprocess_ctr23_style(df, max_levels=1000):
    df = df.copy()

    # Save original categorical columns
    original_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df = collapse_rare_categories(df, max_levels=max_levels)
    df = one_hot_encode(df)

    return df, original_cat_cols


def move_target_last(df, target_name):
    target_col = df[target_name]
    df = df.drop(columns=[target_name])
    df[target_name] = target_col
    return df


def generate_unique_list(df_encoded, original_cat_cols):
    unique_list = []

    for col in df_encoded.columns:
        # If the column name starts with any original categorical column name -> it's one-hot
        is_one_hot = any(col.startswith(cat + "_") for cat in original_cat_cols)
        unique_list.append(is_one_hot)

    return unique_list


def drop_col_with_no_uniques(df):
    """
    Drops columns of dataframe that have not more than 1 different value
    """
    removed = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=removed)
    print("Dropped constant columns:", removed)
    
    return df


def load_preprocess_split(task_id):
    
    # Load dataset
    df, target_name, dataset_info = load_dataset_from_task(task_id)

    # Drop columns with 1 categorical levels
    df = drop_col_with_no_uniques(df)

    # Abort or clean numeric NaN before any preprocessing
    df = handle_numeric_nan(df)

    # Preprocess CTR23-style
    df, original_cat_cols = preprocess_ctr23_style(df)

    # Move target to last column (your optimized idea)
    df = move_target_last(df, target_name)

    # Generate unique_list
    unique_list = generate_unique_list(df, original_cat_cols)

    # Split e convert
    train_arr, val_arr, test_arr = split_convert_dataset(df)

    return train_arr, val_arr, test_arr, unique_list, dataset_info


def generate_configurations(unique_list):
    """
    Generate all 8 combinations of DSI flags.

    parameters
    ----------
    unique_list: list
        List of bool: True where operator forces the division of feature domain
        on the unique values contained. False where the approach follows density
        based or linear feature domain division.

    returns
    -------
    configs: dict
        Various configuration to test the model.
    """

    configs = []

    fire_options = [True, False]
    fuzzy_options = [True, False]
    db_options = [True, False]

    for fire, fuzzy, db in itertools.product(fire_options, fuzzy_options, db_options):

        # Base division depending on DB flag
        if db:
            division = ["D"] * len(unique_list)
        else:
            division = ["L"] * len(unique_list)

        # Enforce class-based division
        for i, is_unique in enumerate(unique_list):
            if is_unique:
                division[i] = "U"

        configs.append({
            "fire_of_train_rules": fire,
            "fuzzy_output": fuzzy,
            "default_division": division,
        })

    return configs


def regression_test(df_train, df_val, df_test, configurations, n_of_sets=None, print_pivots_debug=False):
    """
    Test all 8 configuration for regression task solved by De Simone Inference
    """
    results = []
    
    for config in configurations:

        print("Starting optimizzation for:\n",config)
        
        model = dsi.DSregressor(**config)
        gt, fst, sst, mtt = model.fit_cross_validation(df_train,
                                                       df_val,
                                                       n_of_sets=n_of_sets,
                                                       domains_functions_list=config["default_division"],
                                                       print_pivots_debug=print_pivots_debug)

        # --- Predizione ---
        y_out = model.predict(df_test[:, :-1])
        y_true = df_test[:, -1]
        y_pred = y_out.get()

        # --- Gestione predizioni mancanti ---
        valid_mask = ~np.isnan(y_pred)
        cov = valid_mask.mean()   # coverage

        if valid_mask.sum() == 0:
            y_pred_valid = np.array([])
            y_true_valid = np.array([])
        else:
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]

        # --- Entry base ---
        entry = {
            "fire_of_train_rules": config["fire_of_train_rules"],
            "fuzzy_output": config["fuzzy_output"],
            "default_division": config["default_division"],
            "cov": cov,
            "global_opt_time": gt,
            "first_step_time": fst,
            "second_step_time": sst,
            "mean_train_time": mtt,
        }

        # --- Score ---
        if len(y_pred_valid) > 0:
            entry["R2"] = dsi.r2(y_true_valid, y_pred_valid).get()
            entry["MAE"] = dsi.mae(y_true_valid, y_pred_valid).get()
            entry["RMSE"] = dsi.rmse(y_true_valid, y_pred_valid).get()
        else:
            entry["r2"] = np.nan
            entry["MAE"] = np.nan
            entry["RMSE"] = np.nan

        results.append(entry)

    return results


# Markdown conversion
#==========================================================================
def results_to_latex_table(results, df_info):
    """
    Conversion fo results for Latex pubblication
    """

    # Sort by RMSE values
    results_sorted = sorted(results, key=lambda x: float(x["RMSE"]))

    # Benig table
    latex = (
        "\\begin{table}[H]\n"
    )

    # Info conversion
    name = df_info.get("task_name", "unknown")
    name_for_title = name.replace("_"," ").capitalize()
    rows, cols = df_info.get("dataset_size", ("?", "?"))
    n_cat = df_info.get("n_of_cat_features", "?")

    # Adding info
    latex += (
        f"\\caption {{{name_for_title}; Size ${rows} \\times {cols} $; Categorical. Col. {n_cat}}}\n"
        f"\\label{{tab:{name}}}\n"
    )

    latex += (
        "\\centering\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{c c c c c c c c}\n"
        "\\hline\n"
        "DIV & RS & DEF & RMSE & R2 & COV & OPT(s) & MT(s) \\\\\n"
        "\\hline\n"
    )
    
    for entry in results_sorted:

        # Division
        div = "LIN" if entry["default_division"][0] == "L" else "DEN"

        # Rule Selection
        rule_sel = "WM" if entry["fuzzy_output"] is False else "ENH"

        # Defuzzyfication
        defuzz = "SUG" if entry["fire_of_train_rules"] is False else "HYB"

        # Metrics
        rmse = float(entry["RMSE"])
        r2   = float(entry["R2"])
        cov  = float(entry["cov"])
        gopt = float(entry["global_opt_time"])
        mtrain = float(entry["mean_train_time"])

        latex += (
            f"{div} & {rule_sel} & {defuzz} & "
            f"{rmse:.4e} & {r2:.4f} & {cov:.4f} & {gopt:.4f} & {mtrain:.4e} \\\\\n"
        )

    latex += (
        "\\hline\n"
        "\\end{tabular}%\n"
        "} % end resizebox\n"
        "\\end{table}\n"
    )

    print(latex)



# Final routine
#==========================================================================
def scoring_routine(task_id, print_table = True, n_of_sets=None, print_pivots_debug=False):
    """
    The whole scoring routines, outputs scoring results and separatey dataset info dict
    
    parameters
    ----------
    task_id_ int
        Id given is used to provide datasate on which coring routine is performed
    print_table: Bool, optional
        Enabe (disable) automaic Latex print.
    n_of_sets_override: int or array like
        Ovverrides the initial number of sets per feature.
    """
    df_train, df_val, df_test, unique_list, df_info = load_preprocess_split(task_id)
    configs = generate_configurations(unique_list)
    df_info["n_of_cat_features"] = np.asarray(unique_list).sum() 
    
    results = regression_test(df_train, df_val, df_test, configs,
                              n_of_sets=n_of_sets, print_pivots_debug=print_pivots_debug)

    if print_table:
        print(LINES_SEPARATOR+"\n")
        results_to_latex_table(results, df_info)
        print(LINES_SEPARATOR+"\n")
    return results, df_info

    