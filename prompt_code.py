import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def prepare_data_for_modeling(df):
    # Step 1: Remove unwanted columns
    df = df.drop(['region', 'VIN', 'state'], axis=1)

    # Step 2: Remove rows with missing values
    df_clean = df.dropna()

    # Step 3: Remove outliers (using IQR method for numerical columns)
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    numerical_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if col != 'price':  # Keeping 'price' as it's our target variable
            df_clean = remove_outliers(df_clean, col)

    # Step 4: Feature engineering
    # Create age feature
    current_year = pd.Timestamp.now().year
    df_clean['age'] = current_year - df_clean['year']

    # Create price_per_mile feature
    df_clean['price_per_mile'] = df_clean['price'] / (df_clean['odometer'] + 1)  # Adding 1 to avoid division by zero

    # Step 5: Encode categorical variables
    categorical_columns = df_clean.select_dtypes(include=['object']).columns

    # Use pd.get_dummies() with a more controlled approach
    for col in categorical_columns:
        # Get the top 10 most frequent categories
        top_categories = df_clean[col].value_counts().nlargest(10).index

        # Create dummy variables only for these top categories
        dummies = pd.get_dummies(df_clean[col].apply(lambda x: x if x in top_categories else 'Other'),
                                 prefix=col, drop_first=True)

        # Concatenate the dummy variables with the dataframe
        df_clean = pd.concat([df_clean, dummies], axis=1)

    # Drop the original categorical columns
    df_clean = df_clean.drop(categorical_columns, axis=1)

    # Step 6: Log transform the target variable (price)
    df_clean['log_price'] = np.log1p(df_clean['price'])

    # Step 7: Scale numerical features
    scaler = StandardScaler()
    numerical_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns = numerical_columns.drop(['price', 'log_price', 'id'])  # Exclude target and id
    df_clean[numerical_columns] = scaler.fit_transform(df_clean[numerical_columns])

    # Step 8: Final dataset preparation
    X = df_clean.drop(['price', 'log_price', 'id'], axis=1)
    y = df_clean['log_price']

    return X, y


def get_feature_importance(model, X):
    if hasattr(model, 'coef_'):
        # For linear models, including Ridge and Lasso
        if callable(model.coef_):
            coef = model.coef_()
        else:
            coef = model.coef_
        return pd.DataFrame({'Feature': X.columns, 'Importance': abs(coef)})
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models
        return pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    else:
        return pd.DataFrame({'Feature': X.columns, 'Importance': [np.nan] * len(X.columns)})


def build_and_evaluate_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Define hyperparameters for grid search
    param_grids = {
        'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso Regression': {'alpha': [0.1, 1.0, 10.0]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    }

    results = {}

    for name, model in models.items():
        print("\n====================================================")
        print(f"Evaluating {name}...")
        print("====================================================")

        if name in param_grids:
            # Perform grid search with cross-validation
            print(f"Fitting model via GridSearchCV...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")

        else:
            print(f"Fitting model via default...")
            best_model = model.fit(X_train, y_train)

        # Cross-validation
        print(f"Cross validating...")
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()

        # Predictions on test set
        print(f"Predicting on test set...")
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'Model': best_model,
            'CV MAE': cv_mae,
            'Test MAE': mae,
            'Test MSE': mse,
            'Test R2': r2
        }

        print(f"Cross-validation MAE: {cv_mae:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R2: {r2:.4f}")

        # Get feature importance
        # importance_df = get_feature_importance(model, X)
        importance_df = get_feature_importance(best_model, X)
        print("\nTop 5 most important features:")
        print(importance_df.sort_values('Importance', ascending=False).head())

    return results


# Load the dataset
vehicles_df = pd.read_csv('data/vehicles.csv')

X, y = prepare_data_for_modeling(vehicles_df)
results = build_and_evaluate_models(X, y)

# Find the best model
best_model = min(results, key=lambda x: results[x]['Test MAE'])
print(f"\nBest model based on Test MAE: {best_model}")
print(f"Test MAE: {results[best_model]['Test MAE']:.4f}")

best_model = min(results, key=lambda x: results[x]['Test MSE'])
print(f"\nBest model based on Test MSE: {best_model}")
print(f"Test MSE: {results[best_model]['Test MSE']:.4f}")
