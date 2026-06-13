import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds the Sprint 3 engineered features on top of the Sprint 1
    preprocessed columns:
      - fe__total_nights
      - fe__cancel_ratio
      - fe__lead_x_cancel
      - fe__was_waitlisted

    Designed to be used as a step inside an sklearn Pipeline, placed
    AFTER the preprocessor (ColumnTransformer) and BEFORE the model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['fe__total_nights'] = X['sc__stays_in_week_nights'] + X['sc__stays_in_weekend_nights']

        total_prev = X['sc__previous_cancellations'] + X['sc__previous_bookings_not_canceled']
        X['fe__cancel_ratio'] = X['sc__previous_cancellations'] / (total_prev + 1)

        X['fe__lead_x_cancel'] = X['sc__lead_time_log'] * X['fe__cancel_ratio']

        X['fe__was_waitlisted'] = (X['sc__days_in_waiting_list'] > 0).astype(int)

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects a fixed list of columns (the top-N features chosen in
    Sprint 3 via feature importance). Used as the final preprocessing
    step before the model in the Sprint 4 pipeline.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class RawPreprocessor(BaseEstimator, TransformerMixin):
    """
    Replicates the Sprint 1 transformations applied to the RAW dataframe
    BEFORE the ColumnTransformer (ct) was fit:

      - lead_time_log = log1p(lead_time)
      - adr_capped    = adr capped at the IQR-based upper bound
                        learned from the training data (adr_upper)
      - adds a placeholder 'is_canceled' column if missing
        (ct passes this through as `remainder__is_canceled`,
         but it is dropped before reaching the model)
      - selects exactly the 19 columns ct was fitted on
    """

    REQUIRED_COLUMNS = [
        'hotel', 'is_canceled', 'arrival_date_month',
        'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'country', 'market_segment',
        'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'reserved_room_type',
        'assigned_room_type', 'deposit_type', 'days_in_waiting_list',
        'customer_type', 'lead_time_log', 'adr_capped'
    ]

    def __init__(self, adr_upper):
        self.adr_upper = adr_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['lead_time_log'] = np.log1p(X['lead_time'])
        X['adr_capped'] = np.where(X['adr'] > self.adr_upper, self.adr_upper, X['adr'])

        if 'is_canceled' not in X.columns:
            X['is_canceled'] = 0

        if 'children' in X.columns:
            X['children'] = X['children'].astype(float)

        return X[self.REQUIRED_COLUMNS]


class ToDataFrame(BaseEstimator, TransformerMixin):
    """
    Converts the numpy/sparse output of a ColumnTransformer back into a
    pandas DataFrame with the given (already-deduplicated) column names.

    Stores `feature_names` as an instance attribute (not a closure), so
    this transformer is fully self-contained when pickled - no external
    globals (like a `preprocessor` variable) are needed at unpickle time.
    """

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return pd.DataFrame(X, columns=self.feature_names)
