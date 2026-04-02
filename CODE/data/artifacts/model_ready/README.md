
        # CRITICAL WARNING FOR MODELING TEAM - DIRECTIVE 048

        The attached artifacts (`Amaryllis_ModelReady_*.csv`) have been generated with imputation and scaling applied to the **entire dataset** for the purpose of creating a complete, final data product.

        **UNDER NO CIRCUMSTANCES** are you to replicate this process in the live modeling pipeline. Doing so will cause catastrophic data leakage from the test set into the training set and will invalidate all performance metrics.

        You are **MANDATED** to use a `sklearn.pipeline.Pipeline` object for all model training and evaluation. The `SimpleImputer` (with a 'median' strategy) and `StandardScaler` steps **MUST** be **fit** *only* on the training data partition (`X_train`). These fitted transformers must then be used to **transform** both the training data (`X_train`) and, separately, the testing data (`X_test`).

        Failure to adhere to this `fit-then-transform` paradigm is a critical architectural violation and will guarantee project failure.
        