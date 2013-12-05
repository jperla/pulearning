from sklearn.grid_search import GridSearchCV
from sklearn.base import clone

class FastGridSearchCV(GridSearchCV):
    """Wraps Sci-Kits GridSearchCV so that it uses a different class for
            the search than for the best estimator.
        This is useful when one implementation is faster than another
            but use the same parameters. Specifically, this was
            created for LinearSVC() vs SVC(kernel='linear').
    """
    def __init__(self, estimator, best_fit_estimator, param_grid, **kwargs):
        self.best_fit_estimator = best_fit_estimator
        super(FastGridSearchCV, self).__init__(estimator, param_grid, **kwargs)

    def _fit(self, X, y, parameter_iterable, **keywords):
        super(FastGridSearchCV, self)._fit(X, y, parameter_iterable, **keywords)

        if self.refit:
            best_estimator = clone(self.best_fit_estimator).set_params(
                **self.best_params_)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

