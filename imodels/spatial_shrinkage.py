from copy import deepcopy
from typing import Sequence, Optional

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text

from imodels.util import check_is_fitted, compute_tree_complexity


class SSTree:
    def __init__(
        self,
        estimator_: Optional[BaseEstimator] = None,
        reg_param: float = 1,
        fast: bool = False,
    ):
        """(Tree with spatial shrinkage applied).
        Spatial shinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest).
        It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its neighbours (using a single regularization parameter).
        Experiments over a wide variety of datasets show that spatial shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.

        Params
        ------
        estimator_: sklearn tree or tree ensemble model (e.g. RandomForest or GradientBoosting)
            Defaults to CART Classification Tree with 20 max leaf ndoes

        reg_param: float
            Higher is more regularization (can be arbitrarily large, should not be < 0)

        fast: bool
            Experimental: Used to experiment with different forms of shrinkage. options are:
                (i) True shrinks based on siblings only
                (ii) False shrinks based on all nodes
        """
        super().__init__()
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=20)
        self.reg_param = reg_param
        self.estimator_ = estimator_
        self.fast = fast
        if check_is_fitted(self.estimator_):
            self._shrink()

    def get_params(self, deep=True):
        params = {
            "reg_param": self.reg_param,
            "estimator_": self.estimator_,
            "fast": self.fast,
        }
        if deep:
            return deepcopy(params)
        else:
            return params

    def fit(self, X, y, *args, **kwargs):
        # remove feature_names if it exists (note: only works as keyword-arg)
        self.feature_names = kwargs.pop(
            "feature_names", None
        )  # None returned if not passed
        self.estimator_ = self.estimator_.fit(X, y, *args, **kwargs)

        self._shrink()

        # compute complexity
        if hasattr(self.estimator_, "tree_"):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, "estimators_"):
            self.complexity_ = 0
            for i in range(len(self.estimator_.estimators_)):
                t = deepcopy(self.estimator_.estimators_[i])
                if isinstance(t, np.ndarray):
                    assert t.size == 1, "multiple trees stored under tree_?"
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)
        return self

    def _shrink_tree(self, estimator, reg_param):
        """Shrink the tree"""
        if reg_param is None:
            reg_param = 1.0

        tree = estimator.tree_

        if isinstance(self, ClassifierMixin):
            tree.value[:] /= np.sum(tree.value, axis=-1, keepdims=True)
        value = np.copy(tree.value)

        root = 0
        root_left = tree.children_left[root]
        root_right = tree.children_right[root]
        if root_left == root_right == -1:
            return

        stack = []
        root_samples = tree.n_node_samples[root]
        root_feature = tree.feature[root]
        for node, sibling, sibling_on_right in (
            (root_left, root_right, True),
            (root_right, root_left, False),
        ):
            node_samples = tree.n_node_samples[node]
            node_weight = (root_samples**2 + node_samples * reg_param) / (
                root_samples**2 + root_samples * reg_param
            )
            sibling_list = [(sibling, sibling_on_right, 1 - node_weight, root_feature)]
            stack.append((node, node_weight, sibling_list, 0))

        while stack:
            (
                node,
                node_weight_multiplier,
                sibling_list,
                background,
            ) = stack.pop()
            node_samples = tree.n_node_samples[node]
            if self._is_leaf(tree, node):
                value[node] = node_weight_multiplier * tree.value[node]
                for sibling, _, sibling_weight_multiplier, _ in sibling_list:
                    value[node] += sibling_weight_multiplier * tree.value[sibling]
                value[node] += background

                continue

            node_feature = tree.feature[node]
            node_left = tree.children_left[node]
            node_right = tree.children_right[node]

            for child, child_sibling, child_sibling_on_right in (
                (node_left, node_right, True),
                (node_right, node_left, False),
            ):
                spinoff = 0
                child_sibling_list = []
                for sibling, sibling_on_right, sibling_weight_multiplier, sibling_ancestor_feature in sibling_list:
                    # absorbs sibling into background
                    if self.fast or self._is_leaf(tree, sibling):
                        spinoff += sibling_weight_multiplier * tree.value[sibling]
                        continue

                    # absorbs secondary part of sibling into background
                    sibling_feature = tree.feature[sibling]
                    if sibling_feature == node_feature != sibling_ancestor_feature:
                        signal = child_sibling_on_right
                    else:
                        signal = sibling_on_right

                    if signal:
                        sibling_primary = tree.children_left[sibling]
                        sibling_secondary = tree.children_right[sibling]
                    else:
                        sibling_primary = tree.children_right[sibling]
                        sibling_secondary = tree.children_left[sibling]

                    sibling_samples = tree.n_node_samples[sibling]
                    sibling_primary_samples = tree.n_node_samples[sibling_primary]
                    sibling_primary_weight = (
                        sibling_samples**2 + sibling_primary_samples * reg_param
                    ) / (sibling_samples**2 + sibling_samples * reg_param)
                    if self.fast:
                        spinoff += sibling_weight_multiplier * sibling_primary_weight * tree.value[sibling_primary]
                        spinoff += sibling_weight_multiplier * (1 - sibling_primary_weight) * tree.value[sibling_secondary]
                    else:
                        child_sibling_list.append((sibling_primary, sibling_on_right, sibling_weight_multiplier * sibling_primary_weight, sibling_ancestor_feature))
                        child_sibling_list.append((sibling_secondary, sibling_on_right, sibling_weight_multiplier * (1 - sibling_primary_weight), sibling_ancestor_feature))

                child_samples = tree.n_node_samples[child]
                child_weight = (node_samples**2 + child_samples * reg_param) / (
                    node_samples**2 + node_samples * reg_param
                )
                if self.fast:
                    spinoff += node_weight_multiplier * (1 - child_weight) * tree.value[child_sibling]
                else:
                    # create new siblings
                    child_sibling_list.append((child_sibling, child_sibling_on_right, node_weight_multiplier * (1 - child_weight), node_feature))

                stack.append(
                    (
                        child,
                        node_weight_multiplier * child_weight,
                        child_sibling_list,
                        background + spinoff,
                    )
                )

        tree.value[:] = value

    def _is_leaf(self, tree, node):
        return tree.children_left[node] == tree.children_right[node] == -1

    def _shrink(self):
        if hasattr(self.estimator_, "tree_"):
            self._shrink_tree(self.estimator_, self.reg_param)
        elif hasattr(self.estimator_, "estimators_"):
            for t in self.estimator_.estimators_:
                if isinstance(t, np.ndarray):
                    assert t.size == 1, "multiple trees stored under tree_?"
                    t = t[0]
                self._shrink_tree(t, self.reg_param)

    def _bounds(self, estimator):
        tree = estimator.tree_

        bounds = np.empty((tree.node_count, estimator.n_features_in_, 2))
        bounds[0, :, 0] = -np.inf
        bounds[0, :, 1] = np.inf

        root = 0
        stack = [root]
        while stack:
            node = stack.pop()

            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right == -1:
                continue

            feature = tree.feature[node]
            threshold = tree.threshold[node]

            bounds[left, :, :] = bounds[node, :, :]
            bounds[left, feature, 1] = threshold
            stack.append(left)

            bounds[right, :, :] = bounds[node, :, :]
            bounds[right, feature, 0] = threshold
            stack.append(right)

        return bounds

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X, *args, **kwargs)
        else:
            return NotImplemented

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, "score"):
            return self.estimator_.score(X, y, *args, **kwargs)
        else:
            return NotImplemented

    def __str__(self):
        s = "> ------------------------------\n"
        s += "> Decision Tree with Spatial Shrinkage\n"
        s += "> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n"
        s += "> ------------------------------" + "\n"
        if hasattr(self, "feature_names") and self.feature_names is not None:
            return s + export_text(
                self.estimator_, feature_names=self.feature_names, show_weights=True
            )
        else:
            return s + export_text(self.estimator_, show_weights=True)


class SSTreeRegressor(SSTree, RegressorMixin):
    ...


class SSTreeClassifier(SSTree, ClassifierMixin):
    ...


class SSTreeClassifierCV(SSTreeClassifier):
    def __init__(
        self,
        estimator_: Optional[BaseEstimator] = None,
        reg_param_list: Sequence[float] = (0.1, 1, 10, 50, 100, 500),
        fast: str = "node_based",
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs
    ):
        """Cross-validation is used to select the best regularization parameter for spatial shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passsed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None, *args, **kwargs)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.fast = fast
        # print('estimator', self.estimator_,
        #       'check_is_fitted(estimator)', check_is_fitted(self.estimator_))
        # if check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = SSTreeClassifier(deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y, *args, **kwargs)


class SSTreeRegressorCV(SSTreeRegressor):
    def __init__(
        self,
        estimator_: Optional[BaseEstimator] = None,
        reg_param_list: Sequence[float] = (0.1, 1, 10, 50, 100, 500),
        fast: str = "node_based",
        max_leaf_nodes: int = 20,
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs
    ):
        """Cross-validation is used to select the best regularization parameter for spatial shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passsed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None, *args, **kwargs)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.fast = fast
        # print('estimator', self.estimator_,
        #       'check_is_fitted(estimator)', check_is_fitted(self.estimator_))
        # if check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = SSTreeRegressor(deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y, *args, **kwargs)


if __name__ == "__main__":
    np.random.seed(15)
    # X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    # X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=10
    )
    print("X.shape", X.shape)
    print("ys", np.unique(y_train))

    # m = (estimator_=DecisionTreeClassifier(), reg_param=0.1)
    # m = DecisionTreeClassifier(max_leaf_nodes = 20,random_state=1, max_features=None)
    m = DecisionTreeRegressor(random_state=42, max_leaf_nodes=20)
    # print('best alpha', m.reg_param)
    m.fit(X_train, y_train)
    # m.predict_proba(X_train)  # just run this
    print("score", r2_score(y_test, m.predict(X_test)))
    print("running again....")

    # x = DecisionTreeRegressor(random_state = 42, ccp_alpha = 0.3)
    # x.fit(X_train,y_train)

    # m = (estimator_=DecisionTreeRegressor(random_state=42, max_features=None), reg_param=10)
    # m = (estimator_=DecisionTreeClassifier(random_state=42, max_features=None), reg_param=0)
    m = SSTreeClassifierCV(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=10, random_state=1),
        fast="node_based",
        reg_param_list=[0.1, 1, 2, 5, 10, 25, 50, 100, 500],
    )
    # m = ShrunkTreeCV(estimator_=DecisionTreeClassifier())

    # m = Classifier(estimator_ = GradientBoostingClassifier(random_state = 10),reg_param = 5)
    m.fit(X_train, y_train)
    print("best alpha", m.reg_param)
    # m.predict_proba(X_train)  # just run this
    # print('score', m.score(X_test, y_test))
    print("score", r2_score(y_test, m.predict(X_test)))
