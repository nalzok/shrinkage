from inspect import isclass

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.
    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.
    Parameters
    ----------
    estimator : estimator instance
        estimator instance for which the check is performed.
    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.
    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    fitted: bool
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        return all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        return estimator.__sklearn_is_fitted__()
    else:
        return (
            len(
                [
                    v
                    for v in vars(estimator)
                    if v.endswith("_") and not v.startswith("__")
                ]
            )
            > 0
        )


def compute_tree_complexity(tree, complexity_measure="num_rules"):
    """Calculate number of non-leaf nodes"""
    children_left = tree.children_left
    children_right = tree.children_right
    # num_split_nodes = 0
    complexity = 0
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            if complexity_measure == "num_rules":
                complexity += 1
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            if complexity_measure != "num_rules":
                complexity += 1
    return complexity


if __name__ == "__main__":
    X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    m = DecisionTreeRegressor(random_state=42, max_leaf_nodes=4)
    m.fit(X, y)
    print(compute_tree_complexity(m.tree_, complexity_measure="num_leaves"))
