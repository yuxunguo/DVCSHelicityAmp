"""Shared objective transforms for the gradient-search processes.

The gradient workflow is implemented as a minimizer.  Pairwise concurrence is
instead maximized, so those scans minimize the bounded loss ``1 - C`` while
retaining the original concurrence column in every evaluated phase-space row.
"""

import numpy as np


PAIRWISE_CONCURRENCE_NAMES = (
    "C_e_p",
    "C_e_gamma",
    "C_p_gamma",
)
ONE_MINUS_PREFIX = "one_minus_"


def source_observable_name(objective_name):
    """Return the physical row observable underlying one search objective."""
    objective_name = str(objective_name)
    if objective_name.startswith(ONE_MINUS_PREFIX):
        source_name = objective_name[len(ONE_MINUS_PREFIX):]
        if source_name in PAIRWISE_CONCURRENCE_NAMES:
            return source_name
    return objective_name


def objective_value(row, column_prefix, objective_name, *, store=False):
    """Return the minimization value, optionally storing a derived column."""
    objective_name = str(objective_name)
    source_name = source_observable_name(objective_name)
    source_key = f"{column_prefix}_{source_name}"
    source_value = float(row.get(source_key, np.nan))
    if source_name != objective_name:
        value = 1.0 - source_value
    else:
        value = source_value
    if store:
        row[f"{column_prefix}_{objective_name}"] = value
    return value
