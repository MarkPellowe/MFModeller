import numpy as np

from mfmodeller.test_functions.base_test_function import (
    BaseTestFunction,
    FidelityFunction,
)
from mfmodeller.test_functions.helpers import FloatArray


class ForresterFunction(BaseTestFunction):
    r"""
    The Forrester function is a classic one-dimensional test function used in
    optimisation and surrogate modelling. It provides both low-fidelity and
    high-fidelity versions to test multi-fidelity optimisation algorithms.

    **High-fidelity function:**

    .. math::

        f_1(x) = [6x - 2]^2 \sin(12x - 4)

    **Low-fidelity function:**

    .. math::

        f_0(x) = 0.5 [6x - 2]^2 \sin(12x - 4) + 10(x - 0.5) - 5
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ndim = 1
        self._n_fidelities = 2
        self._correlation_adjustment = None
        fidelity_dictionary: dict[int, FidelityFunction] = {
            0: self.fidelity0_function,
            1: self.fidelity1_function,
        }
        self._fidelity_dictionary = fidelity_dictionary
        self._bounds = [(0, 1)]
        self._standard_samples = [21, 4]
        self._name = "Forrester"

    def fidelity0_function(self, x: FloatArray, noise_level: float = 0.0) -> FloatArray:
        output = 0.5 * np.square(6 * x - 2) * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5
        return self._output_maker(output, noise_level)

    def fidelity1_function(
        self, x: FloatArray, noise_level: float = 0.01
    ) -> FloatArray:
        output = np.square(6 * x - 2) * np.sin(12 * x - 4)
        return self._output_maker(output, noise_level)


class Heterogeneous2D(BaseTestFunction):
    """
    The Heterogeneous2D function is a two-dimensional function combining different
     behaviors, used to test algorithms in modeling complex, heterogeneous data.

    **High-fidelity function:**

    .. math::

        f_1(x) = \\sin\\left(21 (x_1 - 0.9)^4\right)
        \\cos\\left(2 (x_1 - 0.9)\right) + \\dfrac{x_1 - 0.7}{2}
        + 2 x_2^2 \\sin(x_1 x_2)

    **Low-fidelity function:**

    .. math::

        f_0(x) = \\dfrac{f_1(x) - 2(x_1 + x_2)}{5 + 0.25 x_1 + 0.5 x_2}
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ndim = 2
        self._n_fidelities = 2
        self._fidelity_dictionary = {
            0: self.fidelity0_function,
            1: self.fidelity1_function,
        }
        self._bounds = [(0, 1), (0, 1)]
        self._name = "Heterogeneous2D"
        self._correlation_adjustment = None
        self._standard_samples = [32, 16]

    def fidelity0_function(self, x: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
        yh_output = self.fidelity1_function(x, noise_level=0.0)[:, 0]
        term1 = yh_output - 2 * np.sum(x, axis=1)
        term2 = 5 + 0.25 * x[:, 0] + 0.5 * x[:, 1]
        output = term1 / term2
        return self._output_maker(output, noise_level)

    def fidelity1_function(self, x: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
        term_1 = (
            np.sin(21 * np.power(x[:, 0] - 0.9, 4)) * np.cos(2 * (x[:, 0] - 0.9))
            + (x[:, 0] - 0.7) / 2
        )
        term_2 = 2 * np.square(x[:, 1]) * np.sin(np.prod(x, axis=1))
        output = term_1 + term_2
        return self._output_maker(output, noise_level)
