import logging
from abc import ABC, abstractmethod
from typing import Protocol, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from .helpers import x_test_grid_gen, x_test_mesh_flatten, x_test_mesh_gen

FloatArray = NDArray[np.float64]
Bounds = list[tuple[float, float]]
X_TEST_SAMPLES: int = 1000

logger = logging.getLogger(__name__)


class FidelityFunction(Protocol):
    """Callable signature for fidelity-specific evaluators."""

    def __call__(
        self,
        x: FloatArray,
        noise_level: float | None = None,
    ) -> FloatArray: ...


class BaseTestFunction(ABC):
    """Base class for all test functions."""

    _bounds: Bounds
    _correlation_adjustment: float | None
    _fidelity_dictionary: dict[int, FidelityFunction]
    _global_maximum: float | None
    _global_minimum: float | None
    _name: str
    _n_fidelities: int
    _ndim: int
    _standard_samples: list[int]
    _x_test_size: int = X_TEST_SAMPLES

    @abstractmethod
    def __init__(
        self,
        correlation_adjustment: float | None = None,
        **kwargs,
    ) -> None:
        """Initialise the common state for a test function."""
        self._correlation_adjustment = correlation_adjustment
        self._global_minimum = None
        self._global_maximum = None
        self._standard_samples = list(
            kwargs.get(
                "standard_samples",
                getattr(self, "_standard_samples", []),
            )
        )
        self._name = str(kwargs.get("name", self.__class__.__name__))
        self._bounds = list(kwargs.get("bounds", getattr(self, "_bounds", [])))
        self._n_fidelities = int(
            kwargs.get("n_fidelities", getattr(self, "_n_fidelities", 0))
        )
        self._ndim = int(
            kwargs.get("ndim", getattr(self, "_ndim", len(self._bounds)))
        )
        self._fidelity_dictionary = dict(
            kwargs.get(
                "fidelity_dictionary",
                getattr(self, "_fidelity_dictionary", {}),
            )
        )

    def _coerce_input_array(self, x: np.ndarray | int) -> FloatArray:
        """Convert supported input values into a 2D float array."""
        if isinstance(x, int):
            if self.ndim != 1:
                raise TypeError(
                    "Given an integer input, but the function expects "
                    f"{self.ndim} dimensions."
                )
            return np.array([[x]], dtype=float)

        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        return x_array

    @overload
    def evaluate(
        self,
        x: np.ndarray | int,
        fidelity: int,
        return_x_with_fidelity: bool = False,
        noise_level: float | None = None,
    ) -> FloatArray: ...

    @overload
    def evaluate(
        self,
        x: np.ndarray | int,
        fidelity: int,
        return_x_with_fidelity: bool = False,
        noise_level: float | None = None,
    ) -> tuple[FloatArray, FloatArray]: ...

    def evaluate(
        self,
        x: np.ndarray | int,
        fidelity: int,
        return_x_with_fidelity: bool = False,
        noise_level: float | None = None,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Evaluate the function at a given fidelity level."""
        evaluation_function = self.fidelity_dictionary.get(fidelity)
        if evaluation_function is None:
            raise ValueError(
                f"Invalid fidelity value: {fidelity}. "
                f"Valid values are: {list(self.fidelity_dictionary.keys())}"
            )

        x_array = self._coerce_input_array(x)
        y = (
            evaluation_function(x_array, noise_level)
            if noise_level is not None
            else evaluation_function(x_array)
        )

        if not return_x_with_fidelity:
            return y

        if x_array.shape[-1] != self.ndim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.ndim}, "
                f"got {x_array.shape[-1]}."
            )

        x_output = np.column_stack(
            (x_array, np.full(x_array.shape[0], fidelity, dtype=float))
        )
        return x_output, y

    def find_optimum(
        self,
        maximise: bool = False,
        number_of_searches: int = 50,
        bounds: Bounds | None = None,
        print_location_and_value: bool = False,
        fidelity: int | None = None,
    ) -> float:
        """Find the minimum or maximum value of the function."""
        if bounds is None:
            bounds = self.bounds

        if fidelity is None:
            fidelity = self.n_fidelities - 1

        lower_bounds = np.array([lower for lower, _ in bounds], dtype=float)
        upper_bounds = np.array([upper for _, upper in bounds], dtype=float)
        rng = np.random.default_rng()

        def scipy_func(x_values: np.ndarray) -> float:
            value = float(
                self.evaluate(
                    x=x_values,
                    fidelity=fidelity,
                    noise_level=0.0,
                )[0, 0]
            )
            return -value if maximise else value

        search_array = np.zeros(
            (number_of_searches, self.ndim + 1),
            dtype=float,
        )
        for index in range(number_of_searches):
            initial_location = rng.random((1, self.ndim))
            initial_location = (
                lower_bounds
                + (initial_location * (upper_bounds - lower_bounds))
            ).reshape(1, self.ndim)

            result = minimize(
                scipy_func,
                x0=initial_location[0],
                options={"disp": False},
                bounds=bounds,
            )
            if result is None or result.x is None or result.fun is None:
                raise RuntimeError(
                    "scipy.optimize.minimize returned an incomplete result."
                )

            result_x = np.asarray(result.x, dtype=float)
            result_fun = np.asarray(result.fun, dtype=float)
            search_array[index, :-1] = result_x
            search_array[index, -1] = result_fun.reshape(-1)[0]

        objective_values = search_array[:, -1]
        best_index = int(np.argmin(objective_values))
        optimal_value = float(objective_values[best_index])
        if maximise:
            optimal_value = -optimal_value

        if print_location_and_value:
            logger.info("Optimal location: %s", search_array[best_index, :-1])
            logger.info("Optimal value: %s", optimal_value)

        return optimal_value

    def get_random_samples(
        self,
        num_samples: int,
        fidelity: int | None = None,
        bounds: Bounds | None = None,
        sampler_cls: type[qmc.QMCEngine] = qmc.LatinHypercube,
        seed: int | None = None,
        return_x_with_fidelity: bool = False,
    ) -> tuple[FloatArray, FloatArray]:
        """Return random samples and function values."""
        if fidelity is None:
            fidelity = self.n_fidelities - 1

        if bounds is None:
            bounds = self.bounds

        lower_bounds = [lower for lower, _ in bounds]
        upper_bounds = [upper for _, upper in bounds]

        sampler = sampler_cls(d=self.ndim, seed=seed)
        samples = np.asarray(sampler.random(n=num_samples), dtype=float)
        samples = np.asarray(
            qmc.scale(samples, lower_bounds, upper_bounds),
            dtype=float,
        )
        outputs = self.evaluate(samples, fidelity)

        if return_x_with_fidelity:
            samples = np.column_stack(
                (samples, np.full(num_samples, fidelity, dtype=float))
            )

        return samples, outputs

    def initialise(
        self,
        fidelity_samples: list[int] | None = None,
        sampler_cls: type[qmc.QMCEngine] = qmc.LatinHypercube,
        bounds: Bounds | None = None,
        seed: int | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Initialise the function with random samples at each fidelity."""
        if fidelity_samples is None:
            fidelity_samples = self.standard_samples

        inputs: list[FloatArray] = []
        outputs: list[FloatArray] = []
        for fidelity, fidelity_sample_num in enumerate(fidelity_samples):
            inputs_f, outputs_f = self.get_random_samples(
                fidelity_sample_num,
                fidelity=fidelity,
                bounds=bounds,
                sampler_cls=sampler_cls,
                seed=seed,
                return_x_with_fidelity=True,
            )
            inputs.append(inputs_f)
            outputs.append(outputs_f)

        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    @overload
    def get_true_y(
        self,
        fidelity: int | None = None,
        return_with_x: bool = False,
        contour_grid: bool = False,
    ) -> FloatArray: ...

    @overload
    def get_true_y(
        self,
        fidelity: int | None = None,
        return_with_x: bool = False,
        contour_grid: bool = False,
    ) -> tuple[FloatArray, FloatArray]: ...

    def get_true_y(
        self,
        fidelity: int | None = None,
        return_with_x: bool = False,
        contour_grid: bool = False,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Return the true value of the function at a given fidelity."""
        if fidelity is None:
            fidelity = self.n_fidelities - 1

        if contour_grid:
            test_x = x_test_mesh_flatten(
                x_test_mesh_gen(self.ndim, self.bounds, self.x_test_size)
            )
        else:
            test_x = self._x_test_grid(seed=0)

        return self.evaluate(
            test_x,
            fidelity=fidelity,
            return_x_with_fidelity=return_with_x,
        )

    def get_x_test_all_fidelities(
        self,
        num_samples: int = 1000,
        seed: int = 0,
    ) -> FloatArray:
        """Generate a test grid for every fidelity."""
        old_x_test_size = self.x_test_size

        self.x_test_size = num_samples
        x_test_all: list[FloatArray] = []
        for fidelity in range(self.n_fidelities):
            x_test_fidelity = self._x_test_grid(bounds=self.bounds, seed=seed)
            x_test_fidelity = np.column_stack(
                (
                    x_test_fidelity,
                    np.full(x_test_fidelity.shape[0], fidelity, dtype=float),
                )
            )
            x_test_all.append(x_test_fidelity)

        self.x_test_size = old_x_test_size
        return np.concatenate(x_test_all, axis=0)

    def _x_test_grid(
        self,
        bounds: Bounds | None = None,
        seed: int = 0,
    ) -> FloatArray:
        """Create a sorted grid of points for evaluation."""
        if bounds is None:
            bounds = self.bounds

        return x_test_grid_gen(self.ndim, bounds, self._x_test_size, seed=seed)

    def _output_maker(
        self,
        output: FloatArray,
        noise_level: float,
    ) -> FloatArray:
        """Combine the function output with its noise level."""
        return np.column_stack((output, np.ones_like(output) * noise_level))

    @property
    def name(self) -> str:
        """Return the name of the test function."""
        if self.correlation_adjustment is not None:
            return f"{self._name}{self._correlation_adjustment:.2f}"
        return self._name

    @property
    def ndim(self) -> int:
        """Return the number of input dimensions."""
        return self._ndim

    @property
    def correlation_adjustment(self) -> float | None:
        """Return the correlation adjustment."""
        return self._correlation_adjustment

    @property
    def n_fidelities(self) -> int:
        """Return the number of fidelities."""
        return self._n_fidelities

    @property
    def global_maximum(self) -> float:
        """Return the cached global maximum."""
        if self._global_maximum is None:
            self._global_maximum = self._compute_global_max()
        return self._global_maximum

    @global_maximum.setter
    def global_maximum(self, value: float | None) -> None:
        """Override or recompute the cached maximum."""
        self._global_maximum = (
            self._compute_global_max() if value is None else float(value)
        )

    def _compute_global_max(self, print_val: bool = False) -> float:
        """Run a short optimisation to estimate the global maximum."""
        return self.find_optimum(
            maximise=True,
            number_of_searches=100,
            print_location_and_value=print_val,
        )

    def _compute_global_min(self, print_val: bool = False) -> float:
        """Run a short optimisation to estimate the global minimum."""
        return self.find_optimum(
            maximise=False,
            number_of_searches=100,
            print_location_and_value=print_val,
        )

    @property
    def global_minimum(self) -> float:
        """Return the cached global minimum."""
        if self._global_minimum is None:
            self._global_minimum = self._compute_global_min()
        return self._global_minimum

    @global_minimum.setter
    def global_minimum(self, value: float | None) -> None:
        """Override or recompute the cached minimum."""
        self._global_minimum = (
            self._compute_global_min() if value is None else float(value)
        )

    @property
    def fidelity_dictionary(self) -> dict[int, FidelityFunction]:
        """Return the fidelity evaluators."""
        return self._fidelity_dictionary

    @property
    def x_test(self) -> FloatArray:
        """Return the cached test grid."""
        return self._x_test_grid()

    @property
    def x_test_size(self) -> int:
        """Return the size of the test grid."""
        return self._x_test_size

    @x_test_size.setter
    def x_test_size(self, new_size: int) -> None:
        """Set the size of the test grid."""
        if new_size <= 0:
            raise ValueError("x_test_size must be a positive integer.")
        self._x_test_size = new_size

    @property
    def bounds(self) -> Bounds:
        """Return the domain bounds for each dimension."""
        return self._bounds

    @property
    def standard_samples(self) -> list[int]:
        """Return the default sample count for each fidelity."""
        return self._standard_samples

    @property
    def true_fidelity_dictionary(self) -> dict[int, FloatArray]:
        """Return the true outputs for every fidelity."""
        return {
            fidelity: self.get_true_y(fidelity=fidelity)[:, 0]
            for fidelity in range(self.n_fidelities)
        }

    def __call__(self, x: FloatArray) -> FloatArray:
        """Evaluate data where the final column stores fidelity."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[-1] != self.ndim + 1:
            raise ValueError(
                "Input dimension mismatch. Expected "
                f"{self.ndim + 1}, got {x.shape[-1]}. "
                "Expected shape (n_samples, n_dimensions + 1) with "
                "the final column holding an integer fidelity value."
            )

        outputs = np.zeros((x.shape[0], 2), dtype=float)
        for index in range(x.shape[0]):
            fidelity_value = float(x[index, -1])
            if not fidelity_value.is_integer():
                raise ValueError(
                    "The final column must contain integer fidelities."
                )

            row_output = self.evaluate(
                x[index, :-1],
                fidelity=int(fidelity_value),
            )
            outputs[index] = row_output[0]

        return outputs
