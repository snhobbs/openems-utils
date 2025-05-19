import unittest
import numpy as np
from numpy.testing import assert_allclose
from openems_utils.sources import (
    gaussian_step, evaluate_custom_source_string
)

class TestGaussianStep(unittest.TestCase):

    def test_rising_step_shape(self):
        t = np.linspace(0, 2e-9, 5000)
        expr, _ = gaussian_step(rise_time=50e-12, center_time=1e-9, dB_cutoff=6, sign=1)
        _, values = evaluate_custom_source_string(expr, t)
        values = np.array(values)

        self.assertLess(values[0], 0.05, "Start value too high for rising step")
        self.assertGreater(values[-1], 0.95, "End value too low for rising step")
        self.assertTrue(np.all(np.diff(values) >= -1e-3), "Rising edge not monotonically increasing")

    def test_falling_step_shape(self):
        t = np.linspace(0, 2e-9, 5000)
        expr, _ = gaussian_step(rise_time=50e-12, center_time=1e-9, dB_cutoff=6, sign=-1)
        _, values = evaluate_custom_source_string(expr, t)
        values = np.array(values)

        self.assertGreater(values[0], 0.95, "Start value too low for falling step")
        self.assertLess(values[-1], 0.05, "End value too high for falling step")
        self.assertTrue(np.all(np.diff(values) <= 1e-3), "Falling edge not monotonically decreasing")

    def test_nyquist_frequency_scaling(self):
        _, f1 = gaussian_step(rise_time=100e-12, center_time=1e-9, dB_cutoff=3)
        _, f2 = gaussian_step(rise_time=100e-12, center_time=1e-9, dB_cutoff=6)

        self.assertGreater(f2, f1, "Nyquist frequency should increase with dB cutoff")

    def test_basic_expression_evaluation(self):
        t = np.array([0.0, 1.0, 2.0])
        expr = "t * t"
        times, values = evaluate_custom_source_string(expr, t)

        self.assertCountEqual(times, t)
        # Use assert_allclose to compare numpy array or list with floats
        assert_allclose(values, t*t, rtol=1e-12)

    def test_expression_with_context(self):
        t = np.array([1.0, 2.0])
        expr = "scale * t"
        scale = 2
        times, values = evaluate_custom_source_string(expr, t, context={'scale': scale})

        assert_allclose(values, scale*t, rtol=1e-12)

    def test_invalid_sign_raises(self):
        with self.assertRaises(AssertionError):
            gaussian_step(1e-10, 1e-9, sign=0)


if __name__ == '__main__':
    unittest.main()

