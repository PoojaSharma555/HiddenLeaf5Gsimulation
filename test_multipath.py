import tempfile
import unittest
from pathlib import Path
import numpy as np
from mimo import run_mimo
from multipath import channel_taps
from simulate import BINS


class MultipathTests(unittest.TestCase):
    def test_physical_convolution_equals_fd_matrix_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,a=run_mimo({'channel_model':'rayleigh','noise_enabled':False},output=Path(tmp))
            for u in (1,2):
                self.assertEqual(a[f'H{u}'].shape,(128,2,8))
                self.assertFalse(np.allclose(a[f'H{u}'][0],a[f'H{u}'][30]))
                np.testing.assert_allclose(a[f'ue{u}_clean_fd'],a[f'ue{u}_predicted_fd'],atol=2e-13)
                self.assertEqual(report['users'][u-1]['bit_errors'],0)
                np.testing.assert_array_equal(a[f'ue{u}_antenna_noise'],0)
            self.assertEqual(a['path_taps'].shape,(2,3,2,8))
            for u in (1,2):
                np.testing.assert_allclose(a[f'ue{u}_combined_fd']/a[f'ue{u}_desired_gain_fd'][None,:],a[f'ue{u}_rx_grid'],atol=1e-13)
                from receiver_view import receiver_values
                vals=receiver_values(a,u,0,92)
                np.testing.assert_allclose(vals[5],a[f'ue{u}_rx_grid'][0,92],atol=1e-13)
            np.testing.assert_allclose(np.linalg.norm(a['precoder_fd'],axis=1),1)

    def test_unequal_delays_and_noise_match_fd_equation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,a=run_mimo({'channel_model':'rayleigh','delays_samples':[0,4,10],'snr_db':-10},output=Path(tmp))
            for u in (1,2):
                np.testing.assert_allclose(a[f'ue{u}_antenna_fft'],a[f'ue{u}_predicted_fd']+a[f'ue{u}_noise_fd'],atol=5e-13)
        with self.assertRaises(ValueError):channel_taps({'delays_samples':[0,1,11]},555)
        with self.assertRaises(ValueError):channel_taps({'delays_samples':[0,1.5,3]},555)

    def test_rayleigh_path_gain_ensemble_power(self):
        draws=np.array([channel_taps({},seed)[2] for seed in range(1200)])
        expected=channel_taps({},0)[4]
        np.testing.assert_allclose(np.mean(abs(draws)**2,axis=(0,1)),expected,rtol=.08)
        self.assertLess(abs(draws.mean()),.04)

    def test_payload_allocation_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,a=run_mimo({'channel_model':'rayleigh'},output=Path(tmp))
            for u,counts in [(1,[72,72,48]),(2,[72,72,8])]:
                self.assertEqual(np.count_nonzero(a[f'ue{u}_grid'][:,BINS],axis=1).tolist(),counts)


if __name__=='__main__':unittest.main()
