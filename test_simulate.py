"""Mathematical checks independent of the short demonstration payload outcome."""
import unittest
import math
import numpy as np
from simulate import qpsk, decisions, text_bits, ofdm_tx, ofdm_rx, sparse_channels, NFFT, NCP, BINS


class BasebandTests(unittest.TestCase):
    def test_text_and_constellation(self):
        self.assertEqual(''.join(map(str, text_bits('H'))), '01001000')
        expected = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
        np.testing.assert_allclose(qpsk([0,0,0,1,1,0,1,1]), expected)

    def test_parseval_and_noiseless_recovery(self):
        bits = np.random.default_rng(10).integers(0, 2, size=300, dtype=np.uint8)
        s = qpsk(bits)
        grid, tx = ofdm_tx(s, 3)
        blocks = tx.reshape(3, NFFT+NCP)
        np.testing.assert_allclose(blocks[:, :NCP], blocks[:, -NCP:])
        self.assertAlmostEqual(float(np.sum(abs(blocks[:, NCP:])**2)), len(s), places=10)
        np.testing.assert_array_equal(decisions(ofdm_rx(tx)[:len(s)]), bits)
        np.testing.assert_allclose(grid[:, 0], 0)

    def test_fft_tone_frequency_and_scaling(self):
        s = np.zeros(len(BINS), complex)
        s[0] = 1
        _, tx = ofdm_tx(s, 1)
        expected = np.exp(-2j*np.pi*36*np.arange(NFFT)/NFFT)/np.sqrt(NFFT)
        np.testing.assert_allclose(tx[NCP:], expected, atol=2e-14)

    def test_awgn_ber_matches_qpsk_theory(self):
        rng = np.random.default_rng(321)
        bits = rng.integers(0, 2, size=400_000, dtype=np.uint8)
        s = qpsk(bits)
        # Es/N0 = 1 (0 dB); theoretical Pb=Q(sqrt(Es/N0)).
        noise = (rng.normal(size=s.size)+1j*rng.normal(size=s.size))/np.sqrt(2)
        ber = np.mean(decisions(s+noise) != bits)
        theory = 0.5*math.erfc(1/np.sqrt(2))
        self.assertLess(abs(ber-theory), 0.003)

    def test_unitary_fft_noise_variance(self):
        rng = np.random.default_rng(111)
        var = 10**(-15/10)
        noise = np.sqrt(var/2)*(rng.normal(size=(4000,NFFT))+1j*rng.normal(size=(4000,NFFT)))
        transformed = np.fft.fft(noise, axis=1, norm='ortho')
        self.assertLess(abs(np.mean(abs(transformed[:, BINS])**2)/var-1), 0.01)

    def test_channel_shape_beamspace_energy_and_rayleigh_ensemble(self):
        rng = np.random.default_rng(123)
        h, hb, gains, delays = sparse_channels(rng)
        self.assertEqual(h.shape, (2,128,2,8))
        self.assertTrue(np.all(delays >= 0))
        np.testing.assert_allclose(np.sum(abs(h)**2, axis=(-2,-1)), np.sum(abs(hb)**2, axis=(-2,-1)))
        # Check ensemble scaling, not per-realization renormalization.
        energies = [np.sum(abs(sparse_channels(rng)[0][:,0])**2, axis=(-2,-1)) for _ in range(500)]
        self.assertLess(abs(np.mean(energies)-16), 1.5)


if __name__ == '__main__':
    unittest.main()
