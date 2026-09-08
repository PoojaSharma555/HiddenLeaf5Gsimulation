import tempfile
import unittest
from pathlib import Path
import numpy as np
from mimo import defaults, parse_matrix, weights, beam_patterns, run_mimo


class MIMOTests(unittest.TestCase):
    def setUp(self):
        self.h=np.array([parse_matrix(defaults()[k]) for k in ('H1','H2')])

    def test_user_matrix_and_validation(self):
        h=parse_matrix('[[1,2,i,1,-1,-i,-i,i],[1,2,1,2,i,2i,i,2i]]')
        np.testing.assert_array_equal(h,self.h[0])
        with self.assertRaises(ValueError):parse_matrix([[1,2]])
        with self.assertRaises(ValueError):weights(np.zeros((2,2,8)), 'ZF')
        with self.assertRaises(ValueError):weights(np.array([self.h[0],self.h[0]]),'ZF')

    def test_zf_power_interference_and_channel_dependent_beams(self):
        w,c,d=weights(self.h,'ZF')
        np.testing.assert_allclose(np.linalg.norm(w,axis=0),1)
        np.testing.assert_allclose(d-np.diag(np.diag(d)),0,atol=1e-12)
        theta,p,r=beam_patterns(self.h,w)
        np.testing.assert_allclose(p.max(axis=1),0)
        np.testing.assert_allclose(r.max(axis=1),0)
        h=self.h.copy();h[0,0,3]+=3j
        new_w,_,_=weights(h,'ZF')
        self.assertFalse(np.allclose(p,beam_patterns(h,new_w)[1]))
        # Known +30 degree steering vector peaks at +30 with our conjugation.
        steering=np.exp(1j*np.pi*np.arange(8)*.5)/np.sqrt(8)
        _,p,_=beam_patterns(self.h,np.column_stack([steering,steering]))
        self.assertAlmostEqual(theta[np.argmax(p[0])],30.)

    def test_applied_two_antenna_channel_and_snr_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,a=run_mimo({'snr_db':15},output=Path(tmp)/'a')
            _,b=run_mimo({'snr_db':25},output=Path(tmp)/'b')
            np.testing.assert_allclose(a['precoder'],b['precoder'])
            for u in (1,2):
                key=f'ue{u}_'
                np.testing.assert_allclose(a[key+'antenna_rx'],a[f'H{u}']@a['bs_antenna_tx']+a[key+'antenna_noise'])
                np.testing.assert_allclose(a[key+'antenna_noise']/np.sqrt(10),b[key+'antenna_noise'])
                np.testing.assert_allclose(a[key+'rx'],a['combiners'][u-1].conj()@a[key+'antenna_rx']/a['effective_coupling'][u-1,u-1])
                self.assertFalse(np.allclose(a[key+'antenna_rx'][0],a[key+'antenna_rx'][1]))
            self.assertNotIn('H_beamspace',a)
            self.assertIn('APPLIED MIMO ARRAYS',(Path(tmp)/'a'/'output.txt').read_text())

    def test_high_snr_recovers_both_users_and_mrt_leaves_interference(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,_=run_mimo({'snr_db':80},output=Path(tmp))
            self.assertEqual([u['bit_errors'] for u in report['users']],[0,0])
        _,_,d=weights(self.h,'MRT')
        self.assertGreater(abs(d[0,1]),1e-4)


if __name__=='__main__':unittest.main()
