"""Headless checks: true Fourier synthesis and every animation drawing branch."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
from animate import Animation, prepare, partial_ifft, STAGES
from simulate import NFFT, BINS, TONES


class CanvasRecorder:
    """Validate numerical drawing commands without pretending to test native Tk."""
    def __init__(self):
        self.items = []
    def winfo_width(self): return 520
    def winfo_height(self): return 400
    def delete(self, _): self.items.clear()
    def record(self, *coords, **kwargs):
        assert all(np.isfinite(float(v)) for v in coords)
        self.items.append((coords, kwargs))
    create_line = record
    create_oval = record
    create_rectangle = record
    create_text = record


class AnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.report, cls.arrays = prepare(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_partial_synthesis_at_start_one_tone_and_completion(self):
        for u in (1,2):
            for q in range(3):
                grid = self.arrays[f'ue{u}_grid'][q]
                np.testing.assert_array_equal(partial_ifft(grid,0),np.zeros(NFFT))
                expected = grid[BINS[0]]*np.exp(2j*np.pi*TONES[0]*np.arange(NFFT)/NFFT)/np.sqrt(NFFT)
                np.testing.assert_allclose(partial_ifft(grid,1),expected,atol=1e-14)
                np.testing.assert_allclose(partial_ifft(grid,72),self.arrays[f'ue{u}_ifft'][q],atol=3e-14)

    def test_all_blocks_render_for_both_users_and_all_symbols(self):
        app = Animation.__new__(Animation)
        app.report,app.arrays = self.report,self.arrays
        canvas=CanvasRecorder()
        for symbol in ('0','1','2'):
            app.symbol=type('Selection',(),{'get':lambda self,v=symbol:v})()
            for stage in range(len(STAGES)):
                for progress in (0.,.5,1.):
                    app.stage,app.progress=stage,progress
                    for view in ('Rx1','Rx2','Combined / equalized'):
                        app.receiver=type('Selection',(),{'get':lambda self,v=view:v})()
                        for u in (1,2):
                            app.render_user(canvas,u)
                            self.assertTrue(canvas.items)


if __name__ == '__main__':
    unittest.main()
