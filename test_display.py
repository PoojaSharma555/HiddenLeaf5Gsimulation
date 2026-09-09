import unittest
import tempfile
from pathlib import Path
from animate import Animation,prepare
from inline_beams import draw_inline_beams
from fd_view import matrix
from test_animate import CanvasRecorder
import numpy as np

class Selection:
    def __init__(self,value):self.value=value
    def get(self):return self.value

class DisplayTests(unittest.TestCase):
    def test_brackets_preserve_rectangular_matrix_orientation(self):
        c=CanvasRecorder()
        matrix(c,np.zeros((8,2)),0,0)
        texts=[item for item in c.items if 'text' in item[1]]
        self.assertEqual(len(texts),16)
        self.assertEqual(len(set(item[0][0] for item in texts)),2)
        self.assertEqual(len(set(item[0][1] for item in texts)),8)

    def test_stems_and_beams_for_actual_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,arrays=prepare(Path(tmp))
        app=Animation.__new__(Animation)
        app.report,app.arrays,app.config=report,arrays,report['channel_config']
        app.symbol=Selection('2');app.tone=Selection('12');app.receiver=Selection('Rx1')
        app.progress=1.;app.stage=3
        c=CanvasRecorder()
        for source in ['User layer','BS antenna 1','BS antenna 8']:
            app.input_view=Selection(source)
            app.render_user(c,1)
            stems=[coords for coords,kw in c.items if len(coords)==4 and kw.get('width')==1 and coords[0]==coords[2]]
            self.assertEqual(len(stems),256)
        app.inline_beams=CanvasRecorder()
        draw_inline_beams(app)
        self.assertTrue(any('peak' in item[1].get('text','') for item in app.inline_beams.items))

if __name__=='__main__':unittest.main()
