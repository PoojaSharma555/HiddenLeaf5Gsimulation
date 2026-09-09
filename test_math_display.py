import unittest
from math_display import MathCanvas,FORMULAS
from animate import STAGES


class ApproximateFont:
    def __init__(self,size):self.size=size
    def measure(self,text):return len(text)*self.size*.6
    def metrics(self,name):return self.size*1.2


class MathLayoutTests(unittest.TestCase):
    def test_all_formulas_have_bounded_layout_and_fraction_lines(self):
        self.assertEqual(len(FORMULAS),len(STAGES))
        canvas=MathCanvas.__new__(MathCanvas)
        canvas.face=lambda size:ApproximateFont(size)
        for node in FORMULAS:
            w,h,commands=canvas.layout(node,20)
            self.assertGreater(w,0);self.assertLess(h,84)
            self.assertTrue(commands)
            for cmd in commands:
                self.assertGreaterEqual(cmd[1],0);self.assertGreaterEqual(cmd[2],0)
        self.assertTrue(any(c[0]=='line' for c in canvas.layout(FORMULAS[2],20)[2]))

if __name__=='__main__':unittest.main()
