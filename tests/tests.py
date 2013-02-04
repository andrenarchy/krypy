import unittest

import krypy
import numpy

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.tol = 1e-13
        self.v1 = numpy.ones(self.N)
        self.v2 = numpy.ones((self.N,1))
    
    def test_shape_vec(self):
        v = krypy.utils.shape_vec(self.v1)
        self.assertEqual(v.shape, (self.N,1))
        self.assertEqual(numpy.linalg.norm(v-self.v2), 0.0)

if __name__ == '__main__':
    unittest.main()
