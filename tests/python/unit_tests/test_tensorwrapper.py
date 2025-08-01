

import os
import parallelzone as pz
import sys
import unittest

if __name__ == '__main__':
    rv = pz.runtime.RuntimeView()








    

    my_dir = os.path.dirname(os.path.realpath(__file__))

    loader = unittest.TestLoader()
    tests = loader.discover(my_dir)
    testrunner = unittest.runner.TextTestRunner()
    ret = not testrunner.run(tests).wasSuccessful()
    sys.exit(ret)
