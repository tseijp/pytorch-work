### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions
from .test_options import TestOptions

class DemoTestOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)

class DemoPoseOption(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
