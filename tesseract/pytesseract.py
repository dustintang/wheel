#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:08:29 2017

@author: dustin
"""

import pytesseract
from PIL import Image

image = Image.open('/Users/dustin/Downloads/1.JPG')
code = pytesseract.image_to_string(image,lang= 'chi_sim')
print(code)

