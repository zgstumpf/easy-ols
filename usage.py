#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from easy_ols import EasyOLS

data = pd.read_csv("data.csv", sep=";")

myOLS = EasyOLS("pH", "citric acid", data)
myOLS.summary()

myOLS.plot()


