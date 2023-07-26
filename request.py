#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:40:03 2023

@author: aditidadariya
"""

import requests
url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={"AGE":53,
                            "JOB":0,
                            "MARITAL":1,
                            "EDUCATION":0,
                            "DEFAULT":12,
                            "HOUSING":7,
                            "LOAN":1.25,
                            "CONTACT":1,
                            "MONTH":1,
                            "DAY_OF_WEEK":1,
                            "DURATION":0,
                            "CAMPAIGN":0,
                            "PDAYS":68,
                            "PREVIOUS":0,
                            "POUTCOME":0,
                            "EMP.VAR.RATE":0,
                            "CONS.PRICE.IDX":0,
                            "CONS.CONF.IDX":0,
                            "EURIBOR3M": 0,
                            "NR.EMPLOYED":0
                            })

print(r.json())

