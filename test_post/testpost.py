# -*- ecoding: utf-8 -*-
# @ModuleName: testpost
# @Function: 
# @Author: long
# @Time: 2021/11/15 20:44
# *****************conding***************


import requests
import json
request_url = 'http://127.0.0.1:8000/items/'
response = requests.post(url=request_url,json={'name':'kaishui','price':12.2})
print(json.loads(response.text)['name'])
# 一个标准的post接口生成



