# -*- ecoding: utf-8 -*-
# @ModuleName: test_extraction
# @Function: 
# @Author: long
# @Time: 2021/11/15 21:08
# *****************conding***************
import requests
import time
#import json
request_url = 'http://127.0.0.1:8000/event_extraction/'
#response = requests.post(url=request_url,json={'data':'同时，因其违反中国法律、法规规定，妨害社会管理，根据《出境入境管理法》第八十一条第一款，越秀区公安分局于4月3日对其作出限期出境处罚'})
# 输出结果response
start = time.time()
response = requests.post(url=request_url,json={'data':'9月5日晚，仙降街道两名干部在瑞安城区某KTV包厢互殴，造成恶劣影响，严重损害了党员干部形象。'})
# {"id":1,"trigger":"作出","obj_out":"越秀区公安分局","sub_out":"限期出境处罚","time_out":"4月3日","loc_out":""}
end = time.time()
print(response.text, '句子分析的运行时间为：', end-start, '秒')

# 阅读理解方式进行相应的关键词抽取：
# 时间地点，动作，主体和客体

# 使用相关的接口进行自动化测试

# 目前需要的环境有，直接将相应的模型，放到公司的gpu上面






