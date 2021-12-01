# -*- ecoding: utf-8 -*-
# @ModuleName: test_extraction3
# @Function: 
# @Author: long
# @Time: 2021/11/23 14:46
# *****************conding***************

import requests
import time
#import json
request_url = 'http://127.0.0.1:8000/event_extraction3/'
#response = requests.post(url=request_url,json={'data':'同时，因其违反中国法律、法规规定，妨害社会管理，根据《出境入境管理法》第八十一条第一款，越秀区公安分局于4月3日对其作出限期出境处罚'})
# 输出结果response
start = time.time()
# 指定模型的相应的输入为字符串格式的list数据
sen_data = ['9月5日晚，仙降街道两名干部在瑞安城区某KTV包厢互殴，造成恶劣影响，严重损害了党员干部形象.',
            '同时，因其违反中国法律、法规规定，妨害社会管理，根据《出境入境管理法》第八十一条第一款，越秀区公安分局于4月3日对其作出限期出境处罚',
            '21日，俄总统新闻秘书用“歇斯底里”抨击美国，称美军在俄罗斯边境演习，反而“煽风点火”什么乌克兰要被俄罗斯入侵。',
            '22日，俄情报部门称，有关消息“是美国国务院一手编造出来的”，目的是“吓唬世界”。',
            '佩斯科夫同时表示，俄罗斯方面正在密切关注西方如何帮助基辅强化其部队，而乌克兰在顿巴斯地区的挑衅也增多。']
sen_data = str(sen_data)
#print(len(sen_list),type(sen_list))
response = requests.post(url=request_url,json={'data': sen_data})
end = time.time()
print(response.text, '句子分析的运行时间为：', end-start, '秒')
