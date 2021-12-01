# -*- ecoding: utf-8 -*-
# @ModuleName: test
# @Function: 尝试使用轻量级fastapi接口实现相应的算法模型的使用和封装。
# @Author: long
# @Time: 2021/11/8 16:56
# *****************conding***************
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import logging
import warnings
import json
from extraction import extraction,extraction2,extraction3

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

class Sen(BaseModel):
    data: str


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):

    print(item.name)
    return item
@app.post("/event_extraction/")
async def extraction_here(sen: Sen):
    # try:
    sentence = sen.data
    print(type(sentence))
    #data_sen = json.loads(data)['sen']   # post的数据格式应该为data={"sen":"XXXXXX"}
    data_result = extraction(sentence)
    print(data_result)
    return data_result
    # except Exception as e:
    #     print("数据处理出错")
    #     return e
# 编写下一个接口，实现相关的模块化预测功能
@app.post("/event_extraction2/")
async def extraction_here2(sen: Sen):
    sentence = sen.data  # 至此相关的数据都已经拿到，拿到的数据格式为：{'data':"XXXXX"}
    # 编写一个函数，实现第二种事件抽取方案，使用传统的事件抽取思路进行模型的搭建和相关的测试。
    #  def extraction2(): 实现事件抽取功能
    result = extraction2(sentence)
    return result

# 后续需要考虑将相关的语句执行，使用模型只在内存中加载处理一次，加载完成后可以直接使用相关的算法进行语句的处理和完善
@app.post("/event_extraction3/")
async def extraction_here3(sen: Sen):
    print('shujuchenggongfasong')
    print(sen.data)
    sentence_list = eval(sen.data)
    result = extraction3(sentence_list)
    print(result)
    return result





