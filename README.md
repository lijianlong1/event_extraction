# 模型概述
整个模型使用阅读理解和信息抽取方式进行研究，其事件内容的生成方式类似于：槽填充的方式，即为使用MRC方式进行事件的抽取  
整个模型的运行流程为：不需要考虑触发词模板的问题   
1.直接先使用一个roberta-wwm模型先进行触发词的识别  
2.在抽取得到相应的触发词之后，使用相应的判断机制进行辅助触发词的识别判断，但会造成冗余信息的生成。  
3.将触发词和相应的句子拼接起来，使用另一个roberta-wwm模型进行时间、地点、主体、客体的识别  
4.识别出来后直接将结果返回即可  
注意：  
1.整个模型中，我们做了以下限定，触发词的最大个数为：5  
2.因为在相应生成模型是GPU版本的模型，所以机器需要有GPU  
3.在计算时间上，我们还需要进行考虑（在软件启动时，就进行模型的加载，之后，再将数据送入到模型中计算，
这样可以极大程度的减少模型加载预热时间）  


# 需要使用的依赖环境有
pip insatll fastapi  
pip install uvicorn  
pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html  
pip install transformers==4.9.2  
pip install numpy  
pip insatll typing  
pip install pydantic  
pip install 
# 模型使用
cd 到本文件目录下  
使用命令 uvicorn api:app --reload  
模型就能运行了  
测试模型，直接使用request插件进行数据的发送，直接运行test_post
文件夹下面的test_extraction.py文件实现对一句话的多触发词抽取，可以抽取最多5个事件  
运行test_extraction3.py可以实现列表类文本的传送输入。  
  
