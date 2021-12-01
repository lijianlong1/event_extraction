# -*- ecoding: utf-8 -*-
# @ModuleName: extraction
# @Function: 
# @Author: long
# @Time: 2021/11/15 20:58
# *****************conding***************
import torch
from transformers import BertModel, BertTokenizer

from model_method import InputEncoder, OutputDecoder, DomTrigger, AuxTrigger, Argument

def extraction(content):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = {
        "max_object_len": 40,  # 平均长度 7.22, 最大长度93
        "max_subject_len": 40,  # 平均长度10.0, 最大长度138
        "max_time_len": 20,  # 平均长度6.03, 最大长度22
        "max_location_len": 25,  # 平均长度3.79,最大长度41
        "max_trigger_len": 6
    }
    print(content)
    special_map = {'&': '[unused1]', '-': '[unused2]', '*': '[unused3]'}
    pre_train_dir = "./chinese_wwm_pytorch/"
    tokenizer = BertTokenizer(vocab_file=pre_train_dir + "vocab.txt")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 512
    encode_obj = InputEncoder(max_len=max_len, tokenizer=tokenizer, special_query_token_map=special_map)
    decode_obj = OutputDecoder()

    dominant_trigger_model = DomTrigger(pre_train_dir=pre_train_dir)
    auxiliary_trigger_model = AuxTrigger(pre_train_dir=pre_train_dir)
    argument_model = Argument(pre_train_dir=pre_train_dir)

    dominant_trigger_model.load_state_dict(torch.load("./model_extraction/dominant_trigger.pth", map_location=device), strict=False)
    auxiliary_trigger_model.load_state_dict(torch.load("./model_extraction/auxiliary_trigger.pth", map_location=device), strict=False)
    argument_model.load_state_dict(torch.load("./model_extraction/argument.pth", map_location=device), strict=False)

    for i in [dominant_trigger_model, auxiliary_trigger_model, argument_model]:
        for p in i.parameters():
            p.requires_grad = False

    dominant_trigger_model.to(device)
    auxiliary_trigger_model.to(device)
    argument_model.to(device)

    dominant_trigger_model.eval()
    auxiliary_trigger_model.eval()
    argument_model.eval()
    #content = input()

    with torch.no_grad():
        # content = "今天北京市出现了大规模的疫情，已经启动了相关的核酸检测工作"
            # 这个地方的训练数据可以直接进行更改，进行相应的事件抽取的研究，item["n_triggers"]直接赋值成为1
        # 在这个地方可以进行相应的模块化的改进，用以实现对多段文本的事件抽取功能，省去每执行一次接口，就重新加载模型的操作。
        extraction_result_all = []
        id, context, n_triggers = 1, content, 5   # 限定抽取最多的触发词的个数,在此直接限定最大触发词的个数，用于后面对所有文本的触发词进行识别和处理
        trigger_input = encode_obj.trigger_enc(context=context, is_dominant=True)
        s_seq, e_seq, p_seq = dominant_trigger_model.forward(
            input_ids=trigger_input["input_ids"], input_mask=trigger_input["input_mask"],
            input_seg=trigger_input["input_seg"], span_mask=trigger_input["span_mask"]
        )
        trigger_out = decode_obj.dominant_dec(context=context, s_seq=s_seq.cpu().numpy()[0],
                                              e_seq=e_seq.cpu().numpy()[0],
                                              p_seq=p_seq.cpu().numpy()[0],
                                              context_range=trigger_input["context_range"],
                                              n_triggers=n_triggers)
        for i in trigger_out:
            print('这个是句子中提取到的触发词',i['answer'])



        if len(trigger_out) < n_triggers:
            print("---%s号测试文本调用辅助触发词抽取模型---" % id)
            s_seq, e_seq = auxiliary_trigger_model.forward(
                input_ids=trigger_input["input_ids"], input_mask=trigger_input["input_mask"],
                input_seg=trigger_input["input_seg"]
            )
            trigger_out.extend(decode_obj.auxiliary_dec(context=context, s_seq=s_seq.cpu().numpy()[0],
                                                        e_seq=e_seq.cpu().numpy()[0],
                                                        context_range=trigger_input["context_range"]))


        for i in trigger_out:
            print('这个是句子中提取到的触发词和相应的辅助触发词',i['answer'])

        for index,jtem in enumerate(trigger_out, 0):
            obj_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                end=jtem["end"], arg="object")
            sub_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                end=jtem["end"], arg="subject")
            tim_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                end=jtem["end"], arg="time")
            loc_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                end=jtem["end"], arg="location")
            cls, s_seq, e_seq = argument_model.forward(
                input_ids=torch.cat([i["input_ids"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0),
                input_seg=torch.cat([i["input_seg"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0),
                input_mask=torch.cat([i["input_mask"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0)
            )
            cls, s_seq, e_seq = cls.cpu().numpy(), s_seq.cpu().numpy(), e_seq.cpu().numpy()
            obj_out = decode_obj.argument_dec(context=context, context_range=obj_input["context_range"],
                                              s_seq=s_seq[0], e_seq=e_seq[0],
                                              cls=cls[0], arg_type="object")
            sub_out = decode_obj.argument_dec(context=context, context_range=sub_input["context_range"],
                                              s_seq=s_seq[1], e_seq=e_seq[1],
                                              cls=cls[1], arg_type="subject")
            tim_out = decode_obj.argument_dec(context=context, context_range=tim_input["context_range"],
                                              s_seq=s_seq[2], e_seq=e_seq[2],
                                              cls=cls[2], arg_type="time")
            loc_out = decode_obj.argument_dec(context=context, context_range=loc_input["context_range"],
                                              s_seq=s_seq[3], e_seq=e_seq[3],
                                              cls=cls[3], arg_type="location")
            #writer.writerow([id, jtem["answer"], obj_out, sub_out, tim_out, loc_out])
            #print("id->%s已经完成分析" % id)
            #print([id, jtem["answer"], obj_out, sub_out, tim_out, loc_out])   # 在此不进行相关的查看和输出
            extraction_result = {'id':id, 'trigger':jtem["answer"], 'obj_out':obj_out, 'sub_out':sub_out, 'time_out':tim_out, 'loc_out':loc_out}
            if extraction_result not in extraction_result_all:
                extraction_result_all.append(extraction_result)
    print("完成")
    return extraction_result_all

# 至此所有的文本的触发词全部识别完毕，后面还需要对事件类别，或者是触发词的类别进行判断，或者直接返回相应的抽取出来的事件
# 在后续可以将相应的抽取出来的触发词进行分类，用于计算相应的触发词的类别，例如违法诈骗类，和相应的其他类的东西。
# 词语相似度计算。


def extraction2(input_sen):
    """
    直接使用相应的另一种方式进行事件抽取功能的实现
    :param input_sen: 输入的新闻文本句子："此外，深航积极恢复湖北其他城市的航班运力，已于近日陆续恢复襄阳往返深圳、惠州、沈阳、珠海，宜昌往返惠州、温州的航班"
    :return:经过处理后的包含触发词等的文本信息。return {"id":1,"trigger":"作出","obj_out":"越秀区公安分局","sub_out":"限期出境处罚","time_out":"4月3日","loc_out":""}
    """
    content = str(input_sen)
    trigger = ""
    obj_out = ""
    sub_out = ''
    time_out = ""
    loc_out = ""
    result = {"id":1,"trigger":trigger,"obj_out":obj_out,"sub_out":sub_out,"time_out":time_out,"loc_out":loc_out}
    return result


def extraction3(content_all_list: list) -> list: # 指定这个函数的输入输出类型，输入类型为一个list，输出类型也为一个list
    """

    :param content_all_list:输入到模型中的数据
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = {
        "max_object_len": 40,  # 平均长度 7.22, 最大长度93
        "max_subject_len": 40,  # 平均长度10.0, 最大长度138
        "max_time_len": 20,  # 平均长度6.03, 最大长度22
        "max_location_len": 25,  # 平均长度3.79,最大长度41
        "max_trigger_len": 6
    }
    print(content_all_list)
    special_map = {'&': '[unused1]', '-': '[unused2]', '*': '[unused3]'}
    pre_train_dir = "./chinese_wwm_pytorch/"
    tokenizer = BertTokenizer(vocab_file=pre_train_dir + "vocab.txt")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 512 # 表示句子的最大长度
    encode_obj = InputEncoder(max_len=max_len, tokenizer=tokenizer, special_query_token_map=special_map)
    decode_obj = OutputDecoder()

    dominant_trigger_model = DomTrigger(pre_train_dir=pre_train_dir)
    auxiliary_trigger_model = AuxTrigger(pre_train_dir=pre_train_dir)
    argument_model = Argument(pre_train_dir=pre_train_dir)

    dominant_trigger_model.load_state_dict(torch.load("./model_extraction/dominant_trigger.pth", map_location=device), strict=False)
    auxiliary_trigger_model.load_state_dict(torch.load("./model_extraction/auxiliary_trigger.pth", map_location=device), strict=False)
    argument_model.load_state_dict(torch.load("./model_extraction/argument.pth", map_location=device), strict=False)

    for i in [dominant_trigger_model, auxiliary_trigger_model, argument_model]:
        for p in i.parameters():
            p.requires_grad = False

    dominant_trigger_model.to(device)
    auxiliary_trigger_model.to(device)
    argument_model.to(device)

    dominant_trigger_model.eval()
    auxiliary_trigger_model.eval()
    argument_model.eval()
    #content = input()
    with torch.no_grad():
        # content = "今天北京市出现了大规模的疫情，已经启动了相关的核酸检测工作"
            # 这个地方的训练数据可以直接进行更改，进行相应的事件抽取的研究，item["n_triggers"]直接赋值成为1
        # 在这个地方可以进行相应的模块化的改进，用以实现对多段文本的事件抽取功能，省去每执行一次接口，就重新加载模型的操作。
        extraction_result_all = []
        for index,content in enumerate(content_all_list, 0):
            # 至此所有的信息都已经取得并且拿到，可以直接进行模型的使用，和数据的传递了
            id, context, n_triggers = index, content, 5
            trigger_input = encode_obj.trigger_enc(context=context, is_dominant=True)
            s_seq, e_seq, p_seq = dominant_trigger_model.forward(
                input_ids=trigger_input["input_ids"], input_mask=trigger_input["input_mask"],
                input_seg=trigger_input["input_seg"], span_mask=trigger_input["span_mask"]
            )
            trigger_out = decode_obj.dominant_dec(context=context, s_seq=s_seq.cpu().numpy()[0],
                                                  e_seq=e_seq.cpu().numpy()[0],
                                                  p_seq=p_seq.cpu().numpy()[0],
                                                  context_range=trigger_input["context_range"],
                                                  n_triggers=n_triggers)
            if len(trigger_out) < n_triggers:
                print("---%s号测试文本调用辅助触发词抽取模型---" % id)
                s_seq, e_seq = auxiliary_trigger_model.forward(
                    input_ids=trigger_input["input_ids"], input_mask=trigger_input["input_mask"],
                    input_seg=trigger_input["input_seg"]
                )
                trigger_out.extend(decode_obj.auxiliary_dec(context=context, s_seq=s_seq.cpu().numpy()[0],
                                                            e_seq=e_seq.cpu().numpy()[0],
                                                            context_range=trigger_input["context_range"]))
            for jtem in trigger_out:
                obj_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                    end=jtem["end"], arg="object")
                sub_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                    end=jtem["end"], arg="subject")
                tim_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                    end=jtem["end"], arg="time")
                loc_input = encode_obj.argument_enc(context=context, trigger=jtem["answer"], start=jtem["start"],
                                                    end=jtem["end"], arg="location")
                cls, s_seq, e_seq = argument_model.forward(
                    input_ids=torch.cat([i["input_ids"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0),
                    input_seg=torch.cat([i["input_seg"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0),
                    input_mask=torch.cat([i["input_mask"] for i in [obj_input, sub_input, tim_input, loc_input]], dim=0)
                )
                cls, s_seq, e_seq = cls.cpu().numpy(), s_seq.cpu().numpy(), e_seq.cpu().numpy()
                obj_out = decode_obj.argument_dec(context=context, context_range=obj_input["context_range"],
                                                  s_seq=s_seq[0], e_seq=e_seq[0],
                                                  cls=cls[0], arg_type="object")
                sub_out = decode_obj.argument_dec(context=context, context_range=sub_input["context_range"],
                                                  s_seq=s_seq[1], e_seq=e_seq[1],
                                                  cls=cls[1], arg_type="subject")
                tim_out = decode_obj.argument_dec(context=context, context_range=tim_input["context_range"],
                                                  s_seq=s_seq[2], e_seq=e_seq[2],
                                                  cls=cls[2], arg_type="time")
                loc_out = decode_obj.argument_dec(context=context, context_range=loc_input["context_range"],
                                                  s_seq=s_seq[3], e_seq=e_seq[3],
                                                  cls=cls[3], arg_type="location")
                #writer.writerow([id, jtem["answer"], obj_out, sub_out, tim_out, loc_out])
            print("id->%s已经完成分析" % id)
            print([id, tim_out, loc_out, obj_out, jtem["answer"], sub_out])
            extraction_result = {'id': id, 'trigger': jtem["answer"], 'obj_out': obj_out, 'sub_out': sub_out,
                                 'time_out': tim_out, 'loc_out': loc_out}
            extraction_result_all.append(extraction_result)
        print("完成")
    return extraction_result_all