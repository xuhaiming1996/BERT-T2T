import random

class PreHelper:
    def __init__(self,filePath_r,filePath_w_train,filePath_w_eval):
        self.filePath_r=filePath_r
        self.filePath_w_train=filePath_w_train
        self.filePath_w_eval = filePath_w_eval

    def proprocess(self):
        sens = []
        num = 0
        with open(self.filePath_r,mode="r",encoding="utf-8") as fr:
            for line in fr:
                num += 1
                line = line.strip()
                if line != "" and len(line)>8:
                    sens.append(line.strip())
                if num%1000000 == 0:
                    print("数据正在提取中，请耐心等待！！！")

        random.shuffle(sens)
        print("读写完成，开始写入文件")
        sens_eval = sens[:20000]
        sens_train = sens[20000:]
        with open(self.filePath_w_eval,mode="w",encoding="utf-8") as fw:
            for line in sens_eval:
                line = line.strip()
                fw.write("---xhm---".join([line,line])+"\n")


        with open(self.filePath_w_train,mode="w",encoding="utf-8") as fw:
            for line in sens_train:
                line = line.strip()
                fw.write("---xhm---".join([line,line])+"\n")



if __name__=="__main__":
    preHelper=PreHelper(filePath_r="sens_LM_raw.txt",filePath_w_train="train.txt",filePath_w_eval="eval.txt")
    preHelper.proprocess()
