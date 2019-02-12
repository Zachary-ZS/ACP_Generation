import os
FILENAME = r'result15_20000northym.txt'
def process():
    wf = open('final'+FILENAME,'w')
    wf2 = open('generated_poem'+FILENAME,'w')
    with open(FILENAME,'r') as ff:
        lines=ff.readlines()
    for line in lines:
        cnt=0
        num=0
        for word in line.split():
            wf.write(word)
            if num != 0:
                wf2.write(word)
            cnt=cnt+1
            if cnt==7:
                if num==0:
                    wf.write('，')
                elif num==1:
                    wf.write('。')
                    wf2.write('。')
                elif num==2:
                    wf.write('，')
                    wf2.write('，')
                else:
                    wf.write('。')
                    wf2.write('。')
                cnt=0
                num=num+1
        wf.write('\n')
        wf2.write('\n')
if __name__ == '__main__':
    process()
