
from traintester import *
from graphFunction import *
import time
from memory_profiler import profile


def addLine(name, line):
    with open(name, 'a') as f:
       f.write(line)
       f.write("\n")
    f.close()

@profile
def runAlgorithm():
    hist, yL, yP = getHistory(val)
    return hist, yL, yP
    
    

if __name__ == '__main__':
    val = 'T0'
    hist, yL, yP = runAlgorithm()   
    plot_confusion(yL,yP, val)
    plot_accuracies(hist, val)
    plot_losses(hist, val)
    plot_lrs(hist,val)
    timeTaken = time.time()-start_time
    dataLine = val+str(timeTaken)
    addLine('dataList/list.csv',dataLine)


    