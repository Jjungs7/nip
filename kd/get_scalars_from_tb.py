from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import matplotlib.pyplot as plt
import numpy as np

def my_summary_iterator(path):
    idx = 0
    for r in tf_record.tf_record_iterator(path):
        idx += 1
        if idx == 1000000:
            break
        yield event_pb2.Event.FromString(r)

datas = []
i1=0
i2=0
for event in my_summary_iterator("runs/Nov25_14-32-10_poolc2-pc/events.out.tfevents.1574659930.poolc2-pc"):
    for v in event.summary.value:
        if v.tag == "standard_deviation":
            i1+=1
            datas.append(v.simple_value)
        if v.tag == "entropy":
            i2+=1

arr = np.array(datas)
print(arr)
plt.plot(datas, 'ro')
plt.ylabel('values')
plt.show()
