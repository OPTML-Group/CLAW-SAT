import numpy as np
import matplotlib.pyplot as plt 
#x = [0, 0.3 ,1 , 2, 5,10 ]
x=[0.3,1,2,5,10] 
x_baseline = np.arange(0,10,0.2).tolist()
num = len(x_baseline)
CodeClaw_FF_AT_acc = [36.45,42.12,40.56,40.32,40.30]
CodeClaw_FF_AT_pgd1 = [31.66,40.70,39.55,38.64,37.72]
CodeClaw_FF_AT_pgd5 = [30.03,38.10,36.19,35.57,34.85]
Fully_supervised_acc = [33.33] *  num
Fully_supervised_pgd1 = [26.16] * num
Fully_supervised_pgd5 = [22.61] * num
Fully_supervised_AT_acc = [35.98] *  num
Fully_supervised_AT_pgd1 = [31.59] * num
Fully_supervised_AT_pgd5 = [30.61] * num
CL_FF_AT_acc = [35.88] * num
CL_FF_AT_pgd1 = [31.29] * num
CL_FF_AT_pgd5 = [20.60] * num
CL_FF_ST_acc = [36.28] * num
CL_FF_ST_pgd1 = [28.97] * num
CL_FF_ST_pgd5 = [25.14] * num 
fig = plt.figure()
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(30,12))

CodeClaw_FF_AT_acc_java = [38.86,41.77,39.01,38.61,37.65]
CodeClaw_FF_AT_pgd1_java = [36.10,38.80,36.15,35.47,33.30]
l1=ax1.plot(x,CodeClaw_FF_AT_acc,color='r',marker='o',label='Generalization',linewidth=5,markersize=10)
l2=ax1.plot(x,CodeClaw_FF_AT_pgd1,color='b',marker='x',label="Robustness",linewidth=5,markersize=10) 
l3=ax2.plot(x,CodeClaw_FF_AT_acc_java,color='r',marker='o',label='Generalization',linewidth=5,markersize=10)
l4=ax2.plot(x,CodeClaw_FF_AT_pgd1_java,color='b',marker='x',label="Robustness",linewidth=5,markersize=10)    
# plt.plot(x,CodeClaw_FF_AT_acc,color='r',marker='o',label='Generalization')
# plt.plot(x,CodeClaw_FF_AT_pgd1,color='b',marker='x',label="Robustness")
#plt.scatter(x_baseline,Fully_supervised_acc,color='y',marker='.',label='Fully-supervised')
#plt.scatter(x_baseline,Fully_supervised_AT_acc,color='b',marker='|',label='Fully-supervised AT')
#plt.scatter(x_baseline,CL_FF_AT_acc,color='g',marker='_',label='Contracode AT')
#plt.scatter(x_baseline,CL_FF_AT_acc,color='m',marker='x',label='Contracode')
#fig.legend([l1,l2,l3,l4],"Generalization,Robustness,Generalization,Robustness",loc="center right")
#plt.xlabel('Adv. code generation Freq.')
plt.legend(loc="upper right")
ax2.set_xlabel("Epoch")
ax1.set_xlabel("Epoch")
#plt.ylabel('F1 without attack \n (Accuracy evaluation)')
ax2.set_ylabel("F1")
ax1.set_ylabel("F1")
plt.sca(ax2)
plt.xticks([0.3,1,2,5,10],['AT','1','2','5','10'])
plt.sca(ax1)
plt.xticks([0.3,1,2,5,10],['AT','1','2','5','10'])
plt.legend(loc="upper right")
plt.savefig("vsepoch")


# plt.plot(x,CodeClaw_FF_AT_acc,color='r',marker='o',label='Generalization')
# plt.plot(x,CodeClaw_FF_AT_pgd1,color='b',marker='x',label="Robustness")
# #plt.scatter(x_baseline,Fully_supervised_acc,color='y',marker='.',label='Fully-supervised')
# #plt.scatter(x_baseline,Fully_supervised_AT_acc,color='b',marker='|',label='Fully-supervised AT')
# #plt.scatter(x_baseline,CL_FF_AT_acc,color='g',marker='_',label='Contracode AT')
# #plt.scatter(x_baseline,CL_FF_AT_acc,color='m',marker='x',label='Contracode')
# plt.legend()
# #plt.xlabel('Adv. code generation Freq.')
# plt.xlabel("Epoch")
# #plt.ylabel('F1 without attack \n (Accuracy evaluation)')
# plt.ylabel("F1")
# plt.xticks([0.3,1,2,5,10],['AT','1','2','5','10'])
# plt.savefig("java")