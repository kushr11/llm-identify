import math
# x=[25,50,200,1000]
# y=[6,5,2.5,1.5]
# min_l=0
# besta=7
# bestb=0.004
# for a in (0,9,1):
#     for b in(0,0.5,0.001):
#         y1=a*(math.e)**(-b*x[0])
#         y2=a*(math.e)**(-b*x[1])
#         y3=a*(math.e)**(-b*x[2])
#         y4=a*(math.e)**(-b*x[3])
#         loss=(y1-y[0])**2+(y2-y[1])**2+(y3-y[2])**2+(y4-y[3])**2
#         if loss<min_l:
#             min_l=loss
#             besta=a
#             bestb=b
# print(besta,bestb)
# y1=besta*(math.e)**(-bestb*x[0])
# y2=besta*(math.e)**(-bestb*x[1])
# y3=besta*(math.e)**(-bestb*x[2])
# y4=besta*(math.e)**(-bestb*x[3])
# print(y1,y2,y3,y4)
a=4.8
b=0.001
def cal(x):
    y=a*(math.e+0.1)**(-1*b*x)
    return y
print(cal(25),cal(50),cal(200),cal(1000))