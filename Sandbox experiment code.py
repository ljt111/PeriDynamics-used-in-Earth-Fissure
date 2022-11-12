import numpy as np
from numpy import pi
from numpy import sqrt
from time import *
from numba import njit

t1 = time()
# --------------------------------------------------------------------------------------------------------------
shi_jian_bu = 3000                         # 时间步数
shu_chu_shu = 100                          # 输出数,初始值+1
jian_ge = (shi_jian_bu/(shu_chu_shu))      # 间隔步数（整数）

bei_lv=2
dian_shu_x = int(96*bei_lv+6)                            # x方向点数
dian_shu_y = int(36*bei_lv+6)
dian_shu_z = int(28*bei_lv+3)
bian_chang = 0.05/bei_lv                           # 单个质点边长
chang_x = dian_shu_x * bian_chang          # 模型长度
chang_y = dian_shu_y * bian_chang
chang_z = dian_shu_z * bian_chang
zong_dian_shu = dian_shu_x * dian_shu_y * dian_shu_z  # 总点数
ti_ji = bian_chang ** 3      # 体积
lin_yu = 3.015 * bian_chang  # 邻域大小
lin_jie_s0 = 0.0018          # 临界拉伸率

# 时间步长
mi_du = 2630                # 密度
K = 25e6                    # 体积模量
v = 0.25                     # 泊松比
# G = K*(3*(1-2*v))/(2*(1+v))  # 剪切模量
c = (9 * K) / (pi * bian_chang * (lin_yu ** 3))
# dt=0.8 * sqrt((2 * mi_du * bian_chang) / (pi * lin_yu ** 2 * bian_chang * c))  # 0.8是安全系数，参照mesh,2005文章
dt = 0.99 * sqrt((2 * mi_du * bian_chang) / (0.75 * pi * (lin_yu ** 3) * c))

print('模型长宽高为：',chang_x,chang_y,chang_z,'时间步长为：', dt,'时间步为：', shi_jian_bu)
# ----------------------------------------------------------------------------------------------------------------
@njit()
def main():
    # 中心点属性（中心点号，属性表）：
    # 0=邻域点数量，1=原始x坐标，2=原始y坐标，3=原始z坐标，4=总力x，5=总力y，6=总力z，
    # 7=加速度x，8=加速度y，9=加速度z，10=速度x，11=速度y，12=速度z，13=位移x，14=位移y，15=位移z，
    # 16=新坐标x，17=新坐标y，18=新坐标z，19=损伤率，20=密度，21=体积模量K，22=剪切模量G,
    # 23=孔隙水压差，24=m，25=theta（x的），26=组号
    shu_xing = np.zeros((zong_dian_shu, 27))

    # 键、邻域点属性(中心点号，领域点（最多122个），属性表):
    # 0=邻域点编号，1=点对间的初始距离，2=体积修正系数,
    # 3=e,4=ed,5=位移后之间的距离,6=邻域点的theta,7=邻域点的ed,8=剪切部分的临界伸长率,9=拉升部分的伸长率,10=剪切部分的伸长率,11=拉升部分的临界伸长率
    family = np.zeros((zong_dian_shu,7*7*5,12))
    damage = np.zeros((zong_dian_shu,7*7*5))              # 判断是否损伤
    shuchu = np.zeros((zong_dian_shu,8,shu_chu_shu))    # 所需要的信息：新xyz坐标、损伤率、xyz位移、组号

    # 坐标
    jishu=0
    for i in range(0,dian_shu_x):
        for i2 in range(0,dian_shu_y):
            for i3 in range(0,dian_shu_z):
                shu_xing[jishu,1]=i*bian_chang
                shu_xing[jishu,2]=i2*bian_chang
                shu_xing[jishu,3]=i3*bian_chang
                jishu+=1
    print('坐标设置完成')

    # 设置点属性
    for i in range(0,zong_dian_shu):
        if shu_xing[i,3]<0.2+3*bian_chang:
            shu_xing[i,21]=(0.77e6)/(3*(1-2*v))
            shu_xing[i,26]=1
        elif 0.2+3*bian_chang<=shu_xing[i,3]<0.8+3*bian_chang:
            shu_xing[i,21]=(1.54e6)/(3*(1-2*v))
            shu_xing[i,26]=2
        elif 0.8+3*bian_chang<=shu_xing[i,3]:
            shu_xing[i,21]=(1.54e6)/(3*(1-2*v))
            shu_xing[i,26]=3
        if shu_xing[i,3]-3*bian_chang<=0.57*(shu_xing[i,1]-3*bian_chang)-1.14 and \
                shu_xing[i,3]-3*bian_chang<=-0.57*(shu_xing[i,1]-3*bian_chang)+2.74:
            shu_xing[i,26]=4
        shu_xing[i,20]=27000/9.8
        shu_xing[i,22]=shu_xing[i,21]*(3*(1-2*v))/(2*(1+v))

    #邻域点搜索
    for i in range(0,zong_dian_shu):
        jishu=0
        for i2 in range(0,zong_dian_shu):
            if i!=i2:
                ju_li=sqrt((shu_xing[i,1]-shu_xing[i2,1])**2+
                           (shu_xing[i,2]-shu_xing[i2,2])**2+
                           (shu_xing[i,3]-shu_xing[i2,3])**2)
                if ju_li<=lin_yu:
                    family[i,jishu,0]=i2
                    family[i,jishu,1]=ju_li
                    if ju_li<=(lin_yu-(bian_chang/2)):
                        family[i,jishu,2]=1
                    else:
                        family[i,jishu,2]=(lin_yu+(bian_chang/2)-ju_li)/bian_chang
                    jishu+=1
        shu_xing[i,0]=jishu
        m=0
        for i2 in range(0,jishu):
            m+=1*(family[i,i2,1]**2)*ti_ji
        shu_xing[i,24]=m
        # print('点，键，m',i,jishu,m)
    print('邻域搜索完成')

    #时间积分
    jishu=0
    for tt in range(0,shi_jian_bu):
        #初始化总力
        for i in range(0,zong_dian_shu):
            shu_xing[i,4]=0
            shu_xing[i,5]=0
            shu_xing[i,6]=0
        # 总力
        for i in range(0, zong_dian_shu):
            # 距离新，e
            for i2 in range(0,int(shu_xing[i,0])):
                bian_hao=int(family[i,i2,0])
                ju_li_xin=sqrt((shu_xing[i,1]+shu_xing[i,13]-shu_xing[bian_hao,1]-shu_xing[bian_hao,13])**2+
                               (shu_xing[i,2]+shu_xing[i,14]-shu_xing[bian_hao,2]-shu_xing[bian_hao,14])**2+
                               (shu_xing[i,3]+shu_xing[i,15]-shu_xing[bian_hao,3]-shu_xing[bian_hao,15])**2)

                family[i,i2,5]=ju_li_xin
                family[i,i2,3]=ju_li_xin-family[i,i2,1]
                family[i, i2, 8] = ((mi_du) * 9.8 * (chang_z - (shu_xing[i, 3] + shu_xing[bian_hao, 3]) / 2) * np.tan(pi / 6) + 10000) / (((10 ** 1.027) * ((chang_z - shu_xing[i, 3]) ** 0.483)) * 1000000)
                family[i, i2, 11] = 0.0003 * (chang_z - (shu_xing[i, 3] + shu_xing[bian_hao, 3]) / 2) + 0.001975

                cos = ((shu_xing[bian_hao, 1] - shu_xing[i, 1]) * (shu_xing[bian_hao, 1] + shu_xing[bian_hao, 13] - shu_xing[i, 1] - shu_xing[i, 13]) +
                       (shu_xing[bian_hao, 2] - shu_xing[i, 2]) * (shu_xing[bian_hao, 2] + shu_xing[bian_hao, 14] - shu_xing[i, 2] - shu_xing[i, 14]) +
                       (shu_xing[bian_hao, 3] - shu_xing[i, 3]) * (shu_xing[bian_hao, 3] + shu_xing[bian_hao, 15] - shu_xing[i, 3] - shu_xing[i, 15])) / (family[i, i2, 1] * family[i, i2, 5])
                family[i, i2, 9] = ((family[i, i2, 5] - family[i, i2, 1]) * cos) / family[i, i2, 1]
                family[i, i2, 10] = (family[i, i2, 5] - family[i, i2, 1]) * ((1 - cos ** 2) ** 0.5) / family[i, i2, 1]

            #theta(x)
            theta=0
            for i2 in range(0,int(shu_xing[i,0])):
                theta+=1*family[i,i2,1]*family[i,i2,3]*ti_ji
            shu_xing[i,25]=(3/shu_xing[i,24])*theta
            #theta(x')
            for i2 in range(0,int(shu_xing[i,0])):
                bian_hao=int(family[i,i2,0])
                theta2=0
                for i3 in range(0,int(shu_xing[bian_hao,0])):
                    # theta2+=1*family[bian_hao,i3,1]*family[bian_hao,i3,3]*ti_ji#////
                    theta2+=1*family[i,i2,1]*family[i,i2,3]*ti_ji#////
                family[i,i2,6]=(3/shu_xing[bian_hao,24])*theta2

            #ed(x),ed(x')
            for i2 in range(0,int(shu_xing[i,0])):
                family[i,i2,4]=family[i,i2,3]-((shu_xing[i,25]*family[i,i2,1])/3)
                family[i,i2,7]=family[i,i2,3]-((family[i,i2,6]*family[i,i2,1])/3)

            #t(x),t(x')
            for i2 in range(0,int(shu_xing[i,0])):
                bian_hao = int(family[i, i2, 0])
                if damage[i,i2]==0:
                    t_1=(-3*(-shu_xing[i, 21]*shu_xing[i,25]+1*shu_xing[i,23])/shu_xing[i,24])*1*family[i,i2,1]+\
                      (15*shu_xing[i, 22]/shu_xing[i,24])*1*family[i,i2,4]
                    t_2=(-3*(-shu_xing[bian_hao, 21]*family[i,i2,6]+1*shu_xing[bian_hao,23])/shu_xing[bian_hao,24])*1*family[i,i2,1]+\
                      (15*shu_xing[bian_hao, 22]/shu_xing[bian_hao,24])*1*family[i,i2,7]
                    t=t_1+t_2
                elif damage[i,i2]==1:
                    t=0
                shu_xing[i,4]+=t*((shu_xing[bian_hao,1]+shu_xing[bian_hao,13]-shu_xing[i,1]-shu_xing[i,13])/family[i,i2,5])*ti_ji*family[i,i2,2]
                shu_xing[i,5]+=t*((shu_xing[bian_hao,2]+shu_xing[bian_hao,14]-shu_xing[i,2]-shu_xing[i,14])/family[i,i2,5])*ti_ji*family[i,i2,2]
                shu_xing[i,6]+=t*((shu_xing[bian_hao,3]+shu_xing[bian_hao,15]-shu_xing[i,3]-shu_xing[i,15])/family[i,i2,5])*ti_ji*family[i,i2,2]

            #损伤
            jishu2 = 0
            for i2 in range(0,int(shu_xing[i,0])):
                if (family[i,i2,10]>family[i,i2,8] or family[i,i2,9]>family[i,i2,11] or damage[i,i2]==1) and \
                        (10*bian_chang<=shu_xing[i,1]<=chang_x-10*bian_chang and \
                         4*bian_chang<=shu_xing[i,2]<=chang_y-4*bian_chang and \
                         shu_xing[i,3]>1.1):
                    damage[i,i2]=1
                    jishu2+=1
            shu_xing[i,19]=jishu2/shu_xing[i,0]

        # 加速度，速度，位移
        for i in range(0,zong_dian_shu):
            # if 3*bian_chang<=shu_xing[i,3] and \
            #         3*bian_chang<shu_xing[i,1]<chang_x-3*bian_chang and \
            #         3*bian_chang<shu_xing[i,2]<chang_y-3*bian_chang and \
            #         shu_xing[i,26]!=4:
            shu_xing[i,7]=shu_xing[i,4]/shu_xing[i,20]
            shu_xing[i,8]=shu_xing[i,5]/shu_xing[i,20]
            shu_xing[i,9]=shu_xing[i,6]/shu_xing[i,20]

            shu_xing[i,10]+=shu_xing[i,7]*dt
            shu_xing[i,11]+=shu_xing[i,8]*dt
            shu_xing[i,12]+=shu_xing[i,9]*dt

            shu_xing[i,13]+=shu_xing[i,10]*dt
            shu_xing[i,14]+=shu_xing[i,11]*dt
            shu_xing[i,15]+=shu_xing[i,12]*dt

        #边界条件
        #四周
        for i in range(0, zong_dian_shu):
            if shu_xing[i,1]<=bian_chang*3 or shu_xing[i,1]>=chang_x-bian_chang*3:
                shu_xing[i,13]=0
                # shu_xing[i,14]=0
                # shu_xing[i,15]=0
            if shu_xing[i,2]<=bian_chang*3 or shu_xing[i,2]>=chang_y-bian_chang*3:
                # shu_xing[i,13]=0
                shu_xing[i,14]=0
                # shu_xing[i,15]=0
        #底部
        for i in range(0,zong_dian_shu):
            if shu_xing[i,3]<=bian_chang*3:
                shu_xing[i,13]=0
                shu_xing[i,14]=0
                shu_xing[i,15]=0
        #三角
        for i in range(0,zong_dian_shu):
            if shu_xing[i,26]==4:
                shu_xing[i,13]=0
                shu_xing[i,14]=0
                shu_xing[i,15]=0
        #孔隙水压变化
        for i in range(0,zong_dian_shu):
            if shu_xing[i,26]!=4:
                # if tt<=500*2:
                if 0.25+3*bian_chang<shu_xing[i,3]<1.3+3*bian_chang:
                    if tt<=((1.3+3*bian_chang-shu_xing[i,3])/1.05)*shi_jian_bu:
                        shu_xing[i,23]=-1000*10*(1.3+3*bian_chang-shu_xing[i,3])*tt/(((1.3+3*bian_chang-shu_xing[i,3])/1.05)*shi_jian_bu)
                    else:
                        shu_xing[i,23]=-1000*10*(1.3+3*bian_chang-shu_xing[i,3])
                elif shu_xing[i,3]<=0.25+3*bian_chang:
                    shu_xing[i,23]=-1000*10*1.05*tt/shi_jian_bu
                elif 1.3+3*bian_chang<=shu_xing[i,3]:
                    shu_xing[i,23]=0
                # if tt>500*2:
                #     if 0.25+3*bian_chang<shu_xing[i,3]<1.3+3*bian_chang:
                #         shu_xing[i,23]=-1000*10*(1.3+3*bian_chang-shu_xing[i,3])
                #     elif shu_xing[i,3]<=0.25+3*bian_chang:
                #         shu_xing[i,23]=-1000*10*1.05
                #     elif 1.3+3*bian_chang<=shu_xing[i,3]:
                #         shu_xing[i,23]=0
                # if tt<=500:
                #     shu_xing[i,23]=1000*10*1.05*tt/500
                # if tt>500:
                #     shu_xing[i,23]=1000*10*1.05
        # 新坐标
        for i in range(0, zong_dian_shu):
            shu_xing[i,16]=shu_xing[i,13]+shu_xing[i,1]
            shu_xing[i,17]=shu_xing[i,14]+shu_xing[i,2]
            shu_xing[i,18]=shu_xing[i,15]+shu_xing[i,3]

        # 输出
        if tt % jian_ge == 0:
            for i in range(0, zong_dian_shu):
                shuchu[i, 0, jishu] = shu_xing[i, 16]
                shuchu[i, 1, jishu] = shu_xing[i, 17]
                shuchu[i, 2, jishu] = shu_xing[i, 18]
                shuchu[i, 3, jishu] = shu_xing[i, 19]
                shuchu[i, 4, jishu] = shu_xing[i, 13]
                shuchu[i, 5, jishu] = shu_xing[i, 14]
                shuchu[i, 6, jishu] = shu_xing[i, 15]
                shuchu[i, 7, jishu] = shu_xing[i, 26]
            # print(jishu)
            jishu += 1

        print(tt)
    return shuchu

# 传参
shuchu=main()

# 输出到外部文件夹储存
lujing = './../python外部存储/南京实验室模型/'
for i in range(0,shu_chu_shu):
    np.savetxt(lujing + '时间步为%g.txt' % i, shuchu[:, :, i])

# 运行时间
t2 = time()
print((t2 - t1)/3600,'小时')

