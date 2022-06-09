def Memebership(x):
    Ln = LN(x)
    Ze = ZE(x)
    Lp = LP(x)
    return Ln,Ze,Lp
    # return Ln,Mn,Sn,Ze,Sp,Mp,Lp

def LN(x):
    y = 0
    if x <= -0.05:
        y = 1.0
    elif -0.05 < x < 0:
        y = (0-x)/0.05
    elif x >= 0:
        y = 0.0
    return y

def ZE(x):
    y = 0
    if x <= -0.05 or x >= 0.05:
        y = 0.0
    elif -0.05 < x <= 0:
        y = (x+0.05)/0.05
    elif 0 < x < 0.05:
        y = (0.05-x)/0.05
    return y

def LP(x):
    y = 0
    if x <= 0:
        y = 0.0
    elif 0 < x < 0.05:
        y = (x- 0)/0.05
    elif x >= 0.05:
        y = 1.0
    return y


if __name__ == '__main__':
    x = -0.046187180999358
    Ln,Ze,Lp = Memebership(x)
    print('Ln:',Ln)
    print('Ze:',Ze)
    print('Lp:',Lp)