import warnings

import numpy as np
import tqdm

def read_sample(path):
    with open(path) as f:
        a = f.read()
        data_ = a.split('\n')[0].split(' ')[:-1]
        label_ = a.split('\n')[1].split(' ')[:-1]
        data,label = np.zeros(len(data_)),np.zeros(len(label_))
        for i in range(len(data_)):
            data[i] = int(data_[i])
        for i in range(len(label_)):
            label[i] = int(label_[i])
        data = data.reshape([-1,3])
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = -1
    return data,label

class Irelief:
    def __init__(self,data,label):
        self.x = data
        self.y = label
        if len(self.x) == len(self.y):
            self.N = len(self.x)
        else:
            warnings.warn('x和y的长度不一致,len(x)='+str(len(self.x))+',len(y)='+str((self.y)))
        self.get_MnHn()
        self.get_Sn()
        self.o = np.zeros_like(self.y)
        self.w = np.ones(self.x.shape[1]) / self.x.shape[1]
        print('初始化完成')

    ''' 初始化相关函数  start '''
    def get_MnHn(self):
        Mn = []
        Hn = []
        for i,y_i in enumerate(self.y):
            if y_i == 1:
                Hn.append(i)
            elif y_i == -1:
                Mn.append(i)
        self.Mn = Mn
        self.Hn = Hn

    def get_Sn(self):
        Sn = np.zeros([len(self.y),2])
        for i,x in enumerate(self.x):
            Mlength = self.length__(self.x[self.Mn[0]], self.x[self.Hn[0]])  # 初始化
            Hlength = self.length__(self.x[self.Hn[0]], self.x[self.Hn[1]])
            if self.y[i] == 1:
                for x1 in self.Mn:
                    if self.length__(x,self.x[x1]) < Mlength:
                        Sn[i,0] = x1
                        Mlength = self.length__(x,self.x[x1])
                for x2 in self.Hn:
                    if x2 != i:
                        if self.length__(self.x[x2],x) < Hlength:
                            Sn[i,1] = x2
                            Hlength = self.length__(self.x[x2],x)
            elif self.y[i] == -1:
                for x1 in self.Hn:
                    if self.length__(x,self.x[x1]) < Mlength:
                        Sn[i,0] = x1
                        Mlength = self.length__(x,self.x[x1])
                for x2 in self.Mn:
                    if x2 != i:
                        if self.length__(self.x[x2],x) < Hlength:
                            Sn[i,1] = x2
                            Hlength = self.length__(self.x[x2],x)
        self.Sn = Sn

    # get_Sn子函数  计算距离
    def length__(self,x1,x2):
        # 计算向量之间的距离
        return np.sum(np.abs(x1-x2))
    ''' 初始化相关函数  end '''

    # 论文中的公式8,9,10
    # 公式8
    def get_Pm(self,n,i,fun):
        if self.y[n] == 1:
            Mn = self.Mn
        elif self.y[n] == -1:
            Mn = self.Hn.copy()
        else:
            warnings.warn('y[n]不等于1或-1!')
            assert self.y[n] == 1 or self.y[n] == -1
        if i not in Mn:
            return 0
        molecule = fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[i]),self.w)))
        denominator = 0
        for j in Mn:
            denominator += fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[j]), self.w)))
        return molecule/denominator

    # 公式9
    def get_Ph(self,n,i,fun):
        if self.y[n] == 1:
            Hn = self.Hn
        elif self.y[n] == -1:
            Hn = self.Mn.copy()
            Hn.remove(n)
        else:
            warnings.warn('y[n]不等于1或-1!')
            assert self.y[n] == 1 or self.y[n] == -1
        if i not in Hn:
            return 0
        molecule = fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[i]), self.w)))
        denominator = 0
        for j in Hn:
            denominator += fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[j]), self.w)))
        return molecule / denominator

    # 公式10
    def get_o(self,n,fun):
        if self.y[n] == 1:
            Mn = self.Mn
        elif self.y[n] == -1:
            Mn = self.Hn.copy()
        molecule = 0
        for i in Mn:
            molecule += fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[i]),self.w)))
        denominator = 0
        for i in range(len(self.x)):
            if i != n:
                denominator += fun(np.sum(np.multiply(np.abs(self.x[n] - self.x[i]),self.w)))
        return molecule / denominator

    # 核函数
    def k_fun(self,d,sigma=2):
        return np.exp(-d/sigma)

    # 公式11
    def get_m_n(self,n):
        if self.y[n] == 1:
            Mn = self.Mn
        elif self.y[n] == -1:
            Mn = self.Hn.copy()
        m_n = []
        for j in range(len(self.w)):
            s = 0
            for i in Mn:
                m_in = np.multiply(np.abs(self.x[n] - self.x[i]), self.w)
                m_inj = m_in[j]
                alpha_in = self.get_Pm(n,i,fun=self.k_fun)
                s += alpha_in * m_inj
            m_n.append(s)
        return np.array(m_n)

    # 公式11
    def get_h_n(self,n):
        if self.y[n] == 1:
            Hn = self.Hn
        elif self.y[n] == -1:
            Hn = self.Mn.copy()
            Hn.remove(n)
        h_n = []
        for j in range(len(self.w)):
            s = 0
            for i in Hn:
                h_in = np.multiply(np.abs(self.x[n] - self.x[i]), self.w)
                m_inj = h_in[j]
                alpha_in = self.get_Ph(n,i,fun=self.k_fun)
                s += alpha_in * m_inj
            h_n.append(s)
        return np.array(h_n)

    # 公式11
    def get_gamma_n(self,n):
        return 1 - self.get_o(n,fun=self.k_fun)


    def epsilon_init(self,T):
        return np.linspace(0,1,T)

    # 公式24
    def get_pi(self):
        s = np.zeros(len(self.w))
        for n in range(self.N):
            s += self.get_gamma_n(n) * (self.get_m_n(n) * self.get_h_n(n))
        return s / self.N

    # 公式28下面带+
    def v_add(self,v):
        for i,val in enumerate(v):
            if val <= 0:
                v[i] = 0
        return v

    # 公式28
    def mainloop(self,T,theta):
        ver,v = 1,np.zeros(len(self.w))
        epsilon = self.epsilon_init(T + 1)
        for t in tqdm.tqdm(range(T + 1)):
            pi = self.get_pi()
            v = v + (ver/ver + 1) * (pi - v)
            ver = ver / (epsilon[t + 1] * (ver + 1))
            w = self.v_add(v) / np.linalg.norm(self.v_add(v))
            if self.length__(self.w,w) < theta:
                return w
            else:
                self.w = w
        return self.w

def main():
    data,y = read_sample('data.txt')
    irelief = Irelief(data[:100],y[:100])
    w = irelief.mainloop(10,0.01)
    print(w)

if __name__ == '__main__':
    main()

