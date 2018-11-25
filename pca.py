import numpy as np

def zeroMean(datamat):
        mean=np.mean(datamat,axis=0)
        newdata=datamat-mean
        return newdata,mean
#Calculate pca, return the calculated eigenvalues and eigenvectors,
#and the eigenvalues and eigenvectors have been sorted back 
#in order of largest to smallest
def pca(datamat):
        newdata,mean=zeroMean(datamat)
        covmat=np.cov(newdata,rowvar=0)
        eigvals,eigvects=np.linalg.eig(np.mat(covmat))
        eigvals_sort=np.argsort(eigvals)
        eigvals_sort=eigvals_sort[-1::-1]
        eigval_final=eigvals[eigvals_sort]
        eigvect_final=eigvects[:,eigvals_sort]

        return eigval_final,eigvect_final

#Returns the number of features of the previous percentage and their percentage
#in order of largest to smallest
def percent_percentage(eigvals,percentage):
        sumarray = sum(eigvals)
        sumtmp = 0
        num = 0
        for i in eigvals:
                sumtmp = sumtmp+i
                num=num+1
                if sumtmp >= sumarray*percentage:
                        break
        result_percent=np.zeros(num)
        for i in range(num):
                result_percent[i] = eigvals[i]/sumarray

        return num,result_percent

#Returns the percentage of all eigenvalues
def percent_all(eigvals):
        s,percent = percent_percentage(eigvals,1)
        return percent

#Calculate the number of eigenvalues over 10%
def percent_over_10(eigvals):
        s,percent = percent_percentage(eigvals,1)
        num = 0
        for i in percent:
                if i > 0.1:
                        num=num+1
        percent = percent[:num]
        return num,percent

#Calculate the reduced dimensional data
def product(datamat,eigvect,num):

        eigvect_final = eigvect[:,0:num]
        low_data = np.dot(datamat,eigvect_final)

        return low_data

