#-*- coding:utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        PearsonUserNeighCF
# Purpose:     Personalized Recommendation
#
# Author:      Jinkun Wang
# Email:       wangjinkun90@foxmail.com, if you have any question about the
#              code, please do not hesitate to contact me.
#
# Created:     10/09/2014
# Copyright:   (c) Jinkun Wang 2014
#-------------------------------------------------------------------------------
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    trainSet = {}
    testSet = {}
    movieUser = {}
    u2u = {}

    TrainFile = 'ml-100k/u1.base'   #Ö¸¶¨ÑµÁ·¼¯
    TestFile = 'ml-100k/u1.test'    #Ö¸¶¨²âÊÔ¼¯

    #¼ÓÔØÑµÁ·¼¯£¬Éú³ÉµçÓ°ÓÃ»§µÄµ¹ÅÅÐò±í movieUser
    for line in open(TrainFile):
        (userId, itemId, rating, _) = line.strip().split('\t')
        trainSet.setdefault(userId,{})
        trainSet[userId].setdefault(itemId,float(rating))
        movieUser.setdefault(itemId,[])
        movieUser[itemId].append(userId.strip())

    #·ÀÖ¹²âÊÔ¼¯ÓÐÑµÁ·¼¯ÖÐÃ»ÓÐ³öÏÖ¹ýµÄÏîÄ¿
    item_in_train = []
    for m in movieUser.keys():
        item_in_train.append(m)

    #¼ÓÔØ²âÊÔ¼¯
    for line in open(TestFile):
        (userId, itemId, rating, _) = line.strip().split('\t')
        testSet.setdefault(userId,{})
        testSet[userId].setdefault(itemId,float(rating))

    return trainSet,testSet,movieUser,item_in_train

#¼ÆËãÒ»¸öÓÃ»§µÄÆ½¾ùÆÀ·Ö
def getAverageRating(user):
    average = (sum(trainSet[user].values()) * 1.0) / len(trainSet[user].keys())
    return average

#¼ÆËãÓÃ»§ÏàËÆ¶È
def UserSimPearson(trainSet):
    userSim = {}
    for u1 in trainSet.keys():
        userSim.setdefault(u1,{})
        u1_rated = trainSet[u1].keys()
        for u2 in trainSet.keys():
            userSim[u1].setdefault(u2,0)
            if u1 != u2:
                u2_rated = trainSet[u2].keys()
                co_rated = list(set(u1_rated).intersection(set(u2_rated)))
                if co_rated == []:
                    userSim[u1][u2] = 0
                else:
                    num = 0     #Æ¤¶ûÑ·¼ÆËã¹«Ê½µÄ·Ö×Ó²¿·Ö
                    den1 = 0    #Æ¤¶ûÑ·¼ÆËã¹«Ê½µÄ·ÖÄ¸²¿·Ö1
                    den2 = 0    #Æ¤¶ûÑ·¼ÆËã¹«Ê½µÄ·ÖÄ¸²¿·Ö2
                    sigma_u1_m = 0  #¼ÆËãÓÃ»§u1¶Ô¹²Í¬ÆÀ¼ÛÏîÄ¿µÄÆÀ·Ö¾ùÖµ
                    sigma_u2_m = 0  #¼ÆËãÓÃ»§u2¶Ô¹²Í¬ÆÀ¼ÛÏîÄ¿µÄÆÀ·Ö¾ùÖµ
                    for m in co_rated:
                        sigma_u1_m += trainSet[u1][m]
                        sigma_u2_m += trainSet[u2][m]
                    ave_u1_m = sigma_u1_m / len(co_rated)
                    ave_u2_m = sigma_u2_m / len(co_rated)

                    for m in co_rated:
                        num += (trainSet[u1][m] - ave_u1_m) * (trainSet[u2][m] - ave_u2_m) * 1.0
                        den1 += pow(trainSet[u1][m] - ave_u1_m, 2) * 1.0
                        den2 += pow(trainSet[u2][m] - ave_u2_m, 2) * 1.0
                    den1 = sqrt(den1)
                    den2 = sqrt(den2)
                    if den1 == 0 or den2 ==0 :
                        userSim[u1][u2] = 0
                    else:
                        userSim[u1][u2] = num / (den1 * den2)
            else:
                userSim[u1][u2] = 0
    return userSim

#¶ÔÓÃ»§ÏàËÆ¶È±í½øÐÐÅÅÐò´¦Àí
def sortSimMatrix(userSim):
    neighSorted = {}
    for u in userSim.keys():
        neigh_sorted = sorted(userSim[u].items(), key = lambda x:x[1], reverse = True)
        for key, value in neigh_sorted:
            neighSorted.setdefault(u,[])
            neighSorted[u].append(key)
    return neighSorted

#Ñ°ÕÒÓÃ»§×î½üÁÚ²¢Éú³ÉÍÆ¼ö½á¹û£»Óë²âÊÔ¼¯±È½Ï»ñµÃËã·¨µÄ×¼È·¶È
def getAccuracyMetric(N,trainSet,testSet,movieUser,neighSorted, userSim, item_in_train):
    #Ñ°ÕÒÓÃ»§×î½üÁÚ²¢Éú³ÉÍÆ¼ö½á¹û
    pred = {}
    for user, item in testSet.items():    #¶Ô²âÊÔ¼¯ÖÐµÄÃ¿¸öÓÃ»§
        pred.setdefault(user,{})    #Éú³ÉÓÃ»§UserµÄÔ¤²â¿ÕÁÐ±í
        ave_u_rating = getAverageRating(user)
        neigh_uninterseced = neighSorted[user] #»ñÈ¡ÓÃ»§userµÄÁÚ¾Ó¼¯ºÏ£¨ÒÑ°´ÏàËÆ¶È´óÐ¡½µÐòÅÅÁÐ£©
        for m in item.keys():
            if m not in item_in_train:
                pred[user][m] = ave_u_rating
            else:
                rated_m_user = movieUser[m]         #²âÊÔ¼¯ÖÐÆÀ¼Û¹ýµçÓ°mµÄÓÃ»§
                neigh_intersected = sorted(rated_m_user,key = lambda x:neigh_uninterseced.index(x))
                if len(neigh_intersected) > N:
                    neigh = neigh_intersected[0:N]
                else:
                    neigh = neigh_intersected
                neighRating = 0
                neighSimSum = 0
                for neighUser in neigh:
                    neighRating += (trainSet[neighUser][m] - getAverageRating(neighUser)) * userSim[user][neighUser]
                    neighSimSum += abs(userSim[user][neighUser])
                if neighSimSum == 0:
                    pred[user][m] = ave_u_rating
                else:
                    pred[user][m] = ave_u_rating + (neighRating * 1.0) / neighSimSum

    #Óë²âÊÔ¼¯±È½Ï»ñµÃËã·¨µÄ×¼È·¶È
    mae = 0
    rmse = 0
    error_sum = 0
    sqrError_sum = 0
    setSum = 0
    for user,item in pred.items():
        for m in item.keys():
            error_sum += abs(pred[user][m] - testSet[user][m])
            sqrError_sum += pow(pred[user][m] - testSet[user][m],2)
            setSum += 1
    mae = error_sum / setSum
    rmse = sqrt(sqrError_sum / setSum)
    return mae, rmse

if __name__ == '__main__':

    print 'ÕýÔÚ¼ÓÔØÊý¾Ý...'
    trainSet,testSet,movieUser,item_in_train = loadData()

##    print 'ÕýÔÚ¼ÆËãÓÃ»§¼äÏàËÆ¶È...'
##    userSim = UserSimPearson(trainSet)

    '''print '¶ÔÏàËÆ¶ÈÁÐ±í°´ÏàËÆ¶È´óÐ¡½øÐÐÅÅÁÐ...'
    neighSorted = sortSimMatrix(userSim)

    print 'ÕýÔÚÑ°ÕÒ×î½üÁÚ...'
    NeighborSize = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    MAE = []
    RMSE = []
    for N in NeighborSize:            #¶Ô²»Í¬µÄ½üÁÚÊý
        mae, rmse = getAccuracyMetric(N,trainSet,testSet,movieUser,neighSorted, userSim, item_in_train)   #»ñµÃËã·¨ÍÆ¼ö¾«¶È
        MAE.append(mae)
        RMSE.append(rmse)
    plt.subplot(211)
    plt.plot(NeighborSize,MAE)
    plt.xlabel('NeighborSize')
    plt.ylabel('Mean Absolute Error')
    plt.title('Pearson User Neighbor Model Collaborative Filtering')

    plt.subplot(212)
    plt.plot(NeighborSize,RMSE)
    plt.xlabel('NeighborSize')
    plt.ylabel('Root Mean Square Error')
    plt.title('Pearson User Neighbor Model Collaborative Filtering')

    plt.show()
    raw_input('°´ÈÎÒâ¼ü¼ÌÐø...')'''