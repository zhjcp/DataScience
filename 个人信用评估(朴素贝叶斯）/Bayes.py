import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# 导入高斯模型
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
    变量default表示贷款是否违约，也是我们需要预测的目标变量
    
    实验思路：
        1、 分析数据，自变量有7列连续型、13列离散型
        2、 尝试高斯模型（适合连续值）
        3、 尝试多项式模型（适合离散值）
        4、 尝试用AdaBoost来优化贝叶斯模型
'''

# step1: 数据探索和预处理
print("step1: 数据探索和预处理")
credit = pd.read_csv("./credit.csv")
# 样本数量和维数
print("样本数量和维数:")
print(credit.shape)
# ==> 通过观察csv文件，可知只有savings_balance和checking_balance两列有缺失值
# savings_balance列的缺失值数量
savings_balance_values_ravel = credit.savings_balance.values.ravel()
print("savings_balance的缺失值数量: \n" + str(len(savings_balance_values_ravel[savings_balance_values_ravel == 'unknown'])))
# checking_balance列的缺失值数量
checking_balance_values_ravel = credit.checking_balance.values.ravel()
print(
    "checking_balance的缺失值数量: \n" + str(len(checking_balance_values_ravel[checking_balance_values_ravel == 'unknown'])))
# default的值，即：违约贷款的值
# 分析输出结果后发现：正负样本(违约和未违约)严重不均衡，所以在后面我们会修改 正负样本的权值系数
print("default的值，即：违约贷款的值: \n" + str(credit.default.value_counts()))
print("\n \n")

# step2: 将数据中字符串形式的数据编码成数字
print("step2: 将数据中字符串形式的数据编码成数字")
cols = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status',
        'other_debtors', 'property', 'installment_plan', 'housing', 'job', 'telephone', 'foreign_worker']
col_dicts = {'checking_balance': {'1 - 200 DM': 2,
                                  '< 0 DM': 1,
                                  '> 200 DM': 3,
                                  'unknown': 0},
             'credit_history': {'critical': 0,
                                'delayed': 2,
                                'fully repaid': 3,
                                'fully repaid this bank': 4,
                                'repaid': 1},
             'employment_length': {'0 - 1 yrs': 1,
                                   '1 - 4 yrs': 2,
                                   '4 - 7 yrs': 3,
                                   '> 7 yrs': 4,
                                   'unemployed': 0},
             'foreign_worker': {'no': 1, 'yes': 0},
             'housing': {'for free': 1, 'own': 0, 'rent': 2},
             'installment_plan': {'bank': 1, 'none': 0, 'stores': 2},
             'job': {'mangement self-employed': 3,
                     'skilled employee': 2,
                     'unemployed non-resident': 0,
                     'unskilled resident': 1},
             'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},
             'personal_status': {'divorced male': 2,
                                 'female': 1,
                                 'married male': 3,
                                 'single male': 0},
             'property': {'building society savings': 1,
                          'other': 3,
                          'real estate': 0,
                          'unknown/none': 2},
             'purpose': {'business': 5,
                         'car (new)': 3,
                         'car (used)': 4,
                         'domestic appliances': 6,
                         'education': 1,
                         'furniture': 2,
                         'others': 8,
                         'radio/tv': 0,
                         'repairs': 7,
                         'retraining': 9},
             'savings_balance': {'101 - 500 DM': 2,
                                 '501 - 1000 DM': 3,
                                 '< 100 DM': 1,
                                 '> 1000 DM': 4,
                                 'unknown': 0},
             'telephone': {'none': 1, 'yes': 0}}
for col in cols:
    credit[col] = credit[col].map(col_dicts[col])
print("\n \n")

# step3: 数据离散化（有3列连续型数据、17列离散型数据）
print("step3: 数据离散化")
#       months_loan_duration
print("months_loan_duration的离散化")
print("months_loan_duration的最大值：" + str(credit['months_loan_duration'].values.max()))
print("months_loan_duration的最大值：" + str(credit['months_loan_duration'].values.min()))
data1 = credit['months_loan_duration'].copy()
k1 = 5
kmodel1 = KMeans(n_clusters=k1)  # 建立模型
kmodel1.fit(data1.values.reshape((len(data1), 1)))  # 训练模型
c1 = pd.DataFrame(kmodel1.cluster_centers_).sort_values(0)  # 输出聚类中心，并且排序
print("聚类中心：")
print(kmodel1.cluster_centers_)
w1 = c1.rolling(2).mean().iloc[1:]  # 相邻两项求中点，作为边界点
w1 = [0] + list(w1[0]) + [data1.max()]  # 把首末边界点加上，w[0]中0为列索引
d1 = pd.cut(data1, w1, labels=['0', '1', '2', '3', '4'])
credit['months_loan_duration'] = data1
print(d1.head(10))

#       amount
print("amount的离散化")
print("amount的最大值：" + str(credit['amount'].values.max()))
print("amount的最大值：" + str(credit['amount'].values.min()))
data2 = credit['amount'].copy()
k2 = 10
kmodel2 = KMeans(n_clusters=k2)  # 建立模型
kmodel2.fit(data2.values.reshape((len(data2), 1)))  # 训练模型
c2 = pd.DataFrame(kmodel2.cluster_centers_).sort_values(0)  # 输出聚类中心，并且排序
print("聚类中心：")
print(kmodel2.cluster_centers_)
w2 = c2.rolling(2).mean().iloc[1:]  # 相邻两项求中点，作为边界点
w2 = [0] + list(w2[0]) + [data2.max()]  # 把首末边界点加上，w[0]中0为列索引
d2 = pd.cut(data2, w2, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
credit['amount'] = data2
print(d2.head(10))

#       age
print("age的离散化")
print("age的最大值：" + str(credit['age'].values.max()))
print("age的最大值：" + str(credit['age'].values.min()))
data3 = credit['age'].copy()
# 等距离散化，各个类比依次命名为0,1,2,3
d3 = pd.cut(data3, 4, labels=range(4))
credit['age'] = data3
print(d3.head())
print("\n")


# step4：数据标准化（为PCA做准备）
print("step4：数据标准化（为PCA做准备）")
# 标准化
#   iloc的参数：前面是行号范围，后面是列号范围（区间是左闭右开区间）
#       iloc[:, :-1]表示：取所有行、除了最后一列的所有列
#                    最后一列default是样本标签
X = credit.iloc[:, :-1]
y = credit['default']
#   随机化序列
np.random.seed(123)
perm = np.random.permutation(len(X))
#   loc的参数：感兴趣的行号序列
X = X.loc[perm]
y = y[perm]
#   preprocessing.scale：沿着某个轴标准化数据集，以均值为中心，以分量为单位方差
#       由于多项式模型不能处理负数，所以数据标准化时要注意
X = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
print("\n")


# step5：执行PCA处理
print("# step5：执行PCA处理")
#   PCA模型：将样本自变量从20维降到12维 权衡了性能和计算量之后的选择
pca = PCA(copy=True, n_components=12, whiten=False, random_state=1)
X_new = pca.fit_transform(X)
#   输出信息
print(u'所保留的n个主成分的方差贡献率为：')
print(pca.explained_variance_ratio_)
print(u'排名前3的主成分特征向量为：')
print(pca.components_[0:3])
print(u'累计方差贡献率为：')
print(sum(pca.explained_variance_ratio_))
print(u'降维后的数据规模和维数：')
print(X_new.shape)
print(u'降维后的数据：')
print(X_new)
X_new = preprocessing.MinMaxScaler(feature_range=(1, 2)).fit_transform(X_new)
print(u"降维后的数据再次映射后的数据：")
print(X_new)


# step6：分割训练集和测试集的数据
print("step6：分割训练集和测试集的数据")
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1)
#   观察训练集和测试集中的贷款违约比例，两者的比例应该都接近 7:3
print(y_train.value_counts() / len(y_train))
print(y_test.value_counts() / len(y_test))
print("\n")


# step7: 训练模型
print("step7: 训练模型")
#  一、生成样本的权重系数
y_train_list = list(y_train)
sample_weight_Gauss = []
sample_weight_Multinomial = []
sample_weight_Bernoulli = []
sample_weight_AdaBoostGau = []
sample_weight_AdaBoostMul = []
sample_weight_AdaBoostBer = []
for i in range(len(y_train_list)):
    if y_train_list[i] == 1:
        sample_weight_Gauss.append(1)
        sample_weight_Multinomial.append(1)
        sample_weight_Bernoulli.append(1)
        sample_weight_AdaBoostGau.append(1)
        sample_weight_AdaBoostMul.append(1)
        sample_weight_AdaBoostBer.append(1)
    if y_train_list[i] == 2:
        # 2.3 = 7/3 暂时认为可以刚好抵消样本数量差异带来的不良影响
        sample_weight_Gauss.append(2.56)
        # 特别注意：
        #       多项式模型 对样本分布不均匀 十分敏感，在不调整样本数量比例时，只能跳转样本权重
        #           微小的调整，就会导致多项式模型的预测结果有很大的变动
        sample_weight_Multinomial.append(2.2635122)
        #       伯努利模型 只要样本权重有微小的变化，就会产生两个极端的结果
        sample_weight_Bernoulli.append(2.2558139545)
        sample_weight_AdaBoostGau.append(2.65)
        sample_weight_AdaBoostMul.append(2.262)
        sample_weight_AdaBoostBer.append(2.65)
#     二、 普通贝叶斯模型
credit_model_Gau = GaussianNB()
credit_model_Gau.fit(X_train, y_train, sample_weight=sample_weight_Gauss)
credit_model_Mul = MultinomialNB()
credit_model_Mul.fit(X_train, y_train, sample_weight=sample_weight_Multinomial)
#   三、 AdaBoost + 贝叶斯模型
#       AdaBoost受到样本分布不均匀的影响实在太大了，
#       即使调整了正负样本的权重，作用还是不明显
credit_model_AdaBoostGauss = AdaBoostClassifier(base_estimator=credit_model_Gau, n_estimators=4)
credit_model_AdaBoostGauss.fit(X_train, y_train, sample_weight=sample_weight_AdaBoostGau)
credit_model_AdaBoostMul = AdaBoostClassifier(base_estimator=credit_model_Mul, n_estimators=4)
credit_model_AdaBoostMul.fit(X_train, y_train, sample_weight=sample_weight_AdaBoostMul)
print("\n")


# step9: 使用测试集对贝叶斯分类模型进行测试
print("step9: 使用测试集对贝叶斯分类模型进行测试")
credit_pred_Gau = credit_model_Gau.predict(X_test)
credit_pred_Mul = credit_model_Mul.predict(X_test)
credit_pred_AdaBoostGau = credit_model_AdaBoostGauss.predict(X_test)
credit_pred_AdaBoostMul = credit_model_AdaBoostMul.predict(X_test)
print("\n")


# step10: 分析决策树的效果
print("step10: 朴素贝叶斯的效果")
print("======> 高斯模型的预测结果：")
# print("test")
# print(list(y_test))
# print("pred")
# print(credit_pred_Gau)
print(metrics.classification_report(y_test, credit_pred_Gau))
print(metrics.confusion_matrix(y_test, credit_pred_Gau))
print(metrics.accuracy_score(y_test, credit_pred_Gau))
print("======> 多项式模型的预测结果：")
# print("test")
# print(list(y_test))
# print("pred")
# print(credit_pred_Mul)
print(metrics.classification_report(y_test, credit_pred_Mul))
print(metrics.confusion_matrix(y_test, credit_pred_Mul))
print(metrics.accuracy_score(y_test, credit_pred_Mul))

print("\n")
print("========集成后的模型性能=======")
print("\n")

print("======> AdaBoost + Gau 的预测结果：")
print(metrics.classification_report(y_test, credit_pred_AdaBoostGau))
print(metrics.confusion_matrix(y_test, credit_pred_AdaBoostGau))
print(metrics.accuracy_score(y_test, credit_pred_AdaBoostGau))
print("======> AdaBoost + Mul 的预测结果：")
print(metrics.classification_report(y_test, credit_pred_AdaBoostMul))
print(metrics.confusion_matrix(y_test, credit_pred_AdaBoostMul))
print(metrics.accuracy_score(y_test, credit_pred_AdaBoostMul))
