# import pandas as pd
# import numpy as np
# from sklearn import tree
# from sklearn import preprocessing
# from sklearn.decomposition import PCA
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import train_test_split
# from six import StringIO
# from IPython.display import Image
# import pydot
# from sklearn import metrics
#
# # step1: 数据探索和预处理
# print("step1: 数据探索和预处理")
# credit = pd.read_csv("./credit.csv")
# X_neg_samples = credit[credit["default"] == 1].sample(300)
# credit_new = pd.concat([X_neg_samples, credit[credit["default"] == 2].sample(300)])
# print(credit_new.sample(10))
# print(credit_new.shape)
#
#
# # step2: 将数据中字符串形式的数据编码成数字
# #   否则，数据标准化时，processing函数会报错
# print("step2: 将数据中字符串形式的数据编码成数字")
# cols = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status',
#         'other_debtors', 'property', 'installment_plan', 'housing', 'job', 'telephone', 'foreign_worker']
# col_dicts = {'checking_balance': {'1 - 200 DM': 2,
#                                   '< 0 DM': 1,
#                                   '> 200 DM': 3,
#                                   'unknown': 0},
#              'credit_history': {'critical': 0,
#                                 'delayed': 2,
#                                 'fully repaid': 3,
#                                 'fully repaid this bank': 4,
#                                 'repaid': 1},
#              'employment_length': {'0 - 1 yrs': 1,
#                                    '1 - 4 yrs': 2,
#                                    '4 - 7 yrs': 3,
#                                    '> 7 yrs': 4,
#                                    'unemployed': 0},
#              'foreign_worker': {'no': 1, 'yes': 0},
#              'housing': {'for free': 1, 'own': 0, 'rent': 2},
#              'installment_plan': {'bank': 1, 'none': 0, 'stores': 2},
#              'job': {'mangement self-employed': 3,
#                      'skilled employee': 2,
#                      'unemployed non-resident': 0,
#                      'unskilled resident': 1},
#              'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},
#              'personal_status': {'divorced male': 2,
#                                  'female': 1,
#                                  'married male': 3,
#                                  'single male': 0},
#              'property': {'building society savings': 1,
#                           'other': 3,
#                           'real estate': 0,
#                           'unknown/none': 2},
#              'purpose': {'business': 5,
#                          'car (new)': 3,
#                          'car (used)': 4,
#                          'domestic appliances': 6,
#                          'education': 1,
#                          'furniture': 2,
#                          'others': 8,
#                          'radio/tv': 0,
#                          'repairs': 7,
#                          'retraining': 9},
#              'savings_balance': {'101 - 500 DM': 2,
#                                  '501 - 1000 DM': 3,
#                                  '< 100 DM': 1,
#                                  '> 1000 DM': 4,
#                                  'unknown': 0},
#              'telephone': {'none': 1, 'yes': 0}}
# for col in cols:
#     credit_new[col] = credit_new[col].map(col_dicts[col])
# print("\n \n")
#
#
# # step3：数据标准化（为PCA做准备）
# print("# step3：数据标准化（为PCA做准备）")
# # 标准化
# #   iloc的参数：前面是行号范围，后面是列号范围（区间是左闭右开区间）
# #       iloc[:, :-1]表示：取所有行、除了最后一列的所有列
# #                    最后一列default是样本标签
# X = credit_new.iloc[:, :-1]
# y = credit_new['default']
# #   随机化序列
# np.random.seed(123)
# perm = np.random.permutation(len(X))
# #   loc的参数：感兴趣的行号序列
# X = X.loc[X.index.intersection(perm)]
# y = y[y.index.intersection(perm)]
# #   preprocessing.scale：沿着某个轴标准化数据集，以均值为中心，以分量为单位方差
# X = preprocessing.scale(X)
# print("\n \n")
#
#
# # step4：执行PCA处理
# print("# step4：执行PCA处理")
# #   PCA模型：将样本自变量从20维降到12维
# pca = PCA(copy=True, n_components=12, whiten=False, random_state=1)
# X_new = pca.fit_transform(X)
# #   输出信息
# print(u'所保留的n个主成分的方差贡献率为：')
# print(pca.explained_variance_ratio_)
# # print(u'排名前3的主成分特征向量为：')
# # print(pca.components_[0:3])
# print(u'累计方差贡献率为：')
# print(sum(pca.explained_variance_ratio_))
# print(u'降维后的数据规模和维数：')
# print(X_new.shape)
# print("\n \n")
#
#
# # step5：分割训练集和测试集的数据
# print("step5：分割训练集和测试集的数据")
# X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1)
# #   观察训练集和测试集中的贷款违约比例，两者的比例应该都接近 7:3
# print(y_train.value_counts() / len(y_train))
# print(y_test.value_counts() / len(y_test))
# print("\n \n")
#
#
# # step6: 训练模型
# print("step6: 训练模型")
# # 调参：max_depth取10效果最好
# # 【下面尝试使用多种分类算法】
#
# #  1. 初始选择的模型：普通的决策树
# #   正类和负类的权重比
# class_weights1 = {1: 1, 2: 3.7}
# credit_model1_common_tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=4, max_depth=11,
#                                                    class_weight=class_weights1)
# #  2. 改进后的模型：随机森林
# #   正类和负类的权重比
# class_weights2 = {1: 1, 2: 5.2}
# credit_model2_radom_forest = RandomForestClassifier(n_estimators=85, criterion='gini', min_samples_leaf=6,
#                                                     max_depth=9,  # =2时：5% 99%   =9时：60% 84%
#                                                     class_weight=class_weights2)
# #  3. 改进后的模型：AdaBoost
# #  分析输出结果可知：由于正样本的比例太大，AdaBoost为了提高正确率，会“打补丁”将更多的正样本预测对，相对地负样本的预测正确率就下降了
# #       特别是，n_estimators参数越大，补丁打得越多，正样本的预测正确率越高
# credit_model3_adaboost = AdaBoostClassifier(base_estimator=credit_model2_radom_forest, n_estimators=2)
#
# credit_model1_common_tree.fit(X_train, y_train)
# credit_model2_radom_forest.fit(X_train, y_train)
# credit_model3_adaboost.fit(X_train, y_train)
# print("\n \n")
#
#
# # # step7: 绘制决策树
# # 随机森林中的树比较多，没必要全部绘制出来
# print("step7: 绘制决策树")
# dot_data = StringIO()
# tree.export_graphviz(credit_model1_common_tree, out_file=dot_data,
#                      # feature_names=X_train.columns,
#                      class_names=['no default', 'default'],
#                      filled=True, rounded=True,
#                      special_characters=True)
# (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())  # pydot 1.2.0 以上版本
# # graph = pydot.graph_from_dot_data(dot_data.getvalue()) # pydot 1.0 版本以下
# Image(graph.create_png())
# print("\n \n")
#
#
# # step8: 使用测试集对决策树模型进行测试
# print("step8: 使用测试集对决策树模型进行测试")
# credit_pred1_common_tree = credit_model1_common_tree.predict(X_test)
# credit_pred2_radom_forest = credit_model2_radom_forest.predict(X_test)
# credit_pred3_adaboost = credit_model3_adaboost.predict(X_test)
# print("\n \n")
#
#
# # step9: 分析决策树的效果
# print("step9: 分析决策树的效果")
# print("======> 普通决策树的预测结果：")
# print(metrics.classification_report(y_test, credit_pred1_common_tree))
# print(metrics.accuracy_score(y_test, credit_pred1_common_tree))
# print("======> 随机森林的预测结果：")
# print(metrics.classification_report(y_test, credit_pred2_radom_forest))
# print(metrics.accuracy_score(y_test, credit_pred2_radom_forest))
# print("======> AdaBoost的预测结果：")
# print(metrics.classification_report(y_test, credit_pred3_adaboost))
# print(metrics.accuracy_score(y_test, credit_pred3_adaboost))
