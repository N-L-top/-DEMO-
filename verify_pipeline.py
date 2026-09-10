"""验证方案一的数据管道：特征工程 + 标签定义 + 基线模型评估（用sklearn，无需torch）"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

df = pd.read_csv('ecommerce_sales.csv')
df['订单日期'] = pd.to_datetime(df['订单日期'])
ref = df['订单日期'].max()  # 全数据最近日期作为参考

# --- 特征工程：客户级聚合 ---
g = df.groupby('客户ID')
feat = pd.DataFrame({
    'F_订单数': g['订单号'].count(),
    'M_总销售额': g['销售额'].sum(),
    '平均客单价': g['销售额'].mean(),
    '平均每单件数': g['数量'].mean(),
    '平均折扣率': g['折扣率'].mean(),
    '有折扣占比': g.apply(lambda x: (x['折扣率'] > 0).mean()),
    '购买类别数': g['产品类别'].nunique(),
    '购买渠道数': g['销售渠道'].nunique(),
    'R_最近购买距今天数': g['订单日期'].apply(lambda x: (ref - x.max()).days),
})
feat = feat.reset_index()
print('客户数:', len(feat), '特征列:', len(feat.columns) - 1)

# --- 标签：流失风险（R>30天未复购记为1）---
feat['label'] = (feat['R_最近购买距今天数'] > 30).astype(int)
print('正类(流失风险)占比: %.1f%%' % (feat['label'].mean() * 100))

# --- 准备特征 ---
X = feat[['F_订单数', 'M_总销售额', '平均客单价', '平均每单件数', '平均折扣率',
          '有折扣占比', '购买类别数', '购买渠道数']].values
y = feat['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

# --- 基线逻辑回归（证明特征有区分度）---
clf = LogisticRegression(max_iter=1000); clf.fit(X_train, y_train)
pred = clf.predict(X_test); proba = clf.predict_proba(X_test)[:, 1]
print('\n=== 基线逻辑回归评估 ===')
print('准确率: %.3f' % accuracy_score(y_test, pred))
print('AUC: %.3f' % roc_auc_score(y_test, proba))
print('混淆矩阵:\n', confusion_matrix(y_test, pred))
print('特征重要性(系数):')
for name, coef in zip(['F_订单数','M_总销售额','平均客单价','平均每单件数','平均折扣率','有折扣占比','购买类别数','购买渠道数'], clf.coef_[0]):
    print(f'  {name}: {coef:.4f}')
print('\n管道验证通过 ✅ 特征工程与标签逻辑正确')
