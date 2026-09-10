# -*- coding: utf-8 -*-
"""
客户流失预测系统（方案一 · PyTorch 部分）
================================================
基于项目一的电商订单数据，用 PyTorch 搭建一个多层感知机(MLP)，
预测客户是否处于"流失风险"状态（近期未复购），并评估模型效果。

运行前需安装：  pip install torch pandas numpy scikit-learn
本脚本与数据文件 ecommerce_sales.csv 放在同一目录。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 0. 配置 ============
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EPOCHS = 200
LR = 0.01
HIDDEN_DIM = 32
TEST_SIZE = 0.2
CHURN_DAYS = 30

# ============ 1. 加载数据 ============
df = pd.read_csv('ecommerce_sales.csv')
df['订单日期'] = pd.to_datetime(df['订单日期'])
ref_date = df['订单日期'].max()
print('数据规模: %d 条订单, %d 位客户' % (len(df), df['客户ID'].nunique()))
print('参考日期(最近购买):', ref_date.date())

# ============ 2. 特征工程（客户级聚合）============
g = df.groupby('客户ID')
feat = pd.DataFrame({
    'F_订单数':        g['订单号'].count(),
    'M_总销售额':       g['销售额'].sum(),
    '平均客单价':       g['销售额'].mean(),
    '平均每单件数':     g['数量'].mean(),
    '平均折扣率':       g['折扣率'].mean(),
    '有折扣占比':       g.apply(lambda x: (x['折扣率'] > 0).mean()),
    '购买类别数':       g['产品类别'].nunique(),
    '购买渠道数':       g['销售渠道'].nunique(),
    'R_最近购买距今天数': g['订单日期'].apply(lambda x: (ref_date - x.max()).days),
})
feat = feat.reset_index()
print('客户数:', len(feat), '| 特征维度:', len(feat.columns) - 1)

# ============ 3. 定义标签（流失风险）============
# R > CHURN_DAYS 天未复购 -> 1（流失风险），否则 0
feat['label'] = (feat['R_最近购买距今天数'] > CHURN_DAYS).astype(int)
print('正类(流失风险)占比: %.1f%%' % (feat['label'].mean() * 100))

# ============ 4. 标准化 & 划分 ============
feature_cols = ['F_订单数', 'M_总销售额', '平均客单价', '平均每单件数', '平均折扣率',
                '有折扣占比', '购买类别数', '购买渠道数']
X = feat[feature_cols].values.astype(np.float32)
y = feat['label'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# 记录测试集对应的原始行索引（用于输出风险客户名单）
test_feat_idx = np.where(np.isin(feat.index, feat.index))[0]
# 更稳妥：直接保存一个"是否测试集"标志
mask = np.zeros(len(feat), dtype=bool)
# 用 train_test_split 的分层索引还原：重新打一次拿到测试索引
# 简单做法：shuffle 后按比例切（与上面 stratify 一致性的说明见下）
# 为避免复杂还原，这里重新对**全体**预测，并单独输出测试集评估结果

# ============ 5. 定义 MLP ============
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1], HIDDEN_DIM)
print('\n模型结构:\n', model)

# ============ 6. 训练 ============
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).unsqueeze(1)
X_test_t = torch.from_numpy(X_test)

print('\n=== 开始训练 (PyTorch MLP, %d 轮) ===' % EPOCHS)
losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print('Epoch %3d | loss %.4f' % (epoch, loss.item()))

# ============ 7. 评估 ============
model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)
    probs = torch.sigmoid(logits_test).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

print('\n=== 测试集评估 ===')
print('准确率 : %.3f' % accuracy_score(y_test, preds))
print('AUC    : %.3f' % roc_auc_score(y_test, probs))
print('混淆矩阵:\n', confusion_matrix(y_test, preds))
print('\n分类报告:\n', classification_report(y_test, preds, target_names=['正常', '流失风险']))

# ============ 8. 特征重要性（输入层权重绝对值之和，粗略）============
with torch.no_grad():
    w = model.net[0].weight.abs().sum(dim=0).numpy()
print('\n=== 各特征重要性(输入层权重绝对值之和) ===')
for name, v in sorted(zip(feature_cols, w), key=lambda t: -t[1]):
    print('  %s: %.3f' % (name, v))

# ============ 9. 对全体客户输出流失概率 ============
with torch.no_grad():
    full_X = scaler.transform(feat[feature_cols].values.astype(np.float32))
    full_probs = torch.sigmoid(model(torch.from_numpy(full_X))).numpy().flatten()
feat['流失概率'] = full_probs
feat = feat.sort_values('流失概率', ascending=False).reset_index(drop=True)

risk = feat[feat['流失概率'] >= 0.5].copy()
print('\n=== 预测有流失风险客户: %d 位 (占比 %.1f%%) ===' % (len(risk), len(risk) / len(feat) * 100))
print(risk[['客户ID', 'F_订单数', 'M_总销售额', 'R_最近购买距今天数', '流失概率']].head(10).to_string(index=False))

feat.to_csv('churn_result.csv', index=False, encoding='utf-8-sig')
risk.to_csv('churn_risk_customers.csv', index=False, encoding='utf-8-sig')
print('\n结果已保存: churn_result.csv (全部客户), churn_risk_customers.csv (风险客户)')

# ============ 10. 训练 loss 曲线 ============
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), losses)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('训练 Loss 曲线')
plt.tight_layout(); plt.savefig('training_loss.png', dpi=150)
print('训练曲线已保存: training_loss.png')
