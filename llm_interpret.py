# -*- coding: utf-8 -*-
"""
客户流失预测系统（方案一 · 大模型 API 解读部分）
================================================
读取 churn_predictor.py 生成的流失预测结果（churn_risk_customers.csv），
调用大模型 API 自动生成一份面向业务的"客户流失分析与运营建议"报告。
"""

import json
import os
import requests
import pandas as pd

# ============ 0. 配置（按需修改）============
API_KEY = "在此填入API_KEY"          # API Key 填到这里

# ---- DeepSeek 官方 ----
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"

# ---- 如果你想用其它模型，可改成下面任意一组 ----
# 通义千问 Qwen：
#   API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
#   MODEL_NAME = "qwen-plus"
# 智谱清言：
#   API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
#   MODEL_NAME = "glm-4-flash"
# OpenAI：
#   API_URL = "https://api.openai.com/v1/chat/completions"
#   MODEL_NAME = "gpt-4o-mini"

# 分析用到的统计指标（会随 prompt 一起发给模型）
RISK_FILE = "churn_risk_customers.csv"     # 流失风险客户名单
RESULT_FILE = "churn_result.csv"           # 全部客户(含概率)
OUTPUT_FILE = "churn_interpretation.md"    # 输出的解读报告
TOP_N = 15                                 # 给模型看的重点客户数量

# ============ 1. 读取模型结果 ============
if not os.path.exists(RESULT_FILE):
    raise SystemExit("找不到 %s，请先运行 churn_predictor.py 生成结果。" % RESULT_FILE)

df_all = pd.read_csv(RESULT_FILE)
print('→ 读取全部客户结果: %d 位客户' % len(df_all))

# 汇总统计（供大模型参考）
total = len(df_all)
risk_count = int((df_all['流失概率'] >= 0.5).sum())
pos_rate = risk_count / total * 100
avg_amount = df_all['M_总销售额'].mean()
avg_orders = df_all['F_订单数'].mean()
avg_recent = df_all['R_最近购买距今天数'].mean()

# 找出 Top 风险客户（概率最高）
top_risk = df_all.sort_values('流失概率', ascending=False).head(TOP_N)
top_text = "\n".join(
    "- 客户ID %s | 消费%s单 | 累计金额%.0f元 | 最近购买%s天前 | 流失概率%.1f%%" % (
        r['客户ID'], int(r['F_订单数']), r['M_总销售额'],
        int(r['R_最近购买距今天数']), r['流失概率'] * 100)
    for _, r in top_risk.iterrows()
)

summary = (
    "客户总数: %d 位\n"
    "预测流失风险客户: %d 位 (占比 %.1f%%)\n"
    "人均销售额: %.0f 元\n"
    "人均订单数: %.1f 单\n"
    "人均最近购买距今天数: %.1f 天\n"
    "\n以下为流失概率最高的 %d 位重点客户：\n%s"
) % (total, risk_count, pos_rate, avg_amount, avg_orders, avg_recent, TOP_N, top_text)

print('→ 模型统计摘要已生成')

# ============ 2. 组装 Prompt ============
system_prompt = (
    "你是一位资深的数据分析顾问，擅长电商客户运营。"
    "请根据给定的客户流失预测统计结果，写一份面向业务团队的客户流失分析报告。"
    "要求：1) 用通俗易懂的中文，避免过度技术化；2) 结构清晰，包含【总体概览】【核心发现】【流失原因推测】【运营建议】四个部分；"
    "3) 建议要具体、可落地，能直接指导运营或营销团队执行；4) 字数在 500~800 字之间。"
)
user_prompt = (
    "以下是我们电商平台客户流失预测模型的输出结果，请据此撰写分析报告。\n\n"
    "【统计摘要】\n%s\n\n"
    "请开始撰写分析报告。" % summary
)

# ============ 3. 调用大模型 API ============
if API_KEY == "在此填入API_KEY":
    raise SystemExit(
        "\n⚠️  还没有配置 API Key。\n"
        "请打开本文件 llm_interpret.py，找到第 40 行附近，\n"
        "把 API_KEY = \"在此填入你的API_KEY\"  替换成你自己的 Key。\n"
        "如果你还没有 API Key，可以去 https://platform.deepseek.com 注册获取（需要手机号）。"
    )

headers = {
    "Authorization": "Bearer " + API_KEY,
    "Content-Type": "application/json",
}
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    "temperature": 0.7,
    "max_tokens": 1500,
}

print('→ 正在调用大模型 API (%s) ...' % MODEL_NAME)
try:
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    report = data["choices"][0]["message"]["content"]
except Exception as e:
    raise SystemExit(
        "\n❌ 调用失败: %s\n"
        "可能原因：\n"
        "1. API Key 填错或余额不足\n"
        "2. 网络无法访问该 API 地址\n"
        "3. 接口地址/模型名写错\n\n"
        "请检查上面的【配置】部分，或把详细报错发给我帮你排查。" % e
    )

# ============ 4. 保存报告 ============
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("# 客户流失分析报告\n\n")
    f.write("> 本报告由大模型基于流失预测结果自动生成\n\n")
    f.write(report + "\n")

print('\n✅ 大模型解读报告已生成: %s' % OUTPUT_FILE)
print('━' * 50)
print(report)
print('━' * 50)
print('\n运行完成。你可以把 churn_interpretation.md 上传到 GitHub 作为交付物。')
