# 病毒性肺炎重症风险计算器 — Web 应用

> 本工具为「急性呼吸道病毒感染重症病例预警模型构建」(杭州市科技发展计划项目，2023–2025) 的临床落地工具。
> 基于 642 例多中心训练数据 (树兰 + 市中 + 流感)，提供 **Logistic 列线图** + **XGBoost SHAP 解释** + **双模型对比** 三种视角。

---

## 一、本地启动（推荐先用）

### 1. 准备 Python 环境（建议 Python ≥ 3.9）

```bash
# 进入应用目录
cd output/scripts/04_app

# （可选）建立虚拟环境
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动 Web 服务

```bash
streamlit run nomogram_streamlit_app.py --server.port 8501
```

### 3. 浏览器访问

启动后会自动打开 `http://localhost:8501`。
如未自动打开，手动访问该地址即可。

> **首次加载约 5–8 秒**（模型 + SHAP explainer 缓存初始化），之后所有交互都是毫秒级。

---

## 二、应用功能

### Tab 1：🩺 列线图 (Logistic)
- 13 项实验室指标 → 重症概率 + 风险层化（低 / 中 / 高）
- **评分卡**：每个变量对 logit 的贡献量，按绝对值降序
- **横向贡献条形图**：直观看哪些指标推高/降低风险
- **临床建议**：分档给出监护层级建议

### Tab 2：🔬 SHAP 解释器 (XGBoost)
- 同输入 → XGBoost 概率
- **SHAP 瀑布图**：从平均基线 E[f(x)] 出发，每变量按 SHAP 值逐步演进到本患者预测 f(x)
- **个例 SHAP 表**：所有变量的 SHAP 值与方向，按 |SHAP| 降序

### Tab 3：⚖️ 双模型对比
- Logistic 与 XGBoost 概率并排显示
- **一致性自动评估**：高度一致 / 区段一致但数值有差 / 模型分歧 → 三档自动判定
- 概率条形并排 + 阈值线（0.30 / 0.70）

### 通用功能（左侧 sidebar）
- **按器官系统分组**：基本信息 / 血常规 / 炎症 / 肝功 / 肾功 / 心肌酶
- **示例患者**：一键加载「重症典型」或「轻症典型」演示
- **重置默认**：所有变量回到训练集中位数

---

## 三、目录结构

```
04_app/
├── nomogram_streamlit_app.py       # 主程序（Streamlit）
├── requirements.txt                 # Python 依赖
├── README.md                        # 本文件
├── nomogram_shiny_app.R             # R Shiny 备份版本（仅 Tab1 列线图）
└── assets/
    ├── hero_banner.svg              # 顶部装饰 SVG（肺+病毒+神经网络）
    ├── icons.py                     # 分组图标 SVG 集合
    └── logistic_spec.json           # Logistic 系数 + 训练集分布元数据
```

---

## 四、模型依赖

应用需访问以下两个已训练模型文件（路径相对本目录）：

```
../../03_models/Logistic.pkl     # sklearn Pipeline: IterativeImputer → StandardScaler → LogisticRegression
../../03_models/XGBoost.pkl      # sklearn Pipeline: IterativeImputer → StandardScaler → XGBClassifier
```

如需移动或独立部署，请将这两个文件随本目录一起拷贝并相应调整 `nomogram_streamlit_app.py` 中的 `MODEL_DIR` 常量。

---

## 五、公网部署（可选，等账号准备好后做）

### 方案 A：Streamlit Community Cloud（推荐，免费）
1. 注册 [streamlit.io/cloud](https://streamlit.io/cloud)（用 GitHub 登录）
2. 将整个 `04_app/` + `03_models/Logistic.pkl` + `03_models/XGBoost.pkl` 推送到 GitHub public repo
3. 在 Cloud 控制台 → New app → 选择 repo + 主文件 `nomogram_streamlit_app.py`
4. 等 3–5 分钟构建 → 获得公网 URL（形如 `https://xxx.streamlit.app`）

### 方案 B：Hugging Face Spaces（备选，免费）
1. 注册 [huggingface.co](https://huggingface.co)
2. 创建 Space → SDK 选 Streamlit → 上传所有文件
3. 自动部署，5–10 分钟可用

### 方案 C：自建服务器（医院内网/云主机）
```bash
nohup streamlit run nomogram_streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
# 配合 nginx 反向代理 + HTTPS 证书部署到内网域名
```

---

## 六、R Shiny 备份版本

`nomogram_shiny_app.R` 提供仅 **列线图 (Tab1)** 的等效实现，从 Logistic 系数硬编码生成，
适用于无 Python 环境但有 R 的使用场景。启动方式：

```bash
Rscript -e "shiny::runApp('nomogram_shiny_app.R', port=3838)"
```

R 版本仅作降级备份，**SHAP 与双模型对比不在 R 实现**——因 R 难以加载 Python 训练的 XGBoost pickle。

---

## 七、引用

如本工具用于学术发表或临床应用，请引用：

> 杭州市科技发展计划项目研究组. 多中心病毒性肺炎重症风险预测与个体化解释模型 [研究报告]. 2023–2025.

---

## 八、免责

本工具为**辅助临床决策参考**，不能替代医师诊断。所有诊疗决策应结合患者完整病史、体征、动态实验室与影像学综合判断。
