# 公网部署完整指南 — Streamlit Community Cloud

> 全程预计 15–20 分钟。免费、无服务器维护、自动 HTTPS、自动从 GitHub 同步更新。

---

## 总览：3 步走

```
[1] GitHub 注册 + 创建 repo  →  [2] 推送代码  →  [3] Streamlit Cloud 一键部署
                                                    ↓
                                            获得公网 URL
                                    https://xxx.streamlit.app
```

---

## 第 1 步：准备 GitHub 账号与 repo

### 1.1 注册 GitHub（已有账号请跳过）
访问 [github.com/signup](https://github.com/signup) → 注册（用常用邮箱）。

### 1.2 创建 public repo
登录后右上角「+」→ **New repository**：
- **Repository name**：`viral-pneumonia-calculator`（或你喜欢的名字，建议英文）
- **Description**：`Multi-center viral pneumonia severity risk calculator (Logistic + XGBoost + SHAP)`
- **Visibility**：必须选 **Public**（Streamlit Cloud 免费版仅支持 public repo）
- **❌ 不要**勾选 "Add a README file"、"Add .gitignore"、"Choose a license"（我们已经准备好了，避免冲突）
- 点击 **Create repository**

创建后 GitHub 会显示一个空 repo 页面，**保留这个页面，下一步要用页面上提供的 URL**。

### 1.3 准备 GitHub 认证
首次 push 前 GitHub 会要求登录。推荐用 **Personal Access Token**：
- GitHub 右上角头像 → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
- Note：`streamlit-deploy`，Expiration：90 days，Scopes：勾选 ✅ **repo**（全部子项）
- 点击 **Generate token** → **复制并保存**这串 token（页面关闭就再也看不到了）

> 当 push 时被询问 password，**粘贴这个 token**（不是你的 GitHub 密码）。

---

## 第 2 步：把 04_app/ 推送到 GitHub

### 2.1 在终端运行（**把 `<YOUR_USERNAME>` 替换成你的 GitHub 用户名**）

```bash
# 进入应用目录
cd "/Users/elvisfang/Desktop/病毒性肺炎感染重症预测及预后模型构建/output/scripts/04_app"

# 初始化 git 仓库（如果还没初始化）
git init -b main

# 设置 git 用户信息（如果是第一次用 git）
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加所有文件
git add .

# 创建首次提交
git commit -m "init: 病毒性肺炎重症概率计算器 v1.0 (Streamlit + Logistic + XGBoost + SHAP)"

# 关联远程 repo（替换 YOUR_USERNAME）
git remote add origin https://github.com/<YOUR_USERNAME>/viral-pneumonia-calculator.git

# 推送
git push -u origin main
```

> 推送时被询问 username + password：username 填 GitHub 用户名，password 粘贴上一步的 **Personal Access Token**。

### 2.2 推送成功后
回到 GitHub repo 页面刷新，应该能看到所有文件（`nomogram_streamlit_app.py`、`models/`、`assets/`、`requirements.txt` 等）。

---

## 第 3 步：Streamlit Community Cloud 部署

### 3.1 注册并登录
访问 [share.streamlit.io](https://share.streamlit.io) → 点击 **Continue with GitHub** → 授权 Streamlit 访问你的 GitHub。

### 3.2 创建新应用
登录后点击右上角 **Create app** → 选择 **"Deploy a public app from GitHub"** → 填表：

| 字段 | 填写 |
|---|---|
| **Repository** | 下拉选 `<YOUR_USERNAME>/viral-pneumonia-calculator` |
| **Branch** | `main` |
| **Main file path** | `nomogram_streamlit_app.py` |
| **App URL (custom subdomain)** | `viral-pneumonia` 或你想要的英文标识 |

→ 点击 **Deploy!**

### 3.3 等待构建（首次约 3–5 分钟）
Streamlit Cloud 会：
1. 克隆你的 repo
2. 安装 `requirements.txt` 中的所有依赖（streamlit、shap、xgboost、scikit-learn 等）
3. 启动你的 `nomogram_streamlit_app.py`
4. 部署到公网

构建过程实时显示在右下角终端窗口。完成后会自动跳转到你的公网 app。

### 3.4 你的公网 URL
```
https://viral-pneumonia.streamlit.app
```
（具体 URL 取决于你设置的 subdomain）

把这个 URL 替换进结题报告中的 `[URL placeholder]` 位置。

---

## 常见问题

### Q1：构建失败 "ModuleNotFoundError: shap"
A：检查 `requirements.txt` 是否被推送上去（`git status` → `git ls-files`）。如果丢失，重新 `git add requirements.txt && git commit && git push`。

### Q2：构建成功但页面报 "FileNotFoundError: Logistic.pkl"
A：检查 `models/` 目录是否被推送（`.gitignore` 不会忽略它）。运行 `ls -la models/` 确认两个 pkl 文件都在，然后 `git add models/ && git commit && git push`。

### Q3：想更新代码 / 调整布局
A：修改本地代码 → `git add . && git commit -m "update: xxx" && git push`，Streamlit Cloud 自动检测到 push 并 1–2 分钟内重新部署。

### Q4：想限制只有自己/团队能访问
A：免费 Streamlit Cloud 仅支持 public app。如需私有，可考虑：
- 切换到 [Hugging Face Spaces](https://huggingface.co/spaces)（也免费，支持 private space）
- 或自建服务器 + nginx basic auth

### Q5：想用自定义域名（如 `nomogram.yourhospital.org`）
A：免费版仅支持 `.streamlit.app` 子域。自定义域名需要 Streamlit Cloud Teams 计划（付费），或迁移到 Hugging Face Spaces / 自建服务器 + DNS CNAME。

### Q6：app 闲置一段时间后访问变慢
A：免费版 Streamlit Cloud 会让长时间无访问的 app 进入休眠（cold start ~30 秒）。访问后自动唤醒。重要部署建议升级到付费 Teams 计划保持常驻。

---

## 部署清单（自查）

推送前确认以下文件齐全：

- [x] `nomogram_streamlit_app.py` — 主程序
- [x] `requirements.txt` — Python 依赖
- [x] `README.md` — 使用说明
- [x] `models/Logistic.pkl` — Logistic 模型 (137KB)
- [x] `models/XGBoost.pkl` — XGBoost 模型 (452KB)
- [x] `assets/logistic_spec.json` — 变量元数据
- [x] `assets/hero_banner.svg` — 顶部装饰
- [x] `assets/icons.py` — 分组图标
- [x] `.streamlit/config.toml` — Streamlit 主题配置
- [x] `.gitignore` — Git 忽略规则
- [x] `GITHUB_DEPLOY.md` — 本文件
