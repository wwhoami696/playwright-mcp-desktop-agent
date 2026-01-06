# Playwright MCP 桌面助手

一个 **基于 Python + PySide6 的桌面应用**，结合 **Playwright MCP + DeepSeek Reasoner**，实现「自然语言 → 自动操作浏览器」的智能桌面助手。

该项目同时支持 **桌面 GUI** 与 **命令行智能代理核心**，适合做自动化浏览、AI Agent、RPA、智能助手等方向的实验与产品原型。

---

## ✨ 功能特性

### 🖥 桌面端（desktop_app.py）
- 现代化聊天式 UI（深色主题，高对比度）
- 支持流式日志、状态提示、系统消息
- 异步 Agent 线程，UI 不阻塞
- 快捷键支持：`Ctrl + Enter` 发送
- 会话保存 / 清空 / 状态查看

### 🌐 智能浏览器代理（agent_core.py）
- 基于 **Playwright MCP** 的浏览器控制
- DeepSeek Reasoner 驱动的决策与推理
- 智能元素匹配（ref / 文本 / 属性 / 模糊匹配）
- 页面状态机（加载 / 可交互 / 阻挡 / 错误）
- 自动恢复策略（关闭弹窗 / 刷新 / 后退 / 重试）
- 操作循环检测与降级处理
- 完整的操作日志与性能统计

---

## 📁 项目结构

```
.
├─ desktop_app.py     # PySide6 桌面 UI 主程序
├─ agent_core.py      # Playwright MCP 智能代理核心
├─ README.md
├─ config.json        # （运行后生成）API Key 配置
├─ session.json       # （可选）会话保存文件
└─ agent.log          # （可选）运行日志
```

---

## 🚀 快速开始

### 1️⃣ 环境要求

- Python **3.10+**（推荐 3.11 / 3.12 / 3.13）
- Node.js **18+**（用于 Playwright MCP）

### 2️⃣ 安装 Python 依赖

```bash
pip install PySide6 openai mcp
```

### 3️⃣ 安装 Playwright MCP（首次必做）

```bash
npm install -g @playwright/mcp
npx playwright install chromium
```

---

## 🔑 API Key 配置（DeepSeek）

支持三种方式，**推荐环境变量**：

### ✅ 方式一：环境变量（推荐）

```bash
export DEEPSEEK_API_KEY=你的key   # macOS / Linux
setx DEEPSEEK_API_KEY 你的key     # Windows
```

### 方式二：桌面程序中输入

在 UI 顶部输入框填写，程序会自动保存到 `config.json`

### 方式三：手动配置 `config.json`

```json
{
  "api_key": "你的 DeepSeek API Key"
}
```


---

## 🖥 运行桌面程序

```bash
python desktop_app.py
```

启动后：
1. 输入 API Key（或已配置可留空）
2. 点击【连接】等待 Playwright MCP 就绪
3. 在下方输入自然语言任务

示例：
- `打开百度搜索今天的新闻`
- `访问 github.com 搜索 python playwright`
- `打开知乎搜索 人工智能`

---


## 🧠 Agent 工作流程说明

1. **browser_snapshot** 获取页面状态（强制）
2. LLM 分析页面元素（ref / 文本 / role）
3. 执行浏览器操作（点击 / 输入 / 滚动等）
4. 自动验证操作结果
5. 失败 → 进入恢复策略（弹窗 / 刷新 / 后退）
6. 检测循环 → 自动打断并降级



## 📌 适用场景

- AI Agent / AutoGPT 类项目
- RPA / 浏览器自动化
- LLM + 工具调用实验
- 桌面智能助手原型
- Playwright MCP 深度实践

---

## 📄 License

本项目仅用于 **学习 / 研究 / 原型验证**。

如用于商业用途，请自行评估：
- DeepSeek API 使用条款
- Playwright MCP 相关许可证

---

## ⭐ 致谢

- Playwright MCP
- DeepSeek Reasoner
- PySide6
- OpenAI SDK 接口规范

如果这个项目对你有帮助，欢迎 ⭐ Star

