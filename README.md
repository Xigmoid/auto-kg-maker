# Auto-KG-Maker 知识图谱构建工具 🕸️

Auto-KG-Maker 基于大语言模型从任意领域文献中自动提取实体和关系，构建结构化知识图谱。

> 本项目基于 [graph_maker](https://github.com/rahulnyk/graph_maker) ，进一步添加了以下功能：
> - **本地部署模型支持**：支持通过 vllm 在本地部署 LLM 并自动化完成知识图谱构建，无需依赖外部 API
> - **自动化本体标注**：支持基于 LLM 的自动生成领域本体，减少手动配置工作
> - **预定义领域本体**：提供医学、法律、金融等多个领域的预定义本体，开箱即用

## ✨ 功能特性

- 🎯 **通用领域支持**：适用于医学、法律、金融、科技等任意领域的知识图谱构建
- 🔧 **灵活自定义**：支持自定义实体类型和关系定义，适配特定领域需求
- 📄 **智能文档处理**：支持滑动窗口切分，可配置窗口大小和步长
- 🚀 **本地模型支持**：兼容 OpenAI API 格式的本地部署模型
- 📊 **结构化输出**：JSONL 格式，便于后续分析和可视化

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本用法

```bash
python main.py -i ./input_docs -o ./output_kg
```

这会：
1. 从 `./input_docs` 目录读取所有 `.md` 文件
2. 使用内置的默认本体识别实体和关系
3. 将结果保存到 `./output_kg/graph.jsonl`

### 3. 完整示例（医学领域）

```bash
python main.py \
    -i ./medical_docs \
    -o ./knowledge_graph \
    --model qwen3 \
    --base-url http://localhost:8000/v1 \
    --chunk-size 4096 \
    --stride 4000 \
    --verbose
```

---

## ⚙️ 参数说明

### 必需参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--input-dir` | `-i` | 输入文档目录 | `./input_docs` |
| `--output-dir` | `-o` | 输出结果目录 | `./output_kg` |

### LLM 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `qwen3` | LLM 模型名称 |
| `--base-url` | `http://localhost:8000/v1` | LLM API 基础 URL |
| `--api-key` | `ollama` | API 密钥（本地部署通常不需要真实 key） |
| `--temperature` | `0.1` | 生成温度（0.0-1.0），越低越确定 |
| `--top-p` | `0.5` | Top-p 采样参数（0.0-1.0） |

**推荐配置**：
- 确定性任务（如实体识别）：`temperature=0.1`, `top_p=0.5`
- 创造性任务：`temperature=0.7`, `top_p=0.9`

### 文档处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk-size` | `2048` | 文档切分窗口大小（字符数） |
| `--stride` | `2000` | 滑动窗口步长（字符数） |
| `--max-length` | `8000` | 单个文档最大处理长度（字符数） |
| `--file-pattern` | `*.md` | 输入文件匹配模式（支持通配符） |

**切分策略说明**：
- `chunk_size=2048, stride=2000`：相邻片段有 48 字符重叠
- 较大的 chunk：适合长文档，但可能丢失细节
- 较小的 chunk：精度高，但可能丢失上下文

**常用配置**：
```bash
# 精细切分（适合短篇文献）
--chunk-size 1024 --stride 1000

# 标准切分（适合指南/共识）
--chunk-size 2048 --stride 2000

# 粗粒度切分（适合长篇综述）
--chunk-size 4096 --stride 4000
```

### 图谱构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--delay-s` | `10` | API 请求间隔时间（秒），防止速率限制 |
| `--verbose` | `False` | 显示详细输出信息 |

### 本体定义参数 ⭐️

| 参数 | 说明 |
|------|------|
| `--ontology-config` | 本体配置文件路径（JSON 格式），从 `ontologies/` 目录加载 |
| `--auto-ontology` | **使用 LLM 自动生成领域本体** |
| `--domain` | 领域类型（general/medical/legal/financial 等） |
| `--num-entity-types` | 自动生成本体时的实体类型数量（默认 10，范围 5-20） |

---

## 💡 多领域使用示例

### 示例 1：医学知识图谱

```bash
python main.py \
    -i ./hypertension_guidelines \
    -o ./hypertension_kg \
    --model qwen3 \
    --chunk-size 2048 \
    --stride 2000 \
    --delay-s 15
```

### 示例 2：法律知识图谱

```bash
python main.py \
    -i ./court_cases \
    -o ./legal_kg \
    --ontology-config ontologies/legal_ontology.json \
    --file-pattern "*.txt" \
    --max-length 10000
```

### 示例 3：金融企业关系图谱

```bash
python main.py \
    -i ./annual_reports \
    -o ./corporate_kg \
    --chunk-size 4096 \
    --stride 4000 \
    --model gpt-4 \
    --base-url https://api.openai.com/v1
```

### 示例 4：学术论文知识图谱

```bash
python main.py \
    -i ./research_papers \
    -o ./academic_kg \
    --file-pattern "*.pdf.txt" \
    --max-length 15000 \
    --chunk-size 3072 \
    --stride 3000
```

### 示例 5：LLM 自动生成 AI 领域本体 ⭐️

```bash
python main.py \
    -i ./ai_papers \
    -o ./ai_kg \
    --auto-ontology \
    --domain artificial_intelligence \
    --num-entity-types 10
```

**输出示例**：
```
🤖 正在使用 LLM 生成 artificial_intelligence 领域的本体...
✅ LLM 生成本体完成：10 种实体类型，8 种关系

生成的实体类型:
  1. Company
  2. Person
  3. Product
  4. Event
  5. Technology
  ...
```

---

## 🔧 高级配置

### 三种本体定义方式

#### 方法 1: 使用内置默认本体

```bash
# 通用领域
python main.py -i ./docs -o ./kg

# 医学领域
python main.py -i ./medical_docs -o ./kg --domain medical
```

**优点**：无需配置，立即开始

#### 方法 2: 加载自定义配置文件 ⭐️

```bash
python main.py \
    -i ./legal_docs \
    -o ./legal_kg \
    --ontology-config ontologies/legal_ontology.json
```

**预置本体**（在 `ontologies/` 目录）：
- `legal_ontology.json` - 法律领域（10 种实体，8 种关系）
- `financial_ontology.json` - 金融领域（8 种实体，8 种关系）
- `example_ontology.json` - 通用模板

**优点**：精确控制，可复用，适合生产环境

#### 方法 3: LLM 自动生成 ⭐️⭐️⭐️

让大模型根据你的文档自动生成本体定义：

```bash
python main.py \
    -i ./research_papers \
    -o ./academic_kg \
    --auto-ontology \
    --domain academic \
    --num-entity-types 12
```

**优势**：
- 🤖 智能适配领域特点
- 🎯 自动识别核心概念
- 💾 生成的 JSON 可保存复用

**工作流程**：
1. 根据领域名称构建专业 prompt
2. 调用 LLM 生成 JSON 格式的本体定义
3. 解析响应并创建 Ontology 对象
4. 显示生成的实体类型和关系摘要

### 推荐工作流

**探索阶段** → **优化阶段** → **生产阶段**

```bash
# 1. 探索：LLM 生成初稿
python main.py -i ./docs -o ./tmp --auto-ontology --domain custom

# 2. 优化：手动调整生成的 JSON
# 编辑 ontologies/custom_ontology.json

# 3. 生产：使用固定配置
python main.py -i ./docs -o ./final --ontology-config ontologies/custom_ontology.json
```

---

## 📊 输出格式

输出文件为 `graph.jsonl`，每行是一个 JSON 对象，表示一条关系边：

```json
{"head": {"label": "Disease", "text": "高血压"}, "relation": "treated_by", "tail": {"label": "Drug", "text": "ACEI"}}
{"head": {"label": "Symptom", "text": "发热"}, "relation": "indicates", "tail": {"label": "Disease", "text": "新冠肺炎"}}
{"head": {"label": "Drug", "text": "阿司匹林"}, "relation": "has_side_effect", "tail": {"label": "SideEffect", "text": "胃肠道出血"}}
```

---

## ⚠️ 注意事项

1. **API 速率限制**：建议使用 `--delay-s` 参数避免请求过快
2. **显存需求**：本地部署模型需要足够的 GPU 显存（Qwen3 约需 14GB）
3. **文档编码**：确保输入文件为 UTF-8 编码
4. **输出覆盖**：如果输出目录已存在 `graph.jsonl`，会被覆盖

---

## 🐛 常见问题

### Q: 找不到任何文件？
**A**: 检查 `--input-dir` 路径是否正确，或调整 `--file-pattern` 匹配你的文件格式。

### Q: 输出结果为空？
**A**: 
- 检查 LLM 服务是否正常运行
- 尝试调大 `--chunk-size`
- 降低 `--temperature` 提高确定性

### Q: 处理速度太慢？
**A**:
- 减小 `--chunk-size` 减少单次处理量
- 增加 `--delay-s` 避免 API 限流
- 使用更快的模型或 GPU

### Q: 如何选择合适的本体定义方式？
**A**:
- **快速测试**：使用内置默认本体
- **常见领域（法律/金融）**：使用 `ontologies/` 中的预置配置
- **新兴/交叉领域**：使用 LLM 自动生成
- **生产环境**：手动优化后的配置文件

### Q: LLM 生成的本体在哪里保存？
**A**: 运行时会在控制台输出完整的 JSON，可以复制到 `ontologies/` 目录下保存复用。

---

## 🙏 致谢

本项目基于 **[graph_maker](https://github.com/rahulnyk/graph_maker)** 开发，感谢原作者 [rahulnyk](https://github.com/rahulnyk) 的优秀工作。

在 graph_maker 的基础上，本项目进行了以下增强：
- ✅ 完整的命令行参数支持
- ✅ 灵活的文档切分和预处理
- ✅ 多领域本体支持（默认医学，支持自定义和 LLM 生成）
- ✅ 改进的错误处理和日志输出
- ✅ 详细的中文文档和示例

---

## 📝 更新日志

### v1.1.0 (2026-03-06) - 最新版本
- ✨ **新增**: LLM 自动生成领域本体功能 (`--auto-ontology`)
- ✨ **新增**: 领域参数 (`--domain`) 支持通用/医学/法律/金融等领域
- ✨ **新增**: 实体数量控制参数 (`--num-entity-types`)
- 🔧 **改进**: 重构本体初始化逻辑，支持三种灵活的本体定义方式
- 📚 **新增**: `ontologies/` 文件夹存放预置本体配置
- 📚 **新增**: 法律和金融领域示例配置文件

### v1.0.0 (2026-03-05)
- ✅ 完整的命令行参数支持
- ✅ 内置 20 种医学实体类型
- ✅ 滑动窗口文档切分
- ✅ 支持本地 LLM 部署
- ✅ JSONL 格式输出

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

特别欢迎：
- 🌟 分享你所在领域的本体配置文件（放到 `ontologies/` 目录）
- 🐛 报告 Bug 或提出改进建议
- 📚 补充更多使用示例

---

## 📄 许可证

MIT License

---

## 📞 联系方式

如有问题，请提交 Issue 或联系开发者。

---

**Star History**: 如果这个项目对你有帮助，请给个 Star ⭐️

**Original Project**: [graph_maker](https://github.com/rahulnyk/graph_maker)
