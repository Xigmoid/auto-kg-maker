# 本体配置文件目录 📚

本目录存放预定义领域的知识图谱本体配置文件。

## 可用的本体配置

### 1. legal_ontology.json - 法律领域
**实体类型**: 10 种  
**关系类型**: 8 种  

包括：Law, Case, Court, Judge, Lawyer, Defendant, Plaintiff, Crime, Penalty, Evidence

**使用命令**:
```bash
python main.py -i ./legal_docs -o ./legal_kg \
    --ontology-config ontologies/legal_ontology.json
```

### 2. financial_ontology.json - 金融领域
**实体类型**: 8 种  
**关系类型**: 8 种  

包括：Company, Person, Product, Market, Event, Asset, Institution, Regulation

**使用命令**:
```bash
python main.py -i ./financial_reports -o ./finance_kg \
    --ontology-config ontologies/financial_ontology.json
```

### 3. example_ontology.json - 通用模板
用于快速创建自定义领域本体的模板文件。

## 如何创建自定义本体

复制 `example_ontology.json` 并修改：

```bash
cp example_ontology.json my_domain_ontology.json
```

然后在编辑器中修改实体类型和关系定义。

## 文件格式

```json
{
  "labels": [
    {"EntityName1": "中文描述和示例"},
    {"EntityName2": "中文描述和示例"}
  ],
  "relationships": [
    "relation_type_1 (中文说明)",
    "relation_type_2 (中文说明)"
  ]
}
```

## 贡献新的本体

欢迎将你所在领域的本体配置提交到本目录！
