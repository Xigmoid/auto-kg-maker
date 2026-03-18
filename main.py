#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱构建工具
从任意领域文献中自动提取实体和关系，构建知识图谱
支持自定义本体或 LLM 自动生成结构化本体定义
"""
import sys
# sys.path.insert(0, '/root/PycharmCC/Auto-KG-Maker')

from knowledge_graph_maker import GraphMaker, Ontology, OpenAIClient, Document
import glob
import os
from os import path
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='医学知识图谱构建工具 - 从医学文献中自动提取实体和关系',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用
  python main.py -i ./input_docs -o ./output_kg
  
  # 指定模型和 API
  python main.py -i ./docs -o ./kg --model qwen3 --base-url http://localhost:8000/v1
  
  # 调整文档切分参数
  python main.py -i ./docs -o ./kg --chunk-size 4096 --stride 4000
  
  # 使用自定义实体类型（通过配置文件）
  python main.py -i ./docs -o ./kg --ontology-config config.json
        """
    )

    # ========== 必需参数 ==========
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='输入文档目录 (默认：当前目录)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='输出结果目录 (默认：当前目录)'
    )

    # ========== LLM 配置参数 ==========
    llm_group = parser.add_argument_group('LLM 配置')
    llm_group.add_argument(
        '--model',
        type=str,
        default='qwen3',
        help='LLM 模型名称 (默认：qwen3)'
    )

    llm_group.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:8000/v1',
        help='LLM API 基础 URL (默认：http://localhost:8000/v1)'
    )
    
    llm_group.add_argument(
        '--api-key',
        type=str,
        default='ollama',
        help='LLM API 密钥 (默认：ollama，本地部署通常不需要真实 key)'
    )

    llm_group.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='生成温度，控制随机性 (默认：0.1)'
    )

    llm_group.add_argument(
        '--top-p',
        type=float,
        default=0.5,
        help='Top-p 采样参数 (默认：0.5)'
    )

    # ========== 文档处理参数 ==========
    doc_group = parser.add_argument_group('文档处理')
    doc_group.add_argument(
        '--chunk-size',
        type=int,
        default=2048,
        help='文档切分窗口大小（字符数）(默认：2048)'
    )

    doc_group.add_argument(
        '--stride',
        type=int,
        default=2000,
        help='滑动窗口步长（字符数）(默认：2000)'
    )

    doc_group.add_argument(
        '--max-length',
        type=int,
        default=8000,
        help='单个文档最大处理长度（字符数）(默认：8000)'
    )

    doc_group.add_argument(
        '--file-pattern',
        type=str,
        default='*.md',
        help='输入文件匹配模式 (默认：*.md)'
    )

    # ========== 图谱构建参数 ==========
    graph_group = parser.add_argument_group('图谱构建')
    graph_group.add_argument(
        '--delay-s',
        type=int,
        default=10,
        help='API 请求间隔时间（秒）(默认：10)'
    )

    graph_group.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )

    # ========== 其他参数 ==========
    parser.add_argument(
        '--ontology-config',
        type=str,
        default=None,
        help='本体配置文件路径（JSON 格式，不指定则使用内置医学本体）'
    )
    
    parser.add_argument(
        '--auto-ontology',
        action='store_true',
        help='使用 LLM 自动生成领域本体（会先分析文档并生成本体定义）'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        default='general',
        help='领域类型（用于自动生成本体）: general, medical, legal, financial, academic 等 (默认：general)'
    )
    
    parser.add_argument(
        '--num-entity-types',
        type=int,
        default=10,
        help='自动生成本体时的实体类型数量（默认：10，范围 5-20）'
    )

    return parser.parse_args()


def load_ontology_from_file(config_path: str) -> Ontology:
    """从 JSON 配置文件加载本体"""
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    labels = config.get('labels', [])
    relationships = config.get('relationships', ['Relation between any pair of Entities'])
    
    print(f"✅ 已加载自定义本体：{len(labels)} 种实体类型，{len(relationships)} 种关系")
    return Ontology(labels=labels, relationships=relationships)


def generate_ontology_with_llm(llm_client, domain: str = 'general', num_entity_types: int = 10) -> Ontology:
    """使用 LLM 自动生成领域特定的本体"""
    import json
    
    print(f"\n🤖 正在使用 LLM 生成 {domain} 领域的本体...")
    
    # 构建提示词
    prompt = f"""You are an expert in knowledge graph ontology design. I need to build a knowledge graph for the **{domain}** domain.

Please design a structured ontology with the following requirements:

1. Define {num_entity_types} most important entity types in this domain
2. For each entity type, provide:
   - A concise name (single word or short phrase, CamelCase)
   - A clear description in Chinese (what it includes, examples)
3. Suggest 5-10 important relationship types between entities

Format your response as a valid JSON object with this structure:
```json
{{
  "labels": [
    {{"EntityName1": "Description in Chinese with examples"}},
    {{"EntityName2": "Description in Chinese with examples"}}
  ],
  "relationships": [
    "relationship_type_1 (Chinese description)",
    "relationship_type_2 (Chinese description)"
  ]
}}
```

Make sure the ontology covers the core concepts and relationships in the {domain} domain."""

    try:
        # 调用 LLM 生成
        response = llm_client.generate(prompt)
        
        # 提取 JSON 部分
        import re
        json_match = re.search(r'```json\s*(.+?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_str = response.strip()
        
        # 解析 JSON
        ontology_data = json.loads(json_str)
        
        labels = ontology_data.get('labels', [])
        relationships = ontology_data.get('relationships', ['Relation between any pair of Entities'])
        
        print(f"✅ LLM 生成本体完成：{len(labels)} 种实体类型，{len(relationships)} 种关系")
        print("\n生成的实体类型:")
        for i, label in enumerate(labels[:5], 1):
            entity_name = list(label.keys())[0]
            print(f"  {i}. {entity_name}")
        if len(labels) > 5:
            print(f"  ... 还有 {len(labels)-5} 种")
        
        return Ontology(labels=labels, relationships=relationships)
        
    except Exception as e:
        print(f"⚠️  LLM 生成本体失败：{e}")
        print("将使用默认通用本体")
        return create_default_ontology()


def create_default_ontology() -> Ontology:
    """创建默认通用领域本体（简化的通用版本）"""
    return Ontology(
        labels=[
            {"Entity": "主要实体或对象"},
            {"Person": "人物、个体"},
            {"Organization": "组织、机构、公司"},
            {"Location": "地点、位置"},
            {"Event": "事件、活动"},
            {"Concept": "概念、想法、理论"},
            {"Product": "产品、物品"},
            {"Time": "时间、日期"},
        ],
        relationships=[
            "Relation between any pair of Entities",
        ],
    )


def create_medical_ontology():
    """创建医学领域本体"""
    return Ontology(
        labels=[
            {"Disease": "疾病名称，包括急性病、慢性病、综合征、传染病等病理状态，如糖尿病、高血压、新冠肺炎"},
            {"Symptom": "临床症状和体征，包括患者主诉和医生观察到的异常表现，如发热、咳嗽、头痛、呼吸困难"},
            {"Drug": "药物名称，包括化学药、生物制品、中成药等，不含剂量和剂型，如阿司匹林、胰岛素、青霉素"},
            {"Treatment": "治疗方法，包括手术治疗、物理治疗、心理治疗等非药物干预措施，如心脏搭桥术、放疗、认知行为疗法"},
            {"Anatomy": "人体解剖结构，包括器官、组织、系统、部位等，如心脏、肝脏、呼吸系统、左上肢"},
            {"Test": "医学检查检验项目，包括实验室检查、影像学检查、功能检查等，如血常规、CT 扫描、心电图"},
            {"Pathogen": "病原体，包括细菌、病毒、真菌、寄生虫等致病微生物，如 SARS-CoV-2、大肠杆菌、疟原虫"},
            {"Gene": "基因和遗传相关分子，包括基因名称、蛋白质、RNA 等，如 BRCA1、TP53、mRNA"},
            {"Cell": "细胞类型和细胞成分，包括免疫细胞、干细胞、细胞器等，如 T 细胞、红细胞、线粒体"},
            {"Biomarker": "生物标志物，用于诊断、预后或疗效评估的生物学指标，如 PSA、HbA1c、肿瘤标志物"},
            {"SideEffect": "药物或治疗的不良反应和副作用，如恶心、皮疹、肝功能异常、骨髓抑制"},
            {"Contraindication": "禁忌症，指不适合使用某种药物或治疗的情况，如妊娠、严重肾功能不全"},
            {"Indication": "适应症，药物或治疗的适用病症，如用于治疗晚期肺癌、控制高血压"},
            {"Dosage": "药物剂量和用法，包括给药途径、频率、疗程，如口服每日一次、静脉注射 5mg/kg"},
            {"Patient": "患者人口学特征，包括年龄、性别、职业等基本信息，如 65 岁男性、孕妇、儿童"},
            {"MedicalDevice": "医疗器械和设备，如心脏起搏器、人工关节、呼吸机、血糖仪"},
            {"Department": "医学科室和专科，如心内科、神经外科、儿科、急诊科"},
            {"Hospital": "医院和医疗机构名称，如北京协和医院、梅奥诊所"},
            {"Guideline": "临床指南和共识，如 NCCN 指南、ESC 指南、专家共识"},
            {"Research": "医学研究类型，如随机对照试验、队列研究、Meta 分析"},
        ],
        relationships=[
            "Relation between any pair of Entities",
        ],
    )


def main():
    # 模拟命令行参数
    sys.argv = [
        'main.py',
        '-i', './',
        '-o', './kg',
    ]

    # 然后正常调用 parse_args()
    args = parse_args()
    
    # 打印配置信息
    print("=" * 60)
    print("知识图谱构建工具")
    print("=" * 60)
    print(f"输入目录：{args.input_dir}")
    print(f"输出目录：{args.output_dir}")
    print(f"模型：{args.model} @ {args.base_url}")
    print(f"领域：{args.domain}")
    print(f"本体生成方式：", end="")
    
    if args.auto_ontology:
        print("LLM 自动生成")
    elif args.ontology_config:
        print(f"配置文件 ({args.ontology_config})")
    else:
        print("内置默认本体")
    
    print(f"文档切分：窗口={args.chunk_size}, 步长={args.stride}")
    print(f"最大文档长度：{args.max_length}")
    print(f"文件模式：{args.file_pattern}")
    print(f"API 延迟：{args.delay_s}s")
    print(f"详细输出：{'是' if args.verbose else '否'}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
        
    # 2. 初始化 LLM 客户端
    print(f"\n初始化 LLM 客户端...")
    llm = OpenAIClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p
    )
        
    # 1. 创建本体（在 LLM 初始化之后）
    if args.auto_ontology:
        # 使用 LLM 自动生成领域本体
        ontology = generate_ontology_with_llm(
            llm,
            domain=args.domain,
            num_entity_types=args.num_entity_types
        )
    elif args.ontology_config:
        # 从配置文件加载本体
        ontology = load_ontology_from_file(args.ontology_config)
    else:
        # 根据领域选择默认本体
        if args.domain == 'medical':
            print("使用内置医学本体")
            ontology = create_medical_ontology()
        else:
            print("使用内置通用本体")
            ontology = create_default_ontology()

    # 3. 初始化图谱构建器
    graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=args.verbose)

    # 4. 加载文档
    print(f"\n加载文档...")
    input_pattern = path.join(args.input_dir, args.file_pattern)
    file_list = glob.glob(input_pattern)

    if not file_list:
        print(f"❌ 错误：在 {input_pattern} 未找到任何文件")
        return

    print(f"找到 {len(file_list)} 个文件")

    # 5. 切分文档
    print(f"\n切分文档...")
    chunks = []
    total_chars = 0

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            document = f.read()[:args.max_length]

        doc_length = len(document)
        total_chars += doc_length

        # 使用滑动窗口切分文档
        start = 0
        file_chunks = []
        while start < doc_length:
            end = start + args.chunk_size
            chunk = document[start:end]
            if chunk:  # 只添加非空 chunk
                file_chunks.append(chunk)
            start += args.stride

        chunks.extend(file_chunks)
        print(f"  {path.basename(file)}: {doc_length} chars → {len(file_chunks)} chunks")

    print(f"\n总计：{total_chars} 字符 → {len(chunks)} 个片段")

    # 6. 创建文档对象
    docs = [
        Document(
            text=chunk,
            metadata={
                "text": chunk,
                "chunk_size": args.chunk_size,
                "stride": args.stride
            }
        )
        for chunk in chunks
    ]

    # 7. 构建知识图谱
    print(f"\n构建知识图谱...")
    graph = graph_maker.from_documents(
        list(docs),
        delay_s_between=args.delay_s
    )

    # 8. 输出结果
    print(f"\n✅ 完成！共提取 {len(graph)} 条关系边")

    output_file = path.join(args.output_dir, "graph.jsonl")
    if not path.exists(output_file):
        print(f"⚠️  警告：{output_file} 不存在，将创建一个空文件")
        os.makedirs(path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for edge in graph:
            f.write(edge.json() + "\n")

    print(f"结果已保存到：{output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

