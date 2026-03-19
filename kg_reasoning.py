import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
import pickle
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY_API_KEY",
)

model_id = client.models.list().data[0].id
print(model_id)


class KnowledgeGraph:
    """知识图谱推理引擎"""

    def __init__(self):
        # 有向图
        self.graph = nx.DiGraph()
        # 存储关系类型
        self.relation_types: Set[str] = set()
        # 实体->关系->实体 的索引
        self.triple_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add_triple(self, head: str, relation: str, tail: str):
        """添加三元组"""
        # 添加边（带关系属性）
        self.graph.add_edge(head, tail, relation=relation)
        self.relation_types.add(relation)
        self.triple_index[(head, relation)].add(tail)

    def load_triples(self, triples: List[Tuple[str, str, str]]):
        """批量加载三元组"""
        for head, relation, tail in triples:
            self.add_triple(head, relation, tail)
        print(f"✅ 已加载 {len(triples)} 条三元组")
        print(f"   实体数量：{self.graph.number_of_nodes()}")
        print(f"   关系类型：{len(self.relation_types)}")

    # ========== 基本查询 ==========

    def get_neighbors(self, entity: str, relation: Optional[str] = None) -> Set[str]:
        """获取实体的邻居（可选指定关系）"""
        if entity not in self.graph:
            return set()

        neighbors = set()
        for successor in self.graph.successors(entity):
            edge_data = self.graph.edges[entity, successor]
            if relation is None or edge_data.get('relation') == relation:
                neighbors.add(successor)
        return neighbors

    def get_predecessors(self, entity: str, relation: Optional[str] = None) -> Set[str]:
        """获取前驱实体"""
        if entity not in self.graph:
            return set()

        predecessors = set()
        for predecessor in self.graph.predecessors(entity):
            edge_data = self.graph.edges[predecessor, entity]
            if relation is None or edge_data.get('relation') == relation:
                predecessors.add(predecessor)
        return predecessors

    def find_relation(self, head: str, tail: str) -> List[str]:
        """查找两个实体之间的关系"""
        if self.graph.has_edge(head, tail):
            return [self.graph.edges[head, tail].get('relation')]
        # 也检查反向边
        relations = []
        if self.graph.has_edge(tail, head):
            relations.append(self.graph.edges[tail, head].get('relation'))
        return relations

    # ========== 路径推理 ==========

    def find_paths(self, start: str, end: str, max_length: int = 3) -> List[List[Tuple[str, str, str]]]:
        """查找两个实体之间的所有路径"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, start, end, cutoff=max_length
            ))

            # 转换为三元组形式
            result = []
            for path in paths:
                triples = []
                for i in range(len(path) - 1):
                    head, tail = path[i], path[i+1]
                    relation = self.graph.edges[head, tail].get('relation')
                    triples.append((head, relation, tail))
                result.append(triples)
            return result
        except nx.NetworkXNoPath:
            return []

    def find_shortest_path(self, start: str, end: str) -> Optional[List[Tuple[str, str, str]]]:
        """最短路径推理"""
        try:
            path = nx.shortest_path(self.graph, start, end)
            triples = []
            for i in range(len(path) - 1):
                head, tail = path[i], path[i+1]
                relation = self.graph.edges[head, tail].get('relation')
                triples.append((head, relation, tail))
            return triples
        except nx.NetworkXNoPath:
            return None

    def _get_entity_all_relations(self, entity: str) -> List[Tuple[str, str, str]]:
        """获取实体的所有关系三元组"""
        if entity not in self.graph:
            return []

        triples = []
        # 出边关系
        for successor in self.graph.successors(entity):
            edge_data = self.graph.edges[entity, successor]
            relation = edge_data.get('relation')
            triples.append((entity, relation, successor))

        # 入边关系
        for predecessor in self.graph.predecessors(entity):
            edge_data = self.graph.edges[predecessor, entity]
            relation = edge_data.get('relation')
            triples.append((predecessor, relation, entity))

        return triples

    def _llm_select_top_paths(self, current_entity: str, target_entity: str,
                             candidate_triples: List[Tuple[str, str, str]],
                             top_k: int = 1) -> List[Tuple[str, str, str]]:
        """使用 LLM 选择最有价值的 top-k 个路径"""
        if not candidate_triples:
            return []

        # 构建提示词
        prompt = f"""你是一个知识图谱推理专家。当前在实体"{current_entity}"，目标是找到与"{target_entity}"相关的路径。

以下是从"{current_entity}"出发的所有可能关系（三元组形式）：
"""
        for i, (h, r, t) in enumerate(candidate_triples, 1):
            prompt += f"{i}. ({h}, {r}, {t})\n"

        prompt += f"""
请分析这些关系，选择最有价值继续探索以到达目标实体"{target_entity}"的 top-{top_k} 个关系（考虑语义相关性、路径潜力等因素）。

只需返回选择的三元组编号（用逗号分隔，如：1,3,5），不要解释原因。
如果没有合适的路径，返回"无"。
"""

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "你是一个知识图谱推理专家，擅长发现实体间的语义关联。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            answer = response.choices[0].message.content.strip()

            if answer == "无" or not answer:
                return []

            # 解析选择的编号
            selected_indices = []
            for part in answer.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(candidate_triples):
                        selected_indices.append(idx)

            # 返回选中的三元组
            selected_triples = [candidate_triples[i] for i in selected_indices]
            return selected_triples

        except Exception as e:
            print(f"LLM 调用失败：{e}")
            # 降级策略：返回所有候选
            return candidate_triples[:top_k]

    def find_llm_guided_path(self, start: str, end: str, max_length: int = 5,
                            top_k: int = 1) -> Optional[List[Tuple[str, str, str]]]:
        """
        LLM 引导的路径搜索

        在每个节点处：
        1. 汇总该节点的所有关系三元组
        2. 让 LLM 选择最有价值的 top-k 个路径继续探索
        3. 递归搜索直到达到目标或最大长度

        Args:
            start: 起始实体
            end: 目标实体
            max_length: 最大路径长度
            top_k: 每个节点选择的路径数

        Returns:
            路径三元组列表，如果找不到则返回 None
        """
        if start == end:
            return []

        if start not in self.graph or end not in self.graph:
            return None

        # BFS + LLM 引导的搜索
        from collections import deque

        # 队列元素：(当前实体，当前路径三元组列表，已访问集合)
        queue = deque([(start, [], {start})])

        while queue:
            current_entity, current_path, visited = queue.popleft()

            # 检查是否达到目标
            if current_entity == end:
                return current_path

            # 检查是否超过最大长度
            if len(current_path) >= max_length:
                continue

            # 获取当前实体的所有关系三元组
            all_triples = self._get_entity_all_relations(current_entity)

            # 过滤掉已访问的实体
            unvisited_triples = []
            for h, r, t in all_triples:
                next_entity = t if h == current_entity else h
                if next_entity not in visited:
                    unvisited_triples.append((h, r, t))

            if not unvisited_triples:
                continue

            # 使用 LLM 选择 top-k 个最有价值的路径
            selected_triples = self._llm_select_top_paths(
                current_entity, end, unvisited_triples, top_k
            )

            # 将选中的路径加入队列
            for triple in selected_triples:
                h, r, t = triple
                next_entity = t if h == current_entity else h

                new_path = current_path + [triple]
                new_visited = visited | {next_entity}

                queue.append((next_entity, new_path, new_visited))

        # 没有找到路径
        return None

    # ========== 链路预测 ==========

    def common_neighbors(self, entity1: str, entity2: str) -> Set[str]:
        """共同邻居"""
        neighbors1 = self.get_neighbors(entity1)
        neighbors2 = self.get_neighbors(entity2)
        return neighbors1 & neighbors2

    def adamic_adar(self, entity1: str, entity2: str) -> float:
        """Adamic-Adar 链路预测指标"""
        common = self.common_neighbors(entity1, entity2)
        if not common:
            return 0.0

        score = 0.0
        for common_entity in common:
            degree = self.graph.degree(common_entity)
            if degree > 1:
                score += 1 / (degree - 1)
        return score

    def predict_link(self, entity1: str, entity2: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """预测两个实体之间可能的关系（基于共同邻居）"""
        common = self.common_neighbors(entity1, entity2)
        if not common:
            return []

        # 统计共同邻居与目标实体的关系
        relation_scores = defaultdict(float)

        for neighbor in common:
            # neighbor -> entity1 的关系
            for succ in self.graph.successors(neighbor):
                edge_data = self.graph.edges[neighbor, succ]
                relation = edge_data.get('relation')
                if succ == entity2:
                    relation_scores[relation] += 1

            # neighbor -> entity2 的关系
            for succ in self.graph.successors(neighbor):
                edge_data = self.graph.edges[neighbor, succ]
                relation = edge_data.get('relation')
                if succ == entity1:
                    relation_scores[relation] += 1

        # 排序返回
        sorted_relations = sorted(
            relation_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_relations[:top_k]

    # ========== 关系推理 ==========

    def one_hop_infer(self, head: str, relation: str) -> Set[str]:
        """单跳推理：给定头实体和关系，预测尾实体"""
        return self.triple_index.get((head, relation), set())

    def inverse_relation_infer(self, tail: str, relation: str) -> Set[str]:
        """逆向推理：给定尾实体和关系，找头实体"""
        # 查找存在 relation 从 tail 指出的情况
        # 实际上是找 (head, relation, tail) 中的 head
        result = set()
        for head in self.get_predecessors(tail, relation):
            edge_data = self.graph.edges[head, tail]
            if edge_data.get('relation') == relation:
                result.add(head)
        return result

    def relation_chain_infer(self, head: str, relations: List[str]) -> Set[str]:
        """关系链推理：多跳关系推理"""
        current_entities = {head}

        for relation in relations:
            next_entities = set()
            for entity in current_entities:
                next_entities |= self.triple_index.get((entity, relation), set())
            current_entities = next_entities
            if not current_entities:
                break

        return current_entities

    # ========== 图分析 ==========

    def get_entity_info(self, entity: str) -> Dict:
        """获取实体信息"""
        if entity not in self.graph:
            return {}

        return {
            '出度': self.graph.out_degree(entity),
            '入度': self.graph.in_degree(entity),
            '出边关系': [
                self.graph.edges[entity, succ].get('relation')
                for succ in self.graph.successors(entity)
            ],
            '入边关系': [
                self.graph.edges[pred, entity].get('relation')
                for pred in self.graph.predecessors(entity)
            ]
        }

    def save(self, filepath: str):
        """保存图到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'relation_types': self.relation_types,
                'triple_index': dict(self.triple_index)
            }, f)
        print(f"✅ 已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """从文件加载图"""
        kg = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            kg.graph = data['graph']
            kg.relation_types = data['relation_types']
            kg.triple_index = defaultdict(set, data['triple_index'])
        print(f"✅ 已从 {filepath} 加载")
        return kg


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例三元组
    triples = [
        # === 教育背景 ===
        ("张三", "毕业于", "清华大学"),
        ("张三", "获得学位", "博士"),
        ("清华大学", "位于", "北京"),
        ("清华大学", "成立于", "1911 年"),
        ("清华大学", "类型", "985 高校"),
        ("清华大学", "校长", "邱勇"),
        ("邱勇", "毕业于", "清华大学"),
        ("邱勇", "获得学位", "博士"),

        # === 工作经历 ===
        ("张三", "就职于", "阿里巴巴"),
        ("张三", "职位", "高级工程师"),
        ("阿里巴巴", "位于", "杭州"),
        ("阿里巴巴", "成立于", "1999 年"),
        ("阿里巴巴", "创始人", "马云"),
        ("阿里巴巴", "CEO", "张勇"),
        ("阿里巴巴", "类型", "互联网公司"),
        ("阿里巴巴", "子公司", "阿里云"),
        ("阿里云", "位于", "杭州"),
        ("张勇", "毕业于", "上海财经大学"),
        ("张勇", "加入阿里", "2007 年"),

        # === 人物关系 ===
        ("马云", "出生于", "杭州"),
        ("马云", "毕业于", "杭州师范大学"),
        ("马云", "国籍", "中国"),
        ("马云", "职位", "企业家"),
        ("杭州师范大学", "位于", "杭州"),
        ("杭州师范大学", "类型", "师范院校"),

        # === 地域关系 ===
        ("北京", "是首都", "中国"),
        ("北京", "人口", "2100 万"),
        ("杭州", "位于", "浙江省"),
        ("杭州", "著名景点", "西湖"),
        ("浙江省", "省会", "杭州"),
        ("中国", "首都", "北京"),

        # === 社会关系 ===
        ("张三", "同事", "李四"),
        ("李四", "就职于", "阿里巴巴"),
        ("李四", "毕业于", "北京大学"),
        ("李四", "职位", "产品经理"),
        ("北京大学", "位于", "北京"),
        ("北京大学", "类型", "985 高校"),
        ("北京大学", "竞争对手", "清华大学"),

        # === 行业关系 ===
        ("阿里巴巴", "竞争对手", "腾讯"),
        ("腾讯", "位于", "深圳"),
        ("腾讯", "创始人", "马化腾"),
        ("腾讯", "产品", "微信"),
        ("腾讯", "产品", "QQ"),
        ("深圳", "位于", "广东省"),
        ("深圳", "类型", "经济特区"),
        ("马化腾", "毕业于", "深圳大学"),
        ("深圳大学", "位于", "深圳"),

        # === 投资关系 ===
        ("阿里巴巴", "投资", "菜鸟网络"),
        ("阿里巴巴", "投资", "蚂蚁金服"),
        ("蚂蚁金服", "CEO", "井贤栋"),
        ("蚂蚁金服", "位于", "杭州"),
        ("腾讯", "投资", "京东"),
        ("京东", "CEO", "徐雷"),
        ("京东", "位于", "北京"),

        # === 合作关系 ===
        ("清华大学", "合作", "麻省理工学院"),
        ("麻省理工学院", "位于", "美国"),
        ("美国", "首都", "华盛顿"),
        ("北京大学", "合作", "斯坦福大学"),
        ("斯坦福大学", "位于", "美国"),
        ("斯坦福大学", "知名校友", "埃隆·马斯克"),
        ("埃隆·马斯克", "创立", "SpaceX"),
        ("埃隆·马斯克", "创立", "特斯拉"),
        ("特斯拉", "位于", "美国"),
        ("SpaceX", "位于", "美国"),
    ]

    # 构建知识图谱
    kg = KnowledgeGraph()
    kg.load_triples(triples)

    print("\n" + "="*50)
    print("【基本查询】")
    print("="*50)

    # 查询张三的邻居
    print(f"\n张三的邻居：{kg.get_neighbors('张三')}")
    # 查询清华大学的所有关系
    print(f"清华大学的信息：{kg.get_entity_info('清华大学')}")
    # 新增：查询阿里巴巴的信息
    print(f"阿里巴巴的信息：{kg.get_entity_info('阿里巴巴')}")

    print("\n" + "="*50)
    print("【路径推理】")
    print("="*50)

    # 路径查询
    paths = kg.find_paths("张三", "杭州", max_length=3)
    print(f"\n张三到杭州的路径：{paths}")

    # 最短路径
    shortest = kg.find_shortest_path("张三", "邱勇")
    print(f"张三到邱勇的最短路径：{shortest}")

    # 新增路径查询
    shortest2 = kg.find_shortest_path("马云", "马化腾")
    print(f"马云到马化腾的最短路径：{shortest2}")

    print("\n" + "="*50)
    print("【LLM 引导路径搜索】")
    print("="*50)

    # LLM 引导的路径搜索
    llm_path = kg.find_llm_guided_path("张三", "杭州", max_length=5, top_k=2)
    print(f"\nLLM 引导路径（张三->杭州）: {llm_path}")

    llm_path2 = kg.find_llm_guided_path("马云", "清华大学", max_length=5, top_k=2)
    print(f"LLM 引导路径（马云->清华大学）: {llm_path2}")

    # 新增 LLM 路径查询
    llm_path3 = kg.find_llm_guided_path("埃隆·马斯克", "中国", max_length=6, top_k=2)
    print(f"LLM 引导路径（埃隆·马斯克->中国）: {llm_path3}")

    llm_path4 = kg.find_llm_guided_path("李四", "腾讯", max_length=5, top_k=2)
    print(f"LLM 引导路径（李四->腾讯）: {llm_path4}")

    print("\n" + "="*50)
    print("【链路预测】")
    print("="*50)

    # 共同邻居
    common = kg.common_neighbors("清华大学", "阿里巴巴")
    print(f"\n清华大学和阿里巴巴的共同邻居：{common}")

    # Adamic-Adar
    aa_score = kg.adamic_adar("清华大学", "阿里巴巴")
    print(f"AA score: {aa_score}")

    # 新增链路预测
    common2 = kg.common_neighbors("阿里巴巴", "腾讯")
    print(f"\n阿里巴巴和腾讯的共同邻居：{common2}")

    aa_score2 = kg.adamic_adar("阿里巴巴", "腾讯")
    print(f"AA score (阿里 - 腾讯): {aa_score2}")

    print("\n" + "="*50)
    print("【关系推理】")
    print("="*50)

    # 单跳推理
    result = kg.one_hop_infer("张三", "毕业于")
    print(f"\n张三毕业于：{result}")

    # 关系链推理
    chain_result = kg.relation_chain_infer("张三", ["毕业于", "校长"])
    print(f"张三 -> 毕业于 -> 校长：{chain_result}")

    # 新增关系推理
    chain_result2 = kg.relation_chain_infer("马云", ["毕业于", "位于"])
    print(f"马云 -> 毕业于 -> 位于：{chain_result2}")

    chain_result3 = kg.relation_chain_infer("埃隆·马斯克", ["创立", "位于"])
    print(f"埃隆·马斯克 -> 创立 -> 位于：{chain_result3}")

    # 构建知识图谱
    kg = KnowledgeGraph()
    kg.load_triples(triples)

    print("\n" + "="*50)
    print("【基本查询】")
    print("="*50)

    # 查询张三的邻居
    print(f"\n张三的邻居：{kg.get_neighbors('张三')}")
    # 查询清华大学的所有关系
    print(f"清华大学的信息：{kg.get_entity_info('清华大学')}")

    print("\n" + "="*50)
    print("【路径推理】")
    print("="*50)

    # 路径查询
    paths = kg.find_paths("张三", "杭州", max_length=3)
    print(f"\n张三到杭州的路径：{paths}")

    # 最短路径
    shortest = kg.find_shortest_path("张三", "邱勇")
    print(f"张三到邱勇的最短路径：{shortest}")

    print("\n" + "="*50)
    print("【LLM 引导路径搜索】")
    print("="*50)

    # LLM 引导的路径搜索
    llm_path = kg.find_llm_guided_path("张三", "杭州", max_length=5, top_k=2)
    print(f"\nLLM 引导路径（张三->杭州）: {llm_path}")

    llm_path2 = kg.find_llm_guided_path("马云", "清华大学", max_length=5, top_k=2)
    print(f"LLM 引导路径（马云->清华大学）: {llm_path2}")

    print("\n" + "="*50)
    print("【链路预测】")
    print("="*50)

    # 共同邻居
    common = kg.common_neighbors("清华大学", "阿里巴巴")
    print(f"\n清华大学和阿里巴巴的共同邻居：{common}")

    # Adamic-Adar
    aa_score = kg.adamic_adar("清华大学", "阿里巴巴")
    print(f"AA score: {aa_score}")

    print("\n" + "="*50)
    print("【关系推理】")
    print("="*50)

    # 单跳推理
    result = kg.one_hop_infer("张三", "毕业于")
    print(f"\n张三毕业于：{result}")

    # 关系链推理
    chain_result = kg.relation_chain_infer("张三", ["毕业于", "校长"])
    print(f"张三 -> 毕业于 -> 校长：{chain_result}")
