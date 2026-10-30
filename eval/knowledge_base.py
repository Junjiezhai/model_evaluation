# -*- coding: utf-8 -*-
"""
信贷知识库加载与简易检索器。

解析 base_knowledge/credit_standard_QA_intent.md 中的标准问答对，
并提供基于中文二元语法（bigram）重叠度的 top-k 检索，
用于在评测中模拟 RAG 链路的 retrieval_context。
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import config


@dataclass
class KnowledgePair:
    """一条标准问答对。"""
    category: str          # 一级分类（如：提额降息）
    sub_category: str      # 二级分类（如：操作咨询）
    question: str
    answer: str

    def as_context(self) -> str:
        """格式化为检索上下文。"""
        return f"【{self.category}·{self.sub_category}】\n问：{self.question}\n答：{self.answer}"


@dataclass
class KnowledgeBase:
    pairs: List[KnowledgePair] = field(default_factory=list)

    @classmethod
    def from_markdown(cls, path: str = None) -> 'KnowledgeBase':
        """从 markdown 文件解析全部问答对。

        文件格式约定：
          ## 一级分类
          ### 二级分类
          **问**：xxx
          **答**：yyy
        """
        path = path or config.KNOWLEDGE_BASE_PATH
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        kb = cls()
        category, sub_category = '未分类', '未分类'
        # 逐行解析，维护当前分类上下文
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            h2 = re.match(r'^##\s+(.+)$', line)
            h3 = re.match(r'^###\s+(.+)$', line)
            if h2:
                category = h2.group(1).strip()
                sub_category = '未分类'
                i += 1
                continue
            if h3:
                sub_category = h3.group(1).strip()
                i += 1
                continue
            q_match = re.match(r'^\*\*问\*\*[：:]\s*(.+)$', line)
            if q_match:
                question = q_match.group(1).strip()
                # 向下寻找对应的答
                answer = ''
                j = i + 1
                while j < len(lines):
                    a_match = re.match(r'^\*\*答\*\*[：:]\s*(.+)$', lines[j].strip())
                    if a_match:
                        answer = a_match.group(1).strip()
                        break
                    j += 1
                if question and answer:
                    kb.pairs.append(KnowledgePair(category, sub_category, question, answer))
                i = j + 1
                continue
            i += 1
        return kb

    # ---------------- 简易检索 ----------------
    @staticmethod
    def _tokens(text: str) -> List[str]:
        """中文按二元语法切分，英文/数字按单词切分。"""
        text = re.sub(r'\s+', '', text.lower())
        tokens = re.findall(r'[a-z0-9]+', text)
        tokens += [text[k:k + 2] for k in range(len(text) - 1)
                   if re.match(r'[一-鿿]', text[k])]
        return tokens

    def retrieve(self, query: str, top_k: int = None) -> List[KnowledgePair]:
        """基于 token 重叠度返回 top-k 相关问答对。"""
        top_k = top_k or config.RETRIEVAL_TOP_K
        query_tokens = set(self._tokens(query))
        if not query_tokens:
            return []
        scored: List[Tuple[float, KnowledgePair]] = []
        for pair in self.pairs:
            doc_tokens = set(self._tokens(pair.question + pair.answer))
            overlap = len(query_tokens & doc_tokens)
            score = overlap / (len(query_tokens) + 1e-6)
            if score > 0:
                scored.append((score, pair))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in scored[:top_k]]

    def categories(self) -> List[str]:
        return sorted({p.category for p in self.pairs})


if __name__ == '__main__':
    kb = KnowledgeBase.from_markdown()
    print(f'共解析 {len(kb.pairs)} 条问答对，分类: {kb.categories()}')
    demo = kb.retrieve('请问个税认证怎么操作？')
    for p in demo:
        print('-' * 40)
        print(p.as_context())
