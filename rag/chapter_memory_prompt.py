from __future__ import annotations

import json
from typing import Any


CHAPTER_MEMORY_EXTRACTION_CONFIG: dict[str, Any] = {
    "name": "章节记忆提取",
    "agent": (
        "你是一名资深小说记忆分析师，擅长从已经完成的章节中识别可供后续创作"
        "使用的原子事实，并准确区分永久设定与阶段性剧情。\n"
        "你需要为每条记忆评估重要度。raw_importance必须是0.0到1.0之间的数字，"
        "禁止使用1到10的十分制，也禁止输出百分数或带评分说明的字符串。\n"
        "0.0-0.2表示装饰性描写、普通动作或可忽略信息；"
        "0.3-0.4表示局部细节、短期状态或轻微关系变化；"
        "0.5-0.6表示会影响近期章节的一般事件、状态或线索；"
        "0.7-0.8表示关键事件、明显状态变化、重要关系或重要伏笔；"
        "0.9-1.0表示主线转折、重大不可逆事件、核心世界规则或永久身份。\n"
        "重要度高不等于永久记忆。只有章节明确确认的永久身份、固定能力、"
        "不可改变的世界规则等稳定事实，才允许canon_candidate=true并设置"
        "memory_scope=permanent；普通事件、状态、关系和伏笔必须保持temporary。"
    ),
    "rule": (
        "只能依据输入中当前章节的标题和完整正文提取事实，不得参考旧版本、"
        "不得调用数据库信息、不得补充常识推断，也不得编造正文未明确表达的内容。\n"
        "每条fact只表达一个可以独立理解的事实；不同人物、物品、事件或状态变化"
        "应拆分为不同fact，相同事实不得重复输出。content必须包含明确主体和事实，"
        "不能使用‘他发生了变化’之类脱离上下文后无法理解的表述。\n"
        "允许的fact_type为event、state_change、character_state、relation、"
        "foreshadowing_open、foreshadowing_resolved、world_rule。"
        "event记录已经发生的事件；state_change记录对象前后状态变化；"
        "character_state记录人物当前状态；relation记录人物或势力关系；"
        "foreshadowing_open记录尚未解决的线索；foreshadowing_resolved记录本章"
        "已经回收的伏笔；world_rule只记录正文明确确认的世界规则。\n"
        "尽量填写character_names、item_names、event_names；涉及明确对象字段变化时"
        "填写entity_name、field、old_value和new_value。不要提取纯文风、措辞、"
        "修辞、无后续作用的环境描写和重复性的普通动作。"
    ),
    "format": {
        "facts": [
            {
                "fact_type": "event",
                "content": "陈玥在青铜门前交出了月族钥匙",
                "character_names": ["陈玥"],
                "item_names": ["月族钥匙"],
                "event_names": ["交出钥匙"],
                "raw_importance": 0.8,
                "canon_candidate": False,
                "memory_scope": "temporary",
                "entity_name": None,
                "field": None,
                "old_value": None,
                "new_value": None,
                "hook_status": None,
            }
        ]
    },
}


def build_memory_task_prompt(config: dict[str, Any]) -> str:
    return (
        f"任务名称：{config['name']}\n"
        f"【角色】\n{config['agent']}\n"
        f"【规则】\n{config['rule']}\n"
        "【输出格式】\n"
        "只输出符合以下结构的业务结果；字段名、字段类型和层级必须严格一致。"
        "没有可填写内容的数组使用[]，可空字段使用null。\n"
        f"{json.dumps(config['format'], ensure_ascii=False, separators=(',', ':'))}"
    )
