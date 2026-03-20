"""
智能 AI 产品资讯雷达（MVP）- 最终阶段：版面微调 + 可视化辅助分析

业务逻辑：
- SQLite 持久化：RSS 增量入库（link 去重）、未处理行由 LLM 打分回写；按时间范围回溯已分析资讯。
- 版面与可视化：紧凑卡片与操作区弱化、布局切换（经典分栏 / 摘要置顶）、数据趋势看板（赛道与类型分布）。
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from time import mktime

# LLM 与侧边栏共用枚举（须与 analyze_news_with_llm 内 Prompt 严格一致）
LLM_PRODUCT_TYPES: list[str] = [
    "基础大模型",
    "AI Agent/智能体",
    "AI 生产力工具",
    "行业垂直解决方案",
    "开发者工具/基础设施",
    "智能硬件/具身智能",
    "非产品",
    "其他产品",
]
LLM_SCENE_TAGS_ENUM: list[str] = [
    "文本/写作",
    "图像/视频/3D",
    "编程/代码",
    "搜索/资讯",
    "办公/协作",
    "营销/销售",
    "客服/支持",
    "金融/投资",
    "医疗/健康",
    "教育/科研",
    "游戏/娱乐",
    "其他",
]
LLM_PRODUCT_TYPES_SET = frozenset(LLM_PRODUCT_TYPES)
LLM_SCENE_TAGS_SET = frozenset(LLM_SCENE_TAGS_ENUM)
# 侧边栏「赛道/场景」多选：与 LLM 枚举一致，并与已打分条目中出现的标签合并
SCENE_TAG_PRESETS = LLM_SCENE_TAGS_ENUM

import feedparser
import pandas as pd
import streamlit as st
from openai import OpenAI


def _toast_with_icon(message: str, icon: str) -> None:
    """轻提示；无 st.toast 时跳过；不支持 icon 参数时回退为「图标 + 文案」。"""
    fn = getattr(st, "toast", None)
    if fn is None:
        return
    try:
        fn(message, icon=icon)
    except TypeError:
        fn(f"{icon} {message}")

# ============ 智谱 API 客户端 ============
# 使用 OpenAI 客户端格式调用智谱 API
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY", ""),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ============ SQLite 持久化 ============
# 数据库文件放在应用目录下，便于备份与迁移
DB_PATH = Path(__file__).resolve().parent / "ai_news.db"

# 连接级等待锁释放的秒数（默认 0 会立刻报 database is locked）
_SQLITE_BUSY_TIMEOUT_SEC = 30.0


def _configure_sqlite_connection(conn: sqlite3.Connection) -> None:
    """
    降低「database is locked」概率：
    - WAL：读写可更好并发（多标签页 Streamlit / 外部只读工具同时访问时更稳）。
    - busy_timeout：锁冲突时毫秒级重试，配合 connect timeout 使用。
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")


def _get_db_conn() -> sqlite3.Connection:
    """获取 SQLite 连接；row_factory 便于按列名取值。"""
    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        timeout=_SQLITE_BUSY_TIMEOUT_SEC,
    )
    conn.row_factory = sqlite3.Row
    _configure_sqlite_connection(conn)
    return conn


def init_db() -> None:
    """
    程序启动时初始化数据库与表结构。
    表 ai_news：link 唯一索引用于 INSERT OR IGNORE 去重；is_processed 标记 LLM 是否已处理。
    """
    conn = _get_db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT NOT NULL UNIQUE,
                published_at TEXT,
                source_name TEXT,
                region TEXT,
                raw_summary TEXT,
                is_processed INTEGER NOT NULL DEFAULT 0,
                company_name TEXT,
                product_type TEXT,
                scene_tags TEXT,
                business_summary TEXT,
                score INTEGER,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _normalize_scene_tags(raw) -> list[str]:
    """将 LLM 返回的 scene_tags 规范为字符串列表（最多 3 个）。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()][:3]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return _normalize_scene_tags(parsed)
        except json.JSONDecodeError:
            pass
        return [s.strip() for s in re.split(r"[,，;；]", raw) if s.strip()][:3]
    return []


def _clamp_product_type(value: str) -> str:
    """将 LLM 返回的 product_type 限制为固定枚举；非法值归为「其他产品」。"""
    v = (value or "").strip()
    # 旧版枚举「其他」与新版「其他产品」语义一致，自动升级
    if v == "其他":
        v = "其他产品"
    if v in LLM_PRODUCT_TYPES_SET:
        return v
    return "其他产品"


def _clamp_scene_tags_enum(tags: list[str]) -> list[str]:
    """将 scene_tags 限制为固定枚举，1–2 个；全部非法时用「其他」。"""
    out: list[str] = []
    for t in tags:
        t = (t or "").strip()
        if t in LLM_SCENE_TAGS_SET and t not in out:
            out.append(t)
        if len(out) >= 2:
            break
    return out if out else ["其他"]


def _parse_published_dt(published_at: str) -> datetime | None:
    """解析 published_at 字符串为 datetime，失败返回 None。"""
    if not published_at or published_at == "—":
        return None
    s = published_at.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:16] if fmt == "%Y-%m-%d %H:%M" else s[:10], fmt)
        except ValueError:
            continue
    return None


def _within_time_window(published_at: str, window_label: str) -> bool:
    """时间范围筛选；无法解析时间则保留条目。"""
    dt = _parse_published_dt(published_at)
    if dt is None:
        return True
    now = datetime.now()
    if window_label == "近 24 小时":
        return dt >= now - timedelta(hours=24)
    if window_label == "近 7 天":
        return dt >= now - timedelta(days=7)
    if window_label == "近 30 天":
        return dt >= now - timedelta(days=30)
    return True


def _apply_sidebar_filters(items: list[dict], f: dict) -> list[dict]:
    """根据侧边栏状态过滤列表。"""
    out: list[dict] = []
    min_score = float(f["min_score"])
    company_sel = f["company"]
    type_sel = f["product_type"]
    scene_sel = f["scene_tags"]
    only_heavy = f.get("only_heavy", False)
    time_win = f.get("time_range", "近 30 天")

    for it in items:
        sc = it.get("score")
        eff = float(sc) if sc is not None else 0.0
        if eff < min_score:
            continue
        if company_sel not in ("全部", "暂无"):
            if (it.get("company_name") or "—") != company_sel:
                continue
        if type_sel not in ("全部", "暂无"):
            if (it.get("product_type") or "—") != type_sel:
                continue
        if scene_sel:
            tags = set(it.get("scene_tags") or [])
            if not tags.intersection(set(scene_sel)):
                continue
        if only_heavy and not it.get("is_heavy"):
            continue
        if not _within_time_window(it.get("published_at", ""), time_win):
            continue
        out.append(it)
    return out


def analyze_news_with_llm(title: str, summary: str) -> dict:
    """
    调用智谱 LLM 提取结构化字段（含中文商业价值摘要、场景标签）。
    """
    fallback = {
        "company": "无",
        "product_type": "其他产品",
        "score": None,
        "reason": "无",
        "business_summary": "",
        "scene_tags": ["其他"],
    }
    if not os.getenv("ZHIPU_API_KEY"):
        return fallback

    prompt = f"""你是一位资深的 AI 产品演进与趋势分析专家。你的核心任务是：从新闻中敏锐地捕捉 AI 产品的实质性迭代。对于非产品向的噪音、研报、纯链接，必须坚决降权。

请根据以下新闻标题和摘要，严格按照 JSON 格式输出，不要输出其他内容。

标题：{title}
摘要：{summary}

【枚举约束（必须遵守，最高优先级）】
- product_type：**必须且只能**从下列列表中**原样复制恰好一个**字符串作为取值（含标点与空格须完全一致）。**绝对禁止**自造新词、同义词、合并说法或变体（例如禁止「AI产品」「AI产品研发」「人工智能产品」「大模型应用」等任何列表外写法）。若新闻为非产品向，必须选「非产品」；若为产品向但无法匹配前六类，必须选「其他产品」。
  可选列表（仅此 8 项，一字不差）：
{chr(10).join(f"  - {x}" for x in LLM_PRODUCT_TYPES)}

- scene_tags：**必须且只能**从下列列表中选择 **1 至 2 个** 字符串（每个都须与列表**完全一致**）。**绝对禁止**发明列表外的标签、近义词或细分新词。若无法匹配，只输出一个「其他」。
  可选列表（仅此 12 项，一字不差）：
{chr(10).join(f"  - {x}" for x in LLM_SCENE_TAGS_ENUM)}

【分析与评分步骤】
第一步：判断新闻性质。在 reason 中简短分析。若是空壳链接、宏观研报、资讯聚合、非产品向等，须严厉降权（1-2分），且 product_type 必须为「非产品」，scene_tags 可仅含「搜索/资讯」或「其他」等最贴近项。
第二步：确定 company；按上文枚举**逐字**选择唯一的 product_type；撰写 business_summary；按上文枚举选择 1-2 个 scene_tags。
第三步：输出完整 JSON。

【字段规范】
- business_summary：**必须使用中文**，撰写**约 80–100 字**的详细摘要，须覆盖：核心事实（产品/模型/版本或事件）、关键数据或指标（若原文有则写出）、以及对读者或行业的意义；若原文为外文，请翻译后再写。非产品向或无法提炼时可填「无实质产品信息」。

请输出以下格式的 JSON（必须严格包含这 6 个字段；product_type、scene_tags 的取值只能来自上述枚举）：
{{"reason": "简短的判断理由", "company": "公司名或'无'", "product_type": "（从枚举中择一，原样复制）", "score": 1-5的整数, "business_summary": "中文详细商业价值摘要（约80-100字）", "scene_tags": ["枚举标签1", "枚举标签2"]}}

【严格评分标准（1-5分）】
5分：头部大厂发布跨代际的新模型，或颠覆性的全新AI产品形态。
4分：知名AI产品推出核心新功能，或重要的开源模型发布，且摘要中有明确的实质性内容。
3分：普通AI应用的发布，常规的功能微调。
2分（非产品向）：券商研报、宏观政策、纯硬件/算力/电力设施、AI公司的融资、高管变动、商业合作。
1分（噪音/无内容）：摘要无实质内容（如仅含URL和Comments）、资讯聚合、空公关稿。
"""

    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        for start in ("{", "```json", "```"):
            idx = content.find(start)
            if idx >= 0:
                if start != "{":
                    idx = content.find("{", idx)
                if idx >= 0:
                    end = content.rfind("}") + 1
                    if end > idx:
                        content = content[idx:end]
                break
        data = json.loads(content)
        company = data.get("company") or "无"
        product_type = _clamp_product_type(str(data.get("product_type") or ""))
        score = data.get("score")
        if score is not None:
            try:
                score = int(score)
                score = max(1, min(5, score))
            except (ValueError, TypeError):
                score = None
        bs = (data.get("business_summary") or "").strip()
        tags = _clamp_scene_tags_enum(_normalize_scene_tags(data.get("scene_tags")))
        return {
            "company": company,
            "product_type": product_type,
            "score": score,
            "reason": data.get("reason", "无"),
            "business_summary": bs,
            "scene_tags": tags,
        }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, Exception):
        return fallback


# ============ RSS 源配置 ============
# 结构化信源与过滤配置，支持全球化 AI 资讯
RSS_SOURCES_CONFIG = {
    "官方首发 (Official)": [
        {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml", "region": "海外"},
        {"name": "Google AI", "url": "https://blog.google/technology/ai/rss/", "region": "海外"},
        {"name": "Hugging Face", "url": "https://huggingface.co/blog/feed.xml", "region": "海外"},
    ],
    "顶级科技媒体 (Tech Media)": [
        {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "region": "海外"},
        {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "region": "海外"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "region": "海外"},
        {"name": "Wired AI", "url": "https://www.wired.com/feed/tag/ai/latest/rss", "region": "海外"},
    ],
    "前沿与社区策展 (Newsletters)": [
        {"name": "Latent Space", "url": "https://www.latent.space/feed", "region": "海外"},
        {"name": "Import AI", "url": "https://importai.substack.com/feed", "region": "海外"},
        {"name": "Product Hunt", "url": "https://www.producthunt.com/feed", "region": "海外"},
        {"name": "Techmeme", "url": "https://www.techmeme.com/feed.xml", "region": "海外"},
    ],
    "国内精选 (Domestic)": [
        {"name": "少数派", "url": "https://sspai.com/feed", "region": "国内"},
        # 借助 RSSHub 抓取的国内顶级 AI 垂直媒体
        {"name": "机器之心 (前沿/深度)", "url": "https://rsshub.app/jiqizhixin/latest", "region": "国内"},
        {"name": "极客公园 (AGI/产品逻辑)", "url": "https://rsshub.app/geekpark/news", "region": "国内"},
        {"name": "InfoQ AI (开发者/底层技术)", "url": "https://rsshub.app/infoq/topic/33", "region": "国内"},
    ],
}

# 扁平化列表供抓取逻辑使用
RSS_SOURCES = [s for sources in RSS_SOURCES_CONFIG.values() for s in sources]

# ============ 中英双语 AI 硬过滤关键词库 ============
AI_KEYWORDS = [
    # 英文
    "LLM", "AI", "AGI", "Agent", "Generative", "RAG", "Transformer", "GPU",
    "OpenAI", "ChatGPT", "GPT", "Anthropic", "Claude", "DeepMind", "Gemini",
    "Meta", "Llama", "Midjourney", "Sora", "Copilot",
    # 中文
    "大模型", "人工智能", "智能体", "生成式", "算力", "月之暗面", "智谱",
    "具身智能", "机器人", "自动驾驶", "AI创业",
]


def _matches_ai_keywords(title: str, summary: str) -> bool:
    """
    判断标题或摘要中是否包含 AI_KEYWORDS 中的任意关键词。
    忽略大小写（英文），中文原样匹配。
    """
    combined = f"{title} {summary}".lower()
    for kw in AI_KEYWORDS:
        if kw.lower() in combined or kw in combined:
            return True
    return False


def _md_inline_safe(text: str) -> str:
    """避免 * _ 等破坏 Markdown 加粗/斜体。"""
    return (text or "").replace("*", "＊").replace("_", "＿")


def _md_href(url: str) -> str:
    """Markdown 链接内 URL 中的括号易导致截断，做编码。"""
    u = (url or "").strip()
    if not u or u == "#":
        return "#"
    return u.replace("(", "%28").replace(")", "%29")


def _strip_html(text: str) -> str:
    """去除 HTML 标签，保留纯文本，便于在 Streamlit 中展示。"""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()[:500]


def _format_published(entry) -> str:
    """
    从 feedparser 的 entry 中解析发布时间，格式化为可读字符串。
    支持 published_parsed、updated_parsed 或 published/updated 字符串。
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            dt = datetime.fromtimestamp(mktime(parsed))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            pass
    published = entry.get("published") or entry.get("updated") or ""
    return published[:20] if published else "—"


def _aggregate_news(items: list[dict]) -> list[dict]:
    """
    按 company_name 多源聚合：同公司仅保留 score 最高的一条，source_count 累加被合并条数；
    「无」「—」及空公司名不参与聚合。聚合后 source_count >= 2 且 score >= 4 则标记 is_heavy。
    """
    if not items:
        return []
    skip = frozenset({"无", "—"})
    buckets: dict[str, list[dict]] = {}
    standalone: list[dict] = []

    for it in items:
        cn = (it.get("company_name") or "").strip()
        if not cn or cn in skip:
            standalone.append(dict(it))
        else:
            buckets.setdefault(cn, []).append(it)

    merged: list[dict] = []

    def _pick_main(group: list[dict]) -> dict:
        return max(
            group,
            key=lambda x: (
                float(x.get("score")) if x.get("score") is not None else -1.0,
                x.get("published_at") or "",
            ),
        )

    for _company, group in buckets.items():
        if len(group) == 1:
            m = dict(group[0])
        else:
            main = _pick_main(group)
            m = dict(main)
            base_sc = int(m.get("source_count") or 1)
            m["source_count"] = base_sc + (len(group) - 1)
        merged.append(m)

    out = standalone + merged
    for m in out:
        sc = m.get("score")
        sc_val = float(sc) if sc is not None else 0.0
        cnt = int(m.get("source_count") or 1)
        m["is_heavy"] = cnt >= 2 and sc_val >= 4
    return out


def fetch_rss_news() -> tuple[int, int]:
    """
    抓取各 RSS 源，经硬过滤后增量写入 ai_news。
    使用 INSERT OR IGNORE，以 link 唯一约束去重；不返回内存列表。

    返回：(本次尝试写入条数, 实际新插入条数)。
    """
    attempted = 0
    inserted = 0
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        for source in RSS_SOURCES:
            try:
                feed = feedparser.parse(
                    source["url"],
                    agent="Mozilla/5.0 (compatible; AI News Radar/1.0)",
                    etag=None,
                )
            except Exception:
                continue

            if feed.bozo and not getattr(feed, "entries", None):
                continue

            entries = feed.entries or []
            for entry in entries[:15]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title or not link:
                    continue

                summary = entry.get("summary") or entry.get("description", "")
                if hasattr(summary, "value"):
                    summary = summary.value if summary else ""
                summary = _strip_html(str(summary)) if summary else ""

                if not _matches_ai_keywords(title, summary):
                    continue

                if any(kw in title for kw in ["早报", "晚报", "8点1氪", "热点导览"]):
                    continue

                if "Article URL:" in summary and "Comments URL:" in summary:
                    text_only = re.sub(r"https?://\S+", "", summary)
                    text_only = re.sub(
                        r"(Article URL:|Comments URL:|Points:|Comments:)", "", text_only
                    ).strip()
                    if len(text_only) < 20:
                        continue

                published_at = _format_published(entry)
                raw_summary = summary or "暂无摘要"
                attempted += 1
                cur.execute(
                    """
                    INSERT OR IGNORE INTO ai_news (
                        title, link, published_at, source_name, region,
                        raw_summary, is_processed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now', 'localtime'))
                    """,
                    (
                        title,
                        link,
                        published_at,
                        source["name"],
                        source["region"],
                        raw_summary,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
            # 按源提交，缩短写锁占用时间，避免与另一次页面运行或其它进程长时间互斥
            conn.commit()
    finally:
        conn.close()
    if inserted > 0:
        _toast_with_icon(
            f"📡 抓取到 {inserted} 条全新资讯，正在排队等待 AI 分析...",
            icon="⏳",
        )
    return attempted, inserted


def process_unscored_news_from_db(batch_limit: int = 25) -> int:
    """
    从库中读取 is_processed = 0 且仍符合 AI 关键词硬过滤的行，
    调用 analyze_news_with_llm，将结果 UPDATE 回库并置 is_processed = 1。

    每轮最多处理 batch_limit 条（默认与原先「精选前 25 条」量级一致），避免单次页面加载过久。
    返回：本轮成功标记为已处理的条数。
    """
    conn = _get_db_conn()
    processed_count = 0
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, raw_summary, published_at
            FROM ai_news
            WHERE is_processed = 0
            ORDER BY
                CASE WHEN published_at IS NULL OR TRIM(published_at) = '' OR published_at = '—'
                     THEN created_at ELSE published_at END DESC
            LIMIT ?
            """,
            (batch_limit,),
        )
        rows = cur.fetchall()
        for row in rows:
            rid = row["id"]
            title = row["title"] or ""
            raw_summary = row["raw_summary"] or ""
            if not _matches_ai_keywords(title, raw_summary):
                continue
            result = analyze_news_with_llm(title, raw_summary)
            tags = _clamp_scene_tags_enum(_normalize_scene_tags(result.get("scene_tags")))
            scene_json = json.dumps(tags, ensure_ascii=False)
            product_type = _clamp_product_type(str(result.get("product_type") or ""))
            company = result.get("company") or "无"
            score = result.get("score")
            if score is not None:
                try:
                    score = int(score)
                except (TypeError, ValueError):
                    score = None
            reason = result.get("reason", "无")
            business_summary = (result.get("business_summary") or "").strip()
            cur.execute(
                """
                UPDATE ai_news SET
                    company_name = ?,
                    product_type = ?,
                    scene_tags = ?,
                    business_summary = ?,
                    score = ?,
                    reason = ?,
                    is_processed = 1
                WHERE id = ?
                """,
                (
                    company,
                    product_type,
                    scene_json,
                    business_summary,
                    score,
                    reason,
                    rid,
                ),
            )
            processed_count += 1
        conn.commit()
    finally:
        conn.close()
    if processed_count > 0:
        _toast_with_icon(
            f"🤖 刚刚成功分析并入库了 {processed_count} 条新资讯！",
            icon="✅",
        )
    return processed_count


def _row_to_ui_item(row: sqlite3.Row) -> dict:
    """将数据库行转为与卡片/侧边栏一致的 dict（RSS 摘要 vs LLM 摘要字段对齐）。"""
    raw = (row["raw_summary"] or "").strip() or "暂无摘要"
    llm_bs = (row["business_summary"] or "").strip()
    tags_raw = row["scene_tags"] or ""
    try:
        tags = json.loads(tags_raw) if tags_raw.strip().startswith("[") else None
    except json.JSONDecodeError:
        tags = None
    if tags is None:
        tags = [s.strip() for s in re.split(r"[,，;；]", tags_raw) if s.strip()]
    score = row["score"]
    return {
        "id": row["id"],
        "title": row["title"] or "",
        "link": row["link"] or "",
        "published_at": row["published_at"] or "—",
        "business_summary": raw,
        "llm_business_summary": llm_bs,
        "product_name": "—",
        "company_name": row["company_name"] or "—",
        "score": float(score) if score is not None else None,
        "product_type": row["product_type"] or "—",
        "scene_tags": tags if isinstance(tags, list) else [],
        "reason": row["reason"] or "无",
        "is_heavy": False,
        "source_count": 1,
        "region": row["region"] or "—",
        "source_name": row["source_name"] or "—",
    }


def get_news_from_db(time_range: str = "近 30 天") -> list[dict]:
    """
    读取已 LLM 处理的数据，按时间窗口过滤（无有效发布时间时用 created_at），
    排序：score DESC（NULL 在后）、published_at DESC。

    「近 24 小时」使用 timedelta(hours=24) 滚动窗口；7/30 天使用自然日数。
    """
    if time_range == "近 24 小时":
        cutoff_dt = datetime.now() - timedelta(hours=24)
    elif time_range == "近 7 天":
        cutoff_dt = datetime.now() - timedelta(days=7)
    else:
        cutoff_dt = datetime.now() - timedelta(days=30)
    cutoff = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, link, published_at, source_name, region, raw_summary,
                   company_name, product_type, scene_tags, business_summary, score, reason, created_at
            FROM ai_news
            WHERE is_processed = 1
              AND (
                (
                  published_at IS NOT NULL
                  AND TRIM(published_at) != ''
                  AND published_at != '—'
                  AND published_at >= ?
                )
                OR (
                  (published_at IS NULL OR TRIM(published_at) = '' OR published_at = '—')
                  AND created_at >= ?
                )
              )
            ORDER BY (score IS NULL) ASC, score DESC, published_at DESC
            """,
            (cutoff[:16], cutoff),
        )
        rows = cur.fetchall()
        return [_row_to_ui_item(r) for r in rows]
    finally:
        conn.close()


def _mock_news_fallback() -> list[dict]:
    """
    当 RSS 抓取失败或返回空时使用的假数据兜底。
    保证页面始终可展示，便于调试与演示。
    """
    return [
        {
            "title": "Claude 推出团队协作新能力：企业知识库一键接入",
            "link": "https://example.com/1",
            "product_name": "Claude for Work",
            "company_name": "Anthropic",
            "score": 4.6,
            "product_type": "AI 生产力工具",
            "scene_tags": ["办公/协作"],
            "business_summary": "面向企业团队的知识库接入与权限控制，降低落地门槛；更易形成按席位/用量计费的稳定收入。",
            "llm_business_summary": "企业团队可一键接入知识库并做权限管理，降低落地成本，利于按席位或用量形成稳定商业化。",
            "reason": "明确的产品能力发布，面向 B 端协作场景。",
            "is_heavy": True,
            "source_count": 3,
            "published_at": "2026-03-18 09:10",
            "region": "海外",
            "source_name": "Mock",
        },
    ]


def _init_page() -> None:
    st.set_page_config(
        page_title="智能 AI 产品资讯雷达（MVP）",
        page_icon="📡",
        layout="wide",
    )


def _scene_tags_from_item(it: dict) -> list[str]:
    """从单条资讯解析 scene_tags（list / JSON 字符串 / 逗号分隔）。"""
    raw = it.get("scene_tags")
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            pass
        return [x.strip() for x in re.split(r"[,，;；]", s) if x.strip()]
    return []


def _product_types_from_item(it: dict) -> list[str]:
    """
    从单条资讯解析产品/事件类型：依次尝试 product_type、event_type、category；
    支持 list、JSON 数组字符串、逗号分隔或单字符串。
    """
    def _parse(raw) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if str(t).strip()]
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if str(t).strip()]
            except json.JSONDecodeError:
                pass
            parts = [x.strip() for x in re.split(r"[,，;；]", s) if x.strip()]
            return parts if len(parts) > 1 else [s]
        return []

    for key in ("product_type", "event_type", "category"):
        out = _parse(it.get(key))
        if out:
            return out
    return []


def render_trend_dashboard(items: list[dict]) -> None:
    """
    数据趋势看板：赛道标签频次 Top10（剔除泛化标签）、产品/事件类型分布 Top10。
    """
    if len(items) < 3:
        st.info("当前筛选条件下数据量过少，无法生成趋势图表")
        return

    skip_scene_tags = frozenset({"其他", "未知", "无"})
    skip_product_types = frozenset({"其他", "未知", "无", "常规", "—"})

    with st.expander("📊 展开查看当前数据趋势分析", expanded=True):
        col_left, col_right = st.columns(2)

        with col_left:
            st.caption("赛道热度分布（已剔除「其他/未知/无」· 按频次降序 · Top 10）")
            all_tags: list[str] = []
            for it in items:
                for t in _scene_tags_from_item(it):
                    t = t.strip()
                    if t and t not in skip_scene_tags:
                        all_tags.append(t)
            if not all_tags:
                st.caption("暂无有效赛道标签")
            else:
                tag_counts = pd.Series(all_tags).value_counts().head(10)
                df_tags = pd.DataFrame({"出现次数": tag_counts.values}, index=tag_counts.index)
                df_tags = df_tags.sort_values("出现次数", ascending=False)
                st.bar_chart(df_tags)

        with col_right:
            st.caption("产品/事件类型分布（已剔除无效分类 · 按频次降序 · Top 10）")
            all_pt: list[str] = []
            for it in items:
                for p in _product_types_from_item(it):
                    p = p.strip()
                    if p and p not in skip_product_types:
                        all_pt.append(p)
            if not all_pt:
                st.caption("暂无有效产品/事件类型")
            else:
                pt_counts = pd.Series(all_pt).value_counts().head(10)
                df_pt = pd.DataFrame({"出现次数": pt_counts.values}, index=pt_counts.index)
                df_pt = df_pt.sort_values("出现次数", ascending=False)
                st.bar_chart(df_pt)


def _render_sidebar(all_items: list[dict], time_range: str) -> dict:
    """侧边栏筛选器（时间范围由 main 顶部预先渲染，以便先按天数从库中拉数）。"""
    discovered_tags = {t for it in all_items for t in (it.get("scene_tags") or []) if t}
    tag_options = sorted(set(SCENE_TAG_PRESETS) | discovered_tags)
    if not tag_options:
        tag_options = ["暂无"]

    companies = sorted({c for it in all_items for c in [it.get("company_name")] if c and c != "—"})
    product_types = sorted({p for it in all_items for p in [it.get("product_type")] if p and p != "—"})

    scene_pick = st.sidebar.multiselect(
        "赛道/场景（多选，命中任一即保留）",
        options=tag_options,
        default=[],
        help="不选表示不按场景过滤；可多选",
    )
    company_pick = st.sidebar.selectbox(
        "公司",
        options=["全部"] + (companies if companies else ["暂无"]),
        index=0,
    )
    type_pick = st.sidebar.selectbox(
        "产品类型",
        options=["全部"] + (product_types if product_types else ["暂无"]),
        index=0,
    )
    min_score = st.sidebar.slider("评分下限", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    only_heavy = st.sidebar.checkbox("仅看重磅", value=False)

    st.sidebar.divider()
    st.sidebar.subheader("显示设置")
    layout_mode = st.sidebar.radio(
        "页面布局",
        options=["经典左右分栏", "重磅摘要置顶"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("数据说明")
    st.sidebar.write(
        "- 资讯已写入本地 SQLite（ai_news.db），link 去重；未处理条目由 LLM 分批打分后回写。\n"
        "- 上方「时间范围」决定从库中回溯：滚动 24 小时 / 7 天 / 30 天已处理数据；卡片为筛选后 Top 10。\n"
        "- 评分≥4 的资讯在标题旁显示 🔥。"
    )

    st.sidebar.divider()
    st.sidebar.subheader("开发者调试")
    if st.sidebar.button("🔄 刷新页面（重新抓取并继续分析）", use_container_width=True):
        st.session_state.last_update_time = 0
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    # 多选选了占位「暂无」时不生效
    effective_scenes = [x for x in scene_pick if x != "暂无"]

    return {
        "scene_tags": effective_scenes,
        "company": company_pick,
        "product_type": type_pick,
        "min_score": min_score,
        "only_heavy": only_heavy,
        "time_range": time_range,
        "layout_mode": layout_mode,
    }


def _sidebar_time_range() -> str:
    """在数据加载前渲染时间范围，供 get_news_from_db 与侧边栏筛选共用同一标签。"""
    st.sidebar.title("筛选器")
    st.sidebar.caption("最终阶段：版面微调 + 可视化辅助分析。")
    return st.sidebar.radio(
        "时间范围",
        options=["近 24 小时", "近 7 天", "近 30 天"],
        index=2,
        help="近 24 小时为滚动 24 小时（timedelta(hours=24)）；7/30 天为自然日窗口。",
    )


def _ensure_session_state() -> None:
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = set()
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = {}


def _render_card(item: dict, idx: int) -> None:
    """紧凑卡片：标题+评分一行，元数据 caption，摘要，底部四列操作。"""
    item_uid = item.get("id", idx)
    key_read = f"read_{item_uid}_{idx}"
    key_fav = f"fav_{item_uid}_{idx}"
    key_good = f"good_{item_uid}_{idx}"
    key_bad = f"bad_{item_uid}_{idx}"

    title_plain = (item.get("title") or "无标题").strip()
    safe_title = title_plain.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")
    heavy = "🚨 " if item.get("is_heavy") else ""
    sc = item.get("score")
    try:
        fire = "🔥 " if sc is not None and float(sc) >= 4 else ""
    except (TypeError, ValueError):
        fire = ""
    title_line = f"{heavy}{fire}**{safe_title}**"

    pub = item.get("published_at", "—")
    src = item.get("source_name", "—")
    if int(item.get("source_count") or 1) > 1:
        src = f"{src} · 多源×{item['source_count']}"
    company = item.get("company_name") or "—"
    ptype = item.get("product_type") or "—"
    tags_list = item.get("scene_tags") or []
    tags = " · ".join(str(t) for t in tags_list) if tags_list else "—"
    meta_line = f"{pub} · {src} | {company} | {ptype} | {tags}"

    llm_sum = (item.get("llm_business_summary") or "").strip()
    summary = llm_sum or (item.get("business_summary") or "").strip() or "暂无摘要"

    try:
        card_ctx = st.container(border=True)
    except TypeError:
        card_ctx = st.container()
    with card_ctx:
        c_title, c_score = st.columns([4, 1], vertical_alignment="center")
        with c_title:
            st.markdown(title_line)
        with c_score:
            if sc is not None:
                try:
                    st.markdown(
                        f'<p style="text-align:right;margin:0;"><b>⭐ {float(sc):.1f}</b></p>',
                        unsafe_allow_html=True,
                    )
                except (TypeError, ValueError):
                    st.markdown(
                        '<p style="text-align:right;margin:0;"><b>⭐ —</b></p>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<p style="text-align:right;margin:0;"><b>⭐ —</b></p>',
                    unsafe_allow_html=True,
                )

        st.caption(meta_line)
        # 摘要为 llm_business_summary 或 RSS 原文摘要的全文展示，UI 不做 [:n] 截断；更长内容依赖 LLM Prompt（见 analyze_news_with_llm）。
        st.write(summary)

        b0, b1, b2, b3, _btn_spacer = st.columns(
            [1.5, 1.5, 1, 1, 6], vertical_alignment="center"
        )
        with b0:
            link = (item.get("link") or "").strip()
            if link:
                st.link_button("阅读原文", link, type="tertiary")
            else:
                st.button("阅读原文", disabled=True, key=key_read, type="tertiary")
        with b1:
            is_fav = idx in st.session_state["favorites"]
            fav_label = "已收藏" if is_fav else "收藏"
            if st.button(fav_label, key=key_fav, type="tertiary"):
                if is_fav:
                    st.session_state["favorites"].discard(idx)
                else:
                    st.session_state["favorites"].add(idx)
        with b2:
            if st.button("👍", key=key_good, type="tertiary"):
                st.session_state["feedback"][idx] = "有价值"
        with b3:
            if st.button("👎", key=key_bad, type="tertiary"):
                st.session_state["feedback"][idx] = "无价值"


def main() -> None:
    _init_page()
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = 0
    _ensure_session_state()
    init_db()

    # 先渲染时间范围，再按同一标签从库中按窗口查询
    time_range = _sidebar_time_range()

    # 24 小时冷却：避免频繁请求 RSS / LLM；侧栏刷新将 last_update_time 置 0 可跳过冷却
    current_time = time.time()
    if current_time - st.session_state.last_update_time > 86400:
        with st.spinner("正在拉取 RSS 并增量写入数据库..."):
            fetch_rss_news()
        with st.spinner("正在用智谱 LLM 分析库中未处理资讯..."):
            process_unscored_news_from_db()
        st.session_state.last_update_time = current_time

    items = get_news_from_db(time_range=time_range)

    if not items:
        st.warning(
            "当前时间范围内暂无已分析资讯，或库中尚无数据；以下为兜底示例。可扩大「时间范围」或稍后刷新。"
        )
        items = _mock_news_fallback()

    items = _aggregate_news(items)

    st.title("📡 智能 AI 产品资讯雷达（MVP）")
    st.caption(
        "最终阶段：版面微调 + 可视化辅助分析；SQLite 持久化与 LLM 流水线保持不变。"
    )

    # 库中已按 score、publishes 排序；此处保留重磅优先的二次排序以兼容 mock / 后续多源聚合
    items_sorted = sorted(
        items,
        key=lambda x: (
            not x.get("is_heavy", False),
            x.get("score") or 0,
            x.get("published_at", ""),
        ),
        reverse=True,
    )

    top_items = items_sorted[:25]
    tail_items = items_sorted[25:]

    filters = _render_sidebar(items_sorted, time_range)
    filtered_top = _apply_sidebar_filters(top_items, filters)
    filtered_tail = _apply_sidebar_filters(tail_items, filters)

    CARD_DISPLAY_N = 10
    card_items = filtered_top[:CARD_DISPLAY_N]
    more_items = filtered_top[CARD_DISPLAY_N:] + filtered_tail

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("资讯条数", str(len(items_sorted)))
    with kpi2:
        st.metric("重磅条数", str(sum(1 for it in items_sorted if it.get("is_heavy"))))
    with kpi3:
        st.metric("已收藏", str(len(st.session_state["favorites"])))
    with kpi4:
        st.metric("已反馈", str(len(st.session_state["feedback"])))

    st.divider()

    def _render_cards_only() -> None:
        st.subheader(f"精选资讯（卡片列表 · Top {CARD_DISPLAY_N}）")
        st.caption(
            f"侧边栏筛选后：精选池 {len(filtered_top)} 条 · 卡片 {len(card_items)} 条 · "
            f"简略列表 {len(more_items)} 条"
        )
        for idx, item in enumerate(card_items):
            _render_card(item, idx=idx)
            st.write("")

    def _render_more_items_section() -> None:
        if not more_items:
            return
        st.divider()
        st.subheader("更多资讯（简略列表）")
        lines = []
        for it in more_items:
            pt = it.get("published_at", "—")
            src = it.get("source_name", "—")
            title = (it.get("title") or "").replace("|", "｜")
            link = it.get("link") or "#"
            lines.append(f"| {pt} | {src} | [{title}]({link}) |")
        header = "| 发布时间 | 来源 | 标题 |"
        sep = "| :--- | :--- | :--- |"
        st.markdown("\n".join([header, sep] + lines))

    def _render_summary_section() -> None:
        st.subheader("今日摘要")
        st.caption("重磅事件 · 标题为加粗正文，点击右侧 ↗ 在新标签页打开原文")

        heavy_news = [x for x in items_sorted if x.get("is_heavy", False)]

        if heavy_news:
            for news in heavy_news:
                title = news.get("title", "未知标题")
                source = news.get("source_name", "未知来源")
                link = news.get("link") or "#"
                href = _md_href(link)
                st.markdown(
                    f"🚨 **{_md_inline_safe(str(title))}** - *{_md_inline_safe(str(source))}* &nbsp; "
                    f"[↗]({href})"
                )
        else:
            st.info("当前筛选条件下暂无重磅事件")

    if filters.get("layout_mode") == "重磅摘要置顶":
        _render_summary_section()
        st.divider()
        _render_cards_only()
        _render_more_items_section()
    else:
        left, right = st.columns([3, 2], vertical_alignment="top")
        with left:
            _render_cards_only()
            _render_more_items_section()
        with right:
            _render_summary_section()

    st.divider()
    render_trend_dashboard(items_sorted)
    st.divider()
    st.subheader("使用提示")
    st.write(
        "- 仅展示标题或摘要含 AI 关键词的资讯（中英双语过滤），入库时 link 去重。\n"
        "- 未处理行由智谱 LLM 分批打分并写回 SQLite；卡片主文为 AI 提炼摘要。\n"
        "- 侧边栏「时间范围」控制库中回溯天数；其余筛选对精选池与「更多」同时生效；评分≥4 显示 🔥。\n"
        "- 数据持久化在 ai_news.db，刷新页面会继续增量抓取与处理队列。"
    )


if __name__ == "__main__":
    main()
