"""
Playwright MCP 智能浏览器代理 - 终极版
=====================================
特性:
  - 智能元素定位与模糊匹配
  - 多策略错误恢复
  - 操作链优化与预测
  - 自适应等待机制
  - 页面状态机管理
  - 智能重试与降级
  - 完整的操作审计日志
  - 性能监控与优化
"""

import asyncio
import json
import sys
import os
import re
import hashlib
from datetime import datetime
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ==================== 配置管理 ====================

class Config:
    """全局配置"""
    
    # 路径
    BASE_PATH = Path(__file__).parent if not getattr(sys, 'frozen', False) else Path(sys.executable).parent
    CONFIG_FILE = BASE_PATH / "config.json"
    SESSION_FILE = BASE_PATH / "session.json"
    LOG_FILE = BASE_PATH / "agent.log"
    
    # API
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-reasoner"
    
    # 操作限制
    MAX_ITERATIONS = 50
    MAX_RETRIES = 3
    MAX_HISTORY_LENGTH = 100
    
    # 超时设置 (秒)
    TOOL_TIMEOUT = 120
    CONNECT_TIMEOUT = 60
    API_TIMEOUT = 180
    
    # 智能等待 (秒) - (最小等待, 最大等待)
    WAIT_TIMES = {
        'browser_navigate': (2.0, 5.0),
        'browser_click': (0.8, 2.5),
        'browser_type': (0.3, 1.0),
        'browser_select_option': (0.8, 2.0),
        'browser_press_key': (0.5, 1.5),
        'browser_go_back': (1.5, 4.0),
        'browser_go_forward': (1.5, 4.0),
        'browser_scroll_down': (0.5, 1.5),
        'browser_scroll_up': (0.5, 1.5),
        'browser_tab_new': (1.0, 3.0),
        'browser_tab_close': (0.5, 1.5),
    }
    
    @classmethod
    def load(cls) -> dict:
        """加载配置文件"""
        if cls.CONFIG_FILE.exists():
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save(cls, data: dict):
        """保存配置文件"""
        with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def get_api_key(cls) -> str:
        """获取 API Key"""
        config = cls.load()
        return config.get("api_key", "") or os.getenv("DEEPSEEK_API_KEY", "")
    
    @classmethod
    def set_api_key(cls, key: str):
        """保存 API Key"""
        config = cls.load()
        config["api_key"] = key
        cls.save(config)


# ==================== 日志系统 ====================

class LogLevel(Enum):
    """日志级别"""
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()


class Logger:
    """日志管理器"""
    
    ICONS = {
        LogLevel.DEBUG: "🔍",
        LogLevel.INFO: "📝",
        LogLevel.WARN: "⚠️",
        LogLevel.ERROR: "❌"
    }
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO, to_file: bool = False, sink=None):
        self.name = name
        self.level = level
        self.to_file = to_file
        self.sink = sink
        self.logs: deque = deque(maxlen=500)
    
    def _log(self, level: LogLevel, msg: str):
        """记录日志"""
        if level.value < self.level.value:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = self.ICONS.get(level, "")
        formatted = f"[{timestamp}] {icon} {msg}"

        self.logs.append({
            "time": timestamp,
            "level": level.name,
            "msg": msg
        })

        if level.value >= LogLevel.INFO.value:
            print(f"   {formatted}")

        if self.to_file:
            with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(formatted + "\n")

        if self.sink:
            try:
                self.sink(formatted)
            except Exception:
                pass
    
    def debug(self, msg: str):
        self._log(LogLevel.DEBUG, msg)
    
    def info(self, msg: str):
        self._log(LogLevel.INFO, msg)
    
    def warn(self, msg: str):
        self._log(LogLevel.WARN, msg)
    
    def error(self, msg: str):
        self._log(LogLevel.ERROR, msg)
    
    def get_recent(self, n: int = 20) -> list[dict]:
        """获取最近的日志"""
        return list(self.logs)[-n:]


# ==================== 页面状态机 ====================

class PageState(Enum):
    """页面状态枚举"""
    UNKNOWN = auto()      # 未知状态
    LOADING = auto()      # 加载中
    READY = auto()        # 就绪
    INTERACTIVE = auto()  # 可交互
    ERROR = auto()        # 错误
    BLOCKED = auto()      # 被弹窗等阻挡


@dataclass
class ElementInfo:
    """元素信息"""
    ref: str                                    # 元素引用
    tag: str = ""                               # 标签名
    text: str = ""                              # 文本内容
    role: str = ""                              # 角色
    attributes: dict = field(default_factory=dict)  # 属性
    position: str = ""                          # 位置描述
    confidence: float = 1.0                     # 匹配置信度
    
    def matches(self, query: str) -> float:
        """
        计算与查询的匹配度
        
        Args:
            query: 查询字符串
            
        Returns:
            匹配度 (0-1)
        """
        query = query.lower().strip()
        score = 0.0
        
        # 精确匹配 - 最高分
        if query == self.text.lower().strip():
            return 1.0
        if query == self.ref.lower():
            return 1.0
        
        # 包含匹配
        if query in self.text.lower():
            score = max(score, 0.8)
        if query in str(self.attributes).lower():
            score = max(score, 0.6)
        if query in self.role.lower():
            score = max(score, 0.5)
        
        # 关键词匹配
        query_words = set(query.split())
        text_words = set(self.text.lower().split())
        if query_words and text_words:
            overlap = query_words & text_words
            if overlap:
                score = max(score, len(overlap) / len(query_words) * 0.7)
        
        return score


@dataclass
class PageSnapshot:
    """页面快照"""
    url: str = ""
    title: str = ""
    content: str = ""
    elements: list[ElementInfo] = field(default_factory=list)
    state: PageState = PageState.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    content_hash: str = ""
    
    def __post_init__(self):
        """初始化后计算内容哈希"""
        if self.content and not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
    
    def is_stale(self, seconds: float = 30) -> bool:
        """
        检查快照是否过期
        
        Args:
            seconds: 过期时间（秒）
            
        Returns:
            是否过期
        """
        return (datetime.now() - self.timestamp).total_seconds() > seconds
    
    def find_element(self, query: str, threshold: float = 0.5) -> Optional[ElementInfo]:
        """
        智能查找单个元素
        
        Args:
            query: 查询字符串
            threshold: 匹配阈值
            
        Returns:
            匹配的元素或 None
        """
        if not self.elements:
            return None
        
        # 精确 ref 匹配
        for el in self.elements:
            if el.ref == query or f"ref={query}" == el.ref:
                return el
        
        # 模糊匹配
        candidates = []
        for el in self.elements:
            score = el.matches(query)
            if score >= threshold:
                candidates.append((score, el))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None
    
    def find_elements(self, query: str, threshold: float = 0.3, limit: int = 5) -> list[ElementInfo]:
        """
        查找多个匹配元素
        
        Args:
            query: 查询字符串
            threshold: 匹配阈值
            limit: 返回数量限制
            
        Returns:
            匹配的元素列表
        """
        candidates = []
        for el in self.elements:
            score = el.matches(query)
            if score >= threshold:
                el.confidence = score
                candidates.append((score, el))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [el for _, el in candidates[:limit]]


class BrowserStateManager:
    """浏览器状态管理器"""
    
    def __init__(self):
        self.current_snapshot: Optional[PageSnapshot] = None
        self.snapshot_history: deque[PageSnapshot] = deque(maxlen=20)
        self.page_state: PageState = PageState.UNKNOWN
        self.navigation_stack: list[str] = []
        self.blocked_by: Optional[str] = None
    
    def update_snapshot(self, raw_content: str) -> PageSnapshot:
        """
        解析并更新快照
        
        Args:
            raw_content: 原始快照内容
            
        Returns:
            解析后的快照对象
        """
        snapshot = self._parse_snapshot(raw_content)
        snapshot.state = self._detect_state(snapshot)
        
        # 保存历史
        if self.current_snapshot:
            self.snapshot_history.append(self.current_snapshot)
        
        self.current_snapshot = snapshot
        self.page_state = snapshot.state
        
        # 更新导航栈
        if snapshot.url:
            if not self.navigation_stack or self.navigation_stack[-1] != snapshot.url:
                self.navigation_stack.append(snapshot.url)
                if len(self.navigation_stack) > 50:
                    self.navigation_stack = self.navigation_stack[-30:]
        
        return snapshot
    
    def _parse_snapshot(self, content: str) -> PageSnapshot:
        """解析快照内容"""
        snapshot = PageSnapshot(content=content)
        elements = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # 提取 URL 和标题
            line_lower = line.lower()
            if line_lower.startswith('url:'):
                snapshot.url = line.split(':', 1)[-1].strip()
            elif line_lower.startswith('title:'):
                snapshot.title = line.split(':', 1)[-1].strip()
            
            # 提取元素
            if 'ref=' in line or 'ref:' in line:
                el = self._parse_element(line)
                if el:
                    elements.append(el)
        
        snapshot.elements = elements
        return snapshot
    
    def _parse_element(self, line: str) -> Optional[ElementInfo]:
        """解析单个元素"""
        try:
            # 提取 ref
            ref_match = re.search(r'ref[=:][\s]*["\']?([^"\'\s\]>]+)', line)
            if not ref_match:
                return None
            
            ref = ref_match.group(1)
            el = ElementInfo(ref=ref)
            
            # 提取标签
            tag_match = re.search(r'<(\w+)', line)
            if tag_match:
                el.tag = tag_match.group(1).lower()
            
            # 提取文本 - 多种模式匹配
            text_patterns = [
                r'["\']([^"\']{1,100})["\']',
                r'>([^<]{1,100})<',
                r'text[=:][\s]*["\']?([^"\'<>\]]{1,100})',
            ]
            for pattern in text_patterns:
                match = re.search(pattern, line)
                if match and match.group(1).strip():
                    el.text = match.group(1).strip()
                    break
            
            # 提取 role
            role_match = re.search(r'role[=:][\s]*["\']?(\w+)', line)
            if role_match:
                el.role = role_match.group(1)
            
            # 提取常见属性
            for attr in ['placeholder', 'aria-label', 'name', 'id', 'class', 'type', 'value', 'href']:
                attr_match = re.search(rf'{attr}[=:][\s]*["\']?([^"\'<>\]\s]+)', line, re.IGNORECASE)
                if attr_match:
                    el.attributes[attr] = attr_match.group(1)
            
            return el
            
        except Exception:
            return None
    
    def _detect_state(self, snapshot: PageSnapshot) -> PageState:
        """检测页面状态"""
        content_lower = snapshot.content.lower()
        
        # 检测加载状态
        loading_indicators = ['loading', '加载中', 'please wait', '请稍候', 'spinner', '正在加载']
        if any(ind in content_lower for ind in loading_indicators):
            return PageState.LOADING
        
        # 检测错误状态
        error_indicators = ['404', '500', '502', '503', 'not found', '页面不存在', '无法访问']
        if any(ind in content_lower for ind in error_indicators):
            return PageState.ERROR
        
        # 检测阻挡状态
        blocking_indicators = [
            ('cookie', ['accept', '接受', '同意', 'agree']),
            ('登录', ['login', '登录', '注册']),
            ('弹窗', ['close', '关闭', '×', 'x']),
            ('modal', ['close', 'dismiss', '关闭']),
            ('dialog', ['close', 'ok', '确定']),
        ]
        
        for blocker, indicators in blocking_indicators:
            if blocker in content_lower:
                if any(ind in content_lower for ind in indicators):
                    self.blocked_by = blocker
                    return PageState.BLOCKED
        
        # 有元素说明可交互
        if snapshot.elements:
            return PageState.INTERACTIVE
        
        # 有足够内容说明就绪
        if len(snapshot.content) > 100:
            return PageState.READY
        
        return PageState.UNKNOWN
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        parts = []
        
        if self.current_snapshot:
            s = self.current_snapshot
            parts.append(f"📍 URL: {s.url or '未知'}")
            parts.append(f"📄 标题: {s.title or '未知'}")
            parts.append(f"🎯 元素数: {len(s.elements)}")
            parts.append(f"📊 状态: {self.page_state.name}")
            
            if s.is_stale(15):
                parts.append("⚠️ 快照可能已过期")
        else:
            parts.append("📍 尚未获取页面快照")
        
        if self.blocked_by:
            parts.append(f"🚫 被阻挡: {self.blocked_by}")
        
        return "\n".join(parts)
    
    def suggest_action(self) -> Optional[str]:
        """根据当前状态建议操作"""
        if self.page_state == PageState.BLOCKED:
            return f"页面被 {self.blocked_by} 阻挡，建议先关闭（查找关闭按钮或按 Escape）"
        
        if self.page_state == PageState.LOADING:
            return "页面正在加载，建议等待后重新获取快照"
        
        if self.page_state == PageState.ERROR:
            return "页面出现错误，建议检查 URL 或后退重试"
        
        if not self.current_snapshot or self.current_snapshot.is_stale(15):
            return "建议先执行 browser_snapshot 获取页面状态"
        
        return None
    
    def reset(self):
        """重置状态"""
        self.current_snapshot = None
        self.snapshot_history.clear()
        self.page_state = PageState.UNKNOWN
        self.navigation_stack.clear()
        self.blocked_by = None


# ==================== 操作执行器 ====================

@dataclass
class ActionResult:
    """操作结果"""
    success: bool
    output: str
    duration: float = 0.0
    retries: int = 0
    error: Optional[str] = None


class ActionExecutor:
    """智能操作执行器"""
    
    def __init__(self, session: ClientSession, logger: Logger):
        self.session = session
        self.logger = logger
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "retries": 0,
            "total_time": 0.0
        }
    
    async def execute(self, name: str, args: dict, retry: bool = True) -> ActionResult:
        """
        执行操作
        
        Args:
            name: 工具名称
            args: 参数
            retry: 是否重试
            
        Returns:
            操作结果
        """
        self.stats["total"] += 1
        start_time = datetime.now()
        retries = 0
        last_error = None
        
        max_retries = Config.MAX_RETRIES if retry else 1
        
        while retries < max_retries:
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(name, args),
                    timeout=Config.TOOL_TIMEOUT
                )
                
                # 解析结果
                output = self._parse_result(result)
                
                # 智能等待
                await self._smart_wait(name, output)
                
                duration = (datetime.now() - start_time).total_seconds()
                self.stats["success"] += 1
                self.stats["total_time"] += duration
                
                return ActionResult(
                    success=True,
                    output=output,
                    duration=duration,
                    retries=retries
                )
                
            except asyncio.TimeoutError:
                last_error = "操作超时"
                self.logger.warn(f"{name} 超时，重试 {retries + 1}/{max_retries}")
                
            except Exception as e:
                last_error = str(e)
                
                if not self._is_retryable(e):
                    break
                
                self.logger.warn(f"{name} 失败: {e}，重试 {retries + 1}/{max_retries}")
            
            retries += 1
            self.stats["retries"] += 1
            
            if retries < max_retries:
                await asyncio.sleep(1.0 * retries)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats["failed"] += 1
        self.stats["total_time"] += duration
        
        return ActionResult(
            success=False,
            output="",
            duration=duration,
            retries=retries,
            error=last_error
        )
    
    def _parse_result(self, result) -> str:
        """解析工具返回结果"""
        if not hasattr(result, 'content') or not result.content:
            return str(result) if result else "操作完成"
        
        contents = []
        for item in result.content:
            if hasattr(item, 'text'):
                contents.append(item.text)
            elif hasattr(item, 'data'):
                contents.append(f"[二进制数据: {len(str(item.data))} bytes]")
            else:
                contents.append(str(item))
        
        return "\n".join(contents) if contents else "操作完成"
    
    async def _smart_wait(self, action: str, output: str):
        """智能等待"""
        wait_range = Config.WAIT_TIMES.get(action)
        if not wait_range:
            return
        
        min_wait, max_wait = wait_range
        output_lower = output.lower()
        
        # 根据输出内容调整等待时间
        if any(kw in output_lower for kw in ['loading', 'redirect', '跳转', '加载']):
            wait_time = max_wait
        elif any(kw in output_lower for kw in ['error', 'failed', '失败']):
            wait_time = min_wait
        elif len(output) > 5000:
            wait_time = max_wait * 0.8
        else:
            wait_time = min_wait + (max_wait - min_wait) * 0.3
        
        await asyncio.sleep(wait_time)
    
    def _is_retryable(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()
        
        retryable = ['timeout', 'connection', 'network', 'temporary', 'retry']
        if any(kw in error_str for kw in retryable):
            return True
        
        non_retryable = ['invalid', 'not found', 'permission', 'auth']
        if any(kw in error_str for kw in non_retryable):
            return False
        
        return True
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self.stats["total"]
        return {
            **self.stats,
            "success_rate": f"{self.stats['success']/total*100:.1f}%" if total > 0 else "N/A",
            "avg_time": f"{self.stats['total_time']/total:.2f}s" if total > 0 else "N/A"
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "retries": 0,
            "total_time": 0.0
        }


# ==================== 恢复策略 ====================

class RecoveryStrategy:
    """错误恢复策略管理器"""
    
    def __init__(self, executor: ActionExecutor, state_manager: BrowserStateManager, logger: Logger):
        self.executor = executor
        self.state = state_manager
        self.logger = logger
        self.recovery_count = 0
    
    async def try_recover(self, error_context: str = "") -> Optional[str]:
        """
        尝试恢复
        
        Args:
            error_context: 错误上下文
            
        Returns:
            恢复后的快照内容，或 None
        """
        self.recovery_count += 1
        self.logger.info(f"开始恢复流程 (第 {self.recovery_count} 次)")
        
        # 按优先级尝试恢复策略
        strategies = [
            ("关闭弹窗", self._try_close_popup),
            ("刷新快照", self._try_refresh_snapshot),
            ("滚动页面", self._try_scroll_and_snapshot),
            ("后退重试", self._try_go_back),
            ("刷新页面", self._try_refresh_page),
        ]
        
        for name, strategy in strategies:
            result = await strategy()
            if result:
                self.logger.info(f"恢复成功: {name}")
                return result
        
        self.logger.warn("所有恢复策略均失败")
        return None
    
    async def _try_close_popup(self) -> Optional[str]:
        """尝试关闭弹窗"""
        # 按 Escape
        await self.executor.execute('browser_press_key', {'key': 'Escape'}, retry=False)
        await asyncio.sleep(0.5)
        
        # 获取快照检查
        result = await self.executor.execute('browser_snapshot', {}, retry=False)
        if result.success:
            snapshot = PageSnapshot(content=result.output)
            new_state = self.state._detect_state(snapshot)
            if new_state != PageState.BLOCKED:
                return result.output
        
        return None
    
    async def _try_refresh_snapshot(self) -> Optional[str]:
        """刷新快照"""
        await asyncio.sleep(1.0)
        result = await self.executor.execute('browser_snapshot', {})
        if result.success and len(result.output) > 100:
            return result.output
        return None
    
    async def _try_scroll_and_snapshot(self) -> Optional[str]:
        """滚动后获取快照"""
        # 滚动到顶部
        await self.executor.execute('browser_press_key', {'key': 'Home'}, retry=False)
        await asyncio.sleep(0.5)
        
        result = await self.executor.execute('browser_snapshot', {})
        if result.success:
            return result.output
        return None
    
    async def _try_go_back(self) -> Optional[str]:
        """后退重试"""
        if len(self.state.navigation_stack) < 2:
            return None
        
        await self.executor.execute('browser_go_back', {}, retry=False)
        await asyncio.sleep(1.5)
        
        result = await self.executor.execute('browser_snapshot', {})
        if result.success:
            return result.output
        return None
    
    async def _try_refresh_page(self) -> Optional[str]:
        """刷新页面"""
        if not self.state.current_snapshot or not self.state.current_snapshot.url:
            return None
        
        url = self.state.current_snapshot.url
        await self.executor.execute('browser_navigate', {'url': url}, retry=False)
        await asyncio.sleep(2.0)
        
        result = await self.executor.execute('browser_snapshot', {})
        if result.success:
            return result.output
        return None
    
    def reset(self):
        """重置恢复计数"""
        self.recovery_count = 0


# ==================== 循环检测器 ====================

class LoopDetector:
    """操作循环检测器"""
    
    def __init__(self, window_size: int = 15):
        self.actions: deque = deque(maxlen=window_size * 2)
        self.window_size = window_size
    
    def record(self, action: str, args_hash: str = ""):
        """记录操作"""
        self.actions.append(f"{action}:{args_hash}")
    
    def detect(self) -> Optional[str]:
        """
        检测循环模式
        
        Returns:
            循环描述或 None
        """
        if len(self.actions) < 6:
            return None
        
        recent = list(self.actions)
        
        # 1. 简单重复检测 (AAAAAA)
        if len(recent) >= 6:
            last_6 = recent[-6:]
            if len(set(last_6)) == 1:
                action_name = last_6[0].split(':')[0]
                return f"连续重复: {action_name}"
        
        # 2. 模式循环检测 (ABABAB, ABCABC)
        for pattern_len in [2, 3, 4]:
            if len(recent) >= pattern_len * 3:
                pattern = recent[-pattern_len:]
                is_loop = True
                for i in range(2):
                    start = -(pattern_len * (i + 2))
                    end = -(pattern_len * (i + 1))
                    if recent[start:end] != pattern:
                        is_loop = False
                        break
                if is_loop:
                    actions = [a.split(':')[0] for a in pattern]
                    return f"循环模式: {' → '.join(actions)}"
        
        # 3. 高频操作检测
        if len(recent) >= 10:
            action_counts = {}
            for a in recent[-10:]:
                action = a.split(':')[0]
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for action, count in action_counts.items():
                if count >= 8 and action != 'browser_snapshot':
                    return f"高频操作: {action} ({count}/10)"
        
        return None
    
    def clear(self):
        """清空记录"""
        self.actions.clear()


# ==================== 主代理类 ====================

class PlaywrightMCPAgent:
    """Playwright MCP 智能浏览器代理"""
    
    def __init__(self, api_key: str):
        # LLM 客户端
        self.llm = OpenAI(api_key=api_key, base_url=Config.DEEPSEEK_BASE_URL)
        self.model = Config.MODEL
        
        # MCP 会话
        self.session: Optional[ClientSession] = None
        self.tools_schema: list[dict] = []
        self._stdio_context = None
        self._session_context = None
        
        # 核心组件
        self.logger = Logger("Agent", LogLevel.INFO)
        self.state = BrowserStateManager()
        self.executor: Optional[ActionExecutor] = None
        self.recovery: Optional[RecoveryStrategy] = None
        self.loop_detector = LoopDetector()
        
        # 对话历史
        self.conversation: list[dict] = []
        
        # 统计信息
        self.stats = {
            "sessions": 0,
            "api_calls": 0,
            "start_time": None
        }
    
    async def connect(self):
        """连接 MCP 服务器"""
        self.logger.info("正在启动 Playwright MCP...")
        
        server_params = StdioServerParameters(
            command="npx",
            args=["@playwright/mcp@latest"],
            env={**os.environ, "NODE_ENV": "production"}
        )
        
        try:
            self._stdio_context = stdio_client(server_params)
            self._streams = await self._stdio_context.__aenter__()
            read, write = self._streams
            
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()
            
            await asyncio.wait_for(
                self.session.initialize(),
                timeout=Config.CONNECT_TIMEOUT
            )
            
            # 初始化组件
            self.executor = ActionExecutor(self.session, self.logger)
            self.recovery = RecoveryStrategy(self.executor, self.state, self.logger)
            
            # 加载工具
            await self._load_tools()
            
            self.stats["sessions"] += 1
            self.stats["start_time"] = datetime.now()
            
            self.logger.info(f"连接成功！已加载 {len(self.tools_schema)} 个工具")
            
        except asyncio.TimeoutError:
            self.logger.error("连接超时")
            self.logger.error("请确保已安装: npx playwright install chromium")
            raise
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            pass
        self.logger.info("已断开连接")
    
    async def _load_tools(self):
        """加载 MCP 工具"""
        result = await self.session.list_tools()
        
        self.tools_schema = []
        for tool in result.tools:
            schema = tool.inputSchema or {
                "type": "object",
                "properties": {},
                "required": []
            }
            self.tools_schema.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"执行 {tool.name}",
                    "parameters": schema
                }
            })
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的浏览器自动化助手，使用 Playwright MCP 工具精确操作浏览器。

## 🎯 核心原则

### 1. 先观察，后行动
- 任何操作前**必须**先用 `browser_snapshot` 获取页面状态
- 快照返回的 `ref` 是元素的唯一标识，操作元素必须使用 ref
- 页面变化后 ref 会失效，必须重新获取快照

### 2. 精确操作
- `browser_click` 的 `element` 参数必须是快照中的 ref 值
- `browser_type` 前通常需要先点击输入框获得焦点
- 搜索操作：输入关键词后，点击搜索按钮或按 Enter 键

### 3. 验证结果
- 每次操作后通过快照确认是否成功
- 如果操作无效，分析原因并尝试其他方法
- 不要连续重复相同的失败操作

## 📋 标准流程

### 导航

browser_navigate → 目标 URL
browser_snapshot → 确认加载完成


### 点击
browser_snapshot → 获取最新状态
在快照中找到目标元素的 ref
browser_click → element: "找到的ref"
browser_snapshot → 验证效果


### 输入
browser_snapshot → 找到输入框
browser_click → 点击输入框
browser_type → element: "ref", text: "内容"
browser_snapshot → 确认输入


### 搜索
找到搜索框 → 点击 → 输入关键词
找到搜索按钮点击，或 browser_press_key → key: "Enter"
browser_snapshot → 查看结果


## ⚠️ 常见问题处理

| 问题 | 解决方案 |
|-----|---------|
| 元素不在视口 | `browser_scroll_down` 或 `browser_scroll_up` |
| 有弹窗遮挡 | 找关闭按钮点击，或 `browser_press_key` → "Escape" |
| 点击无效 | 重新获取快照，确认 ref 正确；检查是否有遮挡 |
| 输入丢失 | 先点击输入框，再输入；或设置 clear: true |
| 页面未加载 | 等待后重新获取快照 |

## 🔧 工具速查

- `browser_navigate`: 导航到 URL
- `browser_snapshot`: 获取页面状态和元素 ref（最重要！）
- `browser_click`: 点击元素，需要 element 参数
- `browser_type`: 输入文本，需要 element 和 text 参数
- `browser_press_key`: 按键，如 "Enter", "Escape", "Tab"
- `browser_scroll_down/up`: 滚动页面
- `browser_go_back/forward`: 前进/后退
- `browser_select_option`: 下拉选择

## 📝 回复规范
1. 简述当前步骤意图
2. 执行操作
3. 根据结果决定下一步
4. 任务完成后总结结果"""

    async def chat(self, user_message: str) -> str:
        """
        处理用户消息
        
        Args:
            user_message: 用户输入
            
        Returns:
            助手回复
        """
        self.conversation.append({"role": "user", "content": user_message})
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation
        ]
        
        # 添加状态上下文
        context = self.state.get_context_summary()
        suggestion = self.state.suggest_action()
        
        if context or suggestion:
            context_msg = "[当前状态]\n" + context
            if suggestion:
                context_msg += f"\n\n💡 建议: {suggestion}"
            messages.append({"role": "system", "content": context_msg})
        
        iteration = 0
        
        while iteration < Config.MAX_ITERATIONS:
            iteration += 1
            self.stats["api_calls"] += 1
            
            # 调用 LLM
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm.chat.completions.create,
                        model=self.model,
                        messages=messages,
                        tools=self.tools_schema if self.tools_schema else None,
                        tool_choice="auto"
                    ),
                    timeout=Config.API_TIMEOUT
                )
            except asyncio.TimeoutError:
                self.logger.warn("API 调用超时，重试...")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['rate', '429', '503']):
                    self.logger.warn("API 限流，等待重试...")
                    await asyncio.sleep(5)
                    continue
                return f"❌ API 错误: {e}"
            
            assistant_msg = response.choices[0].message
            
            # 处理工具调用
            if assistant_msg.tool_calls:
                # 记录助手消息
                msg_record = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                }
                
                # 保存推理内容
                if hasattr(assistant_msg, "reasoning_content") and assistant_msg.reasoning_content:
                    msg_record["reasoning_content"] = assistant_msg.reasoning_content
                
                messages.append(msg_record)
                
                # 执行每个工具调用
                for tc in assistant_msg.tool_calls:
                    func_name = tc.function.name
                    
                    try:
                        func_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}
                    
                    # 记录操作用于循环检测
                    args_hash = hashlib.md5(
                        json.dumps(func_args, sort_keys=True).encode()
                    ).hexdigest()[:6]
                    self.loop_detector.record(func_name, args_hash)
                    
                    # 检测循环
                    loop_issue = self.loop_detector.detect()
                    if loop_issue:
                        self.logger.warn(f"检测到: {loop_issue}")
                        
                        # 尝试恢复
                        recovery_result = await self.recovery.try_recover(loop_issue)
                        if recovery_result:
                            self.state.update_snapshot(recovery_result)
                            self.loop_detector.clear()
                            messages.append({
                                "role": "system",
                                "content": f"⚠️ 检测到{loop_issue}，已自动恢复。\n\n[当前页面状态]\n{recovery_result[:4000]}"
                            })
                            continue
                        else:
                            stuck_msg = f"⚠️ 操作陷入循环 ({loop_issue})，自动恢复失败。请尝试其他方法。"
                            self.conversation.append({"role": "assistant", "content": stuck_msg})
                            return stuck_msg
                    
                    # 显示执行信息
                    args_preview = json.dumps(func_args, ensure_ascii=False)
                    if len(args_preview) > 60:
                        args_preview = args_preview[:60] + "..."
                    self.logger.info(f"[{iteration}] {func_name}: {args_preview}")
                    
                    # 执行操作
                    result = await self.executor.execute(func_name, func_args)
                    
                    # 更新状态
                    if func_name == 'browser_snapshot' and result.success:
                        self.state.update_snapshot(result.output)
                    
                    # 处理失败
                    if not result.success:
                        self.logger.warn(f"操作失败: {result.error}")
                        
                        # 连续失败时尝试恢复
                        if self.state.page_state == PageState.BLOCKED or result.retries >= 2:
                            recovery_result = await self.recovery.try_recover(result.error)
                            if recovery_result:
                                self.state.update_snapshot(recovery_result)
                                messages.append({
                                    "role": "system",
                                    "content": f"[自动恢复后的页面状态]\n{recovery_result[:3000]}"
                                })
                    
                    # 截断过长输出
                    output = result.output
                    if len(output) > 8000:
                        output = output[:8000] + "\n...[内容已截断]"
                    
                    # 添加工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output if result.success else f"❌ {result.error}\n\n{output}"
                    })
                    
                    # 关键操作后自动验证
                    verify_actions = {'browser_click', 'browser_type', 'browser_navigate', 'browser_press_key'}
                    if result.success and func_name in verify_actions and func_name != 'browser_snapshot':
                        self.logger.info("[自动验证]")
                        verify = await self.executor.execute('browser_snapshot', {})
                        if verify.success:
                            self.state.update_snapshot(verify.output)
                            verify_output = verify.output
                            if len(verify_output) > 4000:
                                verify_output = verify_output[:4000] + "\n...[已截断]"
                            messages.append({
                                "role": "system",
                                "content": f"[操作后页面状态]\n{verify_output}"
                            })
            
            else:
                # 无工具调用，返回最终结果
                final = assistant_msg.content or "任务完成"
                
                msg_record = {"role": "assistant", "content": final}
                if hasattr(assistant_msg, "reasoning_content") and assistant_msg.reasoning_content:
                    msg_record["reasoning_content"] = assistant_msg.reasoning_content
                messages.append(msg_record)
                
                self.conversation.append({"role": "assistant", "content": final})
                return final
        
        timeout_msg = f"⚠️ 达到最大迭代次数 ({Config.MAX_ITERATIONS})"
        self.conversation.append({"role": "assistant", "content": timeout_msg})
        return timeout_msg
    
    def clear(self):
        """清空所有状态"""
        self.conversation.clear()
        self.state.reset()
        self.loop_detector.clear()
        if self.executor:
            self.executor.reset_stats()
        if self.recovery:
            self.recovery.reset()
        self.logger.info("已清空所有状态")
    
    def status(self) -> str:
        """获取状态报告"""
        lines = [
            "=" * 55,
            "📊 状态报告",
            "=" * 55,
            "",
            "🌐 浏览器状态:",
            self.state.get_context_summary(),
            "",
        ]
        
        if self.executor:
            stats = self.executor.get_stats()
            lines.extend([
                "📈 操作统计:",
                f"   总操作: {stats['total']}",
                f"   成功率: {stats['success_rate']}",
                f"   重试次数: {stats['retries']}",
                f"   平均耗时: {stats['avg_time']}",
                "",
            ])
        
        if self.recovery:
            lines.append(f"🔄 恢复次数: {self.recovery.recovery_count}")
        
        lines.extend([
            f"💬 对话轮数: {len(self.conversation) // 2}",
            f"🤖 API 调用: {self.stats['api_calls']}",
        ])
        
        if self.stats["start_time"]:
            duration = datetime.now() - self.stats["start_time"]
            minutes = duration.seconds // 60
            seconds = duration.seconds % 60
            lines.append(f"⏱️ 运行时长: {minutes}分{seconds}秒")
        
        lines.append("=" * 55)
        return "\n".join(lines)
    
    def save_session(self):
        """保存会话"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.conversation,
            "url": self.state.current_snapshot.url if self.state.current_snapshot else "",
            "stats": {
                "api_calls": self.stats["api_calls"],
                "recovery_count": self.recovery.recovery_count if self.recovery else 0
            }
        }
        
        with open(Config.SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"会话已保存到 {Config.SESSION_FILE}")
    
    def load_session(self) -> bool:
        """加载会话"""
        if not Config.SESSION_FILE.exists():
            return False
        
        try:
            with open(Config.SESSION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation = data.get("conversation", [])
            self.logger.info(f"已加载会话 ({len(self.conversation)} 条消息)")
            return True
            
        except Exception as e:
            self.logger.warn(f"加载会话失败: {e}")
            return False


# ==================== 主程序入口 ====================

async def main():
    """主程序"""
    
    # 获取 API Key
    api_key = Config.get_api_key()
    
    if not api_key:
        print("=" * 60)
        print("  🔑 首次运行，请输入 DeepSeek API Key")
        print("=" * 60)
        api_key = input("API Key: ").strip()
        if not api_key:
            print("❌ 未输入 API Key，退出")
            return
        Config.set_api_key(api_key)
        print("✅ 已保存\n")
    
    # 创建代理
    agent = PlaywrightMCPAgent(api_key)
    
    try:
        # 连接
        await agent.connect()
        
        # 尝试加载会话
        if Config.SESSION_FILE.exists():
            choice = input("\n检测到上次会话，是否加载？[y/N]: ").strip().lower()
            if choice in ('y', 'yes'):
                agent.load_session()
        
        # 显示欢迎信息
        print("\n" + "=" * 60)
        print("  🌐 Playwright 浏览器智能助手")
        print("  🧠 DeepSeek Reasoner | 终极版")
        print("=" * 60)
        print("  示例任务:")
        print("  • 打开百度搜索今天的新闻")
        print("  • 访问 github.com 搜索 python")
        print("  • 打开知乎搜索人工智能")
        print()
        print("  命令: q=退出 | c=清空 | s=状态 | save=保存 | h=帮助")
        print("=" * 60)
        
        # 主循环
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            # 退出
            if cmd in ('q', 'quit', 'exit'):
                choice = input("保存会话？[y/N]: ").strip().lower()
                if choice in ('y', 'yes'):
                    agent.save_session()
                break
            
            # 清空
            if cmd in ('c', 'clear'):
                agent.clear()
                continue
            
            # 状态
            if cmd in ('s', 'status'):
                print(agent.status())
                continue
            
            # 保存
            if cmd == 'save':
                agent.save_session()
                continue
            
            # 帮助
            if cmd in ('h', 'help'):
                print("""
📖 使用帮助
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
直接输入任务描述，AI 会自动操作浏览器完成任务。

💡 提高成功率的技巧:
  • 任务描述尽量具体明确
  • 复杂任务可以分步骤描述
  • 遇到问题可以说"重新获取页面状态"
  • 如果卡住，可以说"换一种方法试试"

🔧 可用命令:
  q / quit   - 退出程序
  c / clear  - 清空对话历史和状态
  s / status - 查看详细状态报告
  save       - 保存当前会话
  h / help   - 显示此帮助

🌐 支持的操作:
  • 导航到网页
  • 点击按钮/链接
  • 输入文本
  • 滚动页面
  • 前进/后退
  • 处理下拉选择
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
                continue
            
            # 处理用户消息
            print("\n🤖 思考中...\n")
            response = await agent.chat(user_input)
            print(f"\n🤖 助手: {response}")
    
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.disconnect()


