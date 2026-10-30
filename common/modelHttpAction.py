# -*- coding: utf-8 -*-
"""
# @File    : modelHttpAction.py
# @Date    : 2026-10-26
# @Author  : jeremyzhai
# @Description  : 纯外部模型调用工具（HTTP Client）
#
# 职责：提供统一的模型 HTTP 调用能力，不涉及任何业务逻辑。
# 通过 provider 参数切换不同模型服务，默认使用 Kimi for Coding（Allegretto 会员）。
#
# 支持两种 API 协议：
#   - anthropic: Anthropic Messages API（/v1/messages）
#                适用于 Kimi for Coding（api.kimi.com/coding）及内部 litellm 网关
#   - openai:    OpenAI Chat Completions API（/v1/chat/completions）
#                适用于 Moonshot 开放平台（api.moonshot.cn）等
#
# 使用方式：
#   from common.modelHttpAction import ModelHttpAction
#
#   # 默认：Kimi for Coding
#   client = ModelHttpAction()
#   response = client.chat("你好")
#
#   # 接入任意自定义模型（dict 预设，如内部 litellm 网关、Moonshot 开放平台）
#   client = ModelHttpAction(provider={
#       'api_type': 'openai',
#       'base_url': 'https://api.moonshot.cn/v1',
#       'model': 'moonshot-v1-8k',
#       'auth_token': 'sk-xxxx',
#   })
"""

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
from utils.handle_log import log

import requests

Base_Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_Path)

# 加载项目根目录 .env 文件中的环境变量（密钥放这里，不进入版本库）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(Base_Path, '.env'), override=False)
except ImportError:
    # 未安装 python-dotenv 时跳过，依赖系统环境变量
    pass


# ============================================================
# 模型服务预设（可通过 provider 参数切换，也可用 dict 自定义）
# ============================================================
# 注意：预设中不存放任何密钥，auth_token 统一从环境变量读取：
#   KIMI_API_KEY > ANTHROPIC_AUTH_KEY > MODEL_AUTH_TOKEN
# 推荐将密钥写入项目根目录的 .env 文件（已加入 .gitignore，不会上传）：
#   KIMI_API_KEY=sk-xxxx
PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    'kimi': {
        'api_type': 'anthropic',
        'base_url': 'https://api.kimi.com/coding',
        'model': 'kimi-for-coding',
        'timeout': 600,  # 与 API_TIMEOUT_MS=600000 对齐
    }
}

DEFAULT_PROVIDER = 'kimi'
DEFAULT_TIMEOUT = 240

# 需要重试的 HTTP 状态码
RETRYABLE_STATUS_CODES = (429, 502, 503, 504)

# 用于区分"未传入"与"显式传 None"的哨兵值
_UNSET = object()


class ModelHttpAction:
    """
    外部模型 HTTP 调用客户端

    通过 provider 参数切换模型服务：
    - 内置预设名：'kimi'（默认）
    - 自定义 dict：{'api_type': ..., 'base_url': ..., 'model': ..., 'auth_token': ...}
      用于接入内部网关、Moonshot 开放平台等任意服务

    参数优先级（高 → 低）：
      初始化显式参数 > 环境变量 > provider 预设

    相关环境变量：
      MODEL_NAME / MODEL_BASE_URL / MODEL_AUTH_TOKEN
      KIMI_API_KEY / ANTHROPIC_AUTH_KEY（Kimi 凭据）
    """

    def __init__(
        self,
        provider: Any = DEFAULT_PROVIDER,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        auth_token: Any = _UNSET,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        thinking_type: Optional[str] = 'adaptive',
    ):
        """
        初始化模型 HTTP 客户端

        Args:
            provider: 预设名称（内置仅 'kimi'）或自定义配置 dict
            model_name: 模型名称，缺省用预设的模型
            api_type: API 协议类型，'anthropic' 或 'openai'，缺省用预设的协议
            auth_token: 认证 Token；不传时按环境变量 → 预设的顺序取值
            base_url: 模型接口域名，缺省用预设的地址
            timeout: 请求超时时间（秒），缺省用预设的超时
            thinking_type: 实例级默认思考模式（仅 anthropic 协议生效）；
                           传 None 则请求中不带 thinking 字段
        """
        # 1. 解析预设
        if isinstance(provider, dict):
            preset = dict(provider)
            self.provider = preset.get('name', 'custom')
        else:
            if provider not in PROVIDER_PRESETS:
                raise ValueError(
                    f"未知 provider: {provider!r}，内置预设: {list(PROVIDER_PRESETS.keys())}，"
                    "或传入自定义配置 dict"
                )
            preset = PROVIDER_PRESETS[provider]
            self.provider = provider

        # 2. 各参数：显式传入 > 环境变量 > 预设
        self.model_name = model_name or os.getenv('MODEL_NAME') or preset['model']
        self.api_type = api_type or preset.get('api_type', 'anthropic')
        if self.api_type not in ('anthropic', 'openai'):
            raise ValueError(f"不支持的 api_type: {self.api_type}，仅支持 'anthropic' / 'openai'")

        if auth_token is _UNSET:
            auth_token = (
                os.getenv('KIMI_API_KEY')
                or os.getenv('ANTHROPIC_AUTH_KEY')
                or os.getenv('MODEL_AUTH_TOKEN')
                or preset.get('auth_token')
            )
        self.auth_token = auth_token

        self.base_url = (
            base_url
            or os.getenv('MODEL_BASE_URL')
            or preset['base_url']
        ).rstrip('/')
        self.timeout = timeout or preset.get('timeout', DEFAULT_TIMEOUT)

        # 3. 按协议组装请求地址与请求头
        if self.api_type == 'anthropic':
            self.url = f'{self.base_url}/v1/messages'
            self.headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json',
            }
        else:  # openai
            self.url = f'{self.base_url}/chat/completions'
            self.headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json',
            }

        # 实例级默认思考模式；传 None 则请求中不带 thinking 字段
        self._default_thinking_type = thinking_type

    # --------------------------------------------------------
    # 协议相关的钩子方法（子类可按需覆写）
    # --------------------------------------------------------
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float],
        thinking_type: Optional[str],
        json_schema: Optional[Dict],
    ) -> Dict[str, Any]:
        """根据协议类型构建请求体."""
        if self.api_type == 'anthropic':
            data: Dict[str, Any] = {
                'model': self.model_name,
                'max_tokens': max_tokens,
                'cache_control': {'type': 'ephemeral'},
                'messages': messages,
            }
            # thinking 为 None 时不传该字段（部分 Anthropic 兼容网关不支持）
            if thinking_type is not None:
                data['thinking'] = {'type': thinking_type}
            if temperature is not None:
                data['temperature'] = temperature
            if json_schema:
                data['output_config'] = {
                    'effort': 'low',
                    'format': {
                        'type': 'json_schema',
                        'schema': json_schema,
                    },
                }
            return data

        # openai 协议（Moonshot / OpenAI 兼容）
        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens,
        }
        if temperature is not None:
            data['temperature'] = temperature
        if json_schema:
            data['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'structured_output',
                    'schema': json_schema,
                },
            }
        return data

    def _parse_response(self, resp: Dict[str, Any], desc: str) -> str:
        """从响应 JSON 中提取文本内容."""
        if self.api_type == 'anthropic':
            text = ''
            for block in resp.get('content', []):
                if block.get('type') == 'text':
                    text = block.get('text', '')
                    break
            return text

        # openai 协议
        choices = resp.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '')
            if isinstance(content, list):  # 部分实现返回 content block 列表
                return ''.join(
                    block.get('text', '') for block in content
                    if isinstance(block, dict) and block.get('type') == 'text'
                )
            return content or ''
        return ''

    def _extract_usage(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """提取 token 用量信息（用于调试日志）."""
        usage = resp.get('usage') or {}
        return {
            k: usage[k]
            for k in ('input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens')
            if k in usage
        }

    # --------------------------------------------------------
    # 公共方法
    # --------------------------------------------------------
    def _check_auth(self) -> None:
        """校验认证信息."""
        if not self.auth_token:
            raise ValueError(
                '未找到模型认证 Token，请设置环境变量 KIMI_API_KEY（或 MODEL_AUTH_TOKEN），'
                '或在初始化时传入 auth_token 参数'
            )

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        thinking_type: Any = _UNSET,
        json_schema: Optional[Dict] = None,
        desc: str = '【模型调用】',
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> str:
        """
        发送单条用户消息并获取模型回复

        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词（可选）
            max_tokens: 最大生成 token 数
            temperature: 温度参数（0-1）
            thinking_type: 思考模式，默认取实例级配置（'adaptive'）；
                           传 None 表示不传该字段（仅 anthropic 协议生效）
            json_schema: 结构化输出 JSON Schema（可选）
            desc: 日志描述前缀
            max_retries: 最大重试次数（默认3次）
            retry_backoff: 重试退避基数（默认2.0秒）

        Returns:
            模型返回的文本内容

        Example（仅说明，不可作为 doctest 执行）:
            client = ModelHttpAction()  # 默认 kimi
            response = client.chat("请总结这段文本：...")
            print(response)
        """
        self._check_auth()

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        return self.chat_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            json_schema=json_schema,
            desc=desc,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        thinking_type: Any = _UNSET,
        json_schema: Optional[Dict] = None,
        desc: str = '【模型调用】',
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> str:
        """
        发送多轮对话消息并获取模型回复

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}, ...]
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            thinking_type: 思考模式，默认取实例级配置（'adaptive'）；
                           传 None 表示不传该字段（仅 anthropic 协议生效）
            json_schema: 结构化输出 JSON Schema
            desc: 日志描述前缀
            max_retries: 最大重试次数（默认3次）
            retry_backoff: 重试退避基数（默认2.0秒）

        Returns:
            模型返回的文本内容

        Example（仅说明，不可作为 doctest 执行）:
            client = ModelHttpAction()  # 默认 kimi
            messages = [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "你好"},
            ]
            response = client.chat_messages(messages)
        """
        self._check_auth()
        if thinking_type is _UNSET:
            thinking_type = self._default_thinking_type

        data = self._build_payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            json_schema=json_schema,
        )

        log.info(f"{desc}请求入参: provider={self.provider}, model={self.model_name}, api_type={self.api_type}, messages_count={len(messages)}")

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                    timeout=self.timeout,
                    verify=True,
                )
                log.info(f"{desc}响应状态: {response.status_code}")

                if response.status_code == 200:
                    resp = json.loads(response.text)
                    usage = self._extract_usage(resp)
                    if usage:
                        log.info(f"{desc}token 用量: {usage}")

                    text = self._parse_response(resp, desc)

                    if not text:
                        raise RuntimeError(f'{desc}模型返回内容为空，响应: {resp}')

                    return text

                # 非200状态码处理
                if response.status_code in RETRYABLE_STATUS_CODES:
                    # 需要重试的状态码
                    if attempt < max_retries:
                        wait_time = retry_backoff ** attempt
                        log.warning(
                            f"{desc}请求失败，状态码: {response.status_code}，"
                            f"{wait_time}秒后重试（第{attempt + 1}/{max_retries}次）"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(
                            f'{desc}请求失败，状态码: {response.status_code}，响应: {response.text}')
                else:
                    raise RuntimeError(
                        f'{desc}请求失败，状态码: {response.status_code}，响应: {response.text}')

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = retry_backoff ** attempt
                    log.warning(
                        f"{desc}请求异常，{wait_time}秒后重试（第{attempt + 1}/{max_retries}次）: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f'{desc}请求异常: {traceback.format_exc()}')
                    raise RuntimeError(f'{desc}请求异常（已重试{max_retries}次）: {e}') from e

        # 理论上不会到达这里，但保险起见
        if last_exception:
            raise RuntimeError(f'{desc}请求最终失败: {last_exception}')
        raise RuntimeError(f'{desc}请求未知错误')

    def chat_with_json_output(
        self,
        prompt: str,
        *,
        json_schema: Dict,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        desc: str = '【模型结构化调用】',
    ) -> Dict:
        """
        调用模型并强制返回结构化 JSON

        Args:
            prompt: 用户提示词
            json_schema: JSON Schema 定义输出结构
            system_prompt: 系统提示词
            max_tokens: 最大 token 数
            temperature: 温度参数
            desc: 日志描述

        Returns:
            解析后的 JSON 字典
        """
        text = self.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=json_schema,
            desc=desc,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f'无法解析为 JSON: {text}')

    def __repr__(self) -> str:
        return (
            f"ModelHttpAction(provider='{self.provider}', model_name='{self.model_name}', "
            f"api_type='{self.api_type}', base_url='{self.base_url}')"
        )


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 默认 provider='kimi'，即 Kimi for Coding（Allegretto 会员）
    client = ModelHttpAction()
    print(client.chat("你好，请介绍一下自己", max_tokens=256))
