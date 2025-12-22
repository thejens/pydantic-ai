"""AG-UI adapter for handling requests."""

from __future__ import annotations

from base64 import b64decode
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from uuid import uuid4

from ... import ExternalToolset, ToolDefinition
from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset

try:
    from ag_ui.core import (
        ActivityMessage,
        AssistantMessage,
        BaseEvent,
        BinaryInputContent,
        DeveloperMessage,
        FunctionCall,
        Message,
        RunAgentInput,
        SystemMessage,
        TextInputContent,
        Tool as AGUITool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from .. import MessagesBuilder, UIAdapter, UIEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    pass

__all__ = ['AGUIAdapter']


# Frontend toolset


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for AG-UI frontend tools."""

    def __init__(self, tools: list[AGUITool]):
        """Initialize the toolset with AG-UI tools.

        Args:
            tools: List of AG-UI tool definitions.
        """
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        """Return the label for this toolset."""
        return 'the AG-UI frontend tools'  # pragma: no cover


class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the AG-UI run input."""
        if self.run_input.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.run_input.tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the AG-UI run input."""
        state = self.run_input.state
        if state is None:
            return None

        if isinstance(state, Mapping) and not state:
            return None

        return cast('dict[str, Any]', state)

    @classmethod
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:  # noqa: C901
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
        for msg in messages:
            match msg:
                case UserMessage(content=content):
                    if isinstance(content, str):
                        builder.add(UserPromptPart(content=content))
                    else:
                        user_prompt_content: list[Any] = []
                        for part in content:
                            match part:
                                case TextInputContent(text=text):
                                    user_prompt_content.append(text)
                                case BinaryInputContent():
                                    if part.url:
                                        try:
                                            binary_part = BinaryContent.from_data_uri(part.url)
                                        except ValueError:
                                            media_type_constructors = {
                                                'image': ImageUrl,
                                                'video': VideoUrl,
                                                'audio': AudioUrl,
                                            }
                                            media_type_prefix = part.mime_type.split('/', 1)[0]
                                            constructor = media_type_constructors.get(media_type_prefix, DocumentUrl)
                                            binary_part = constructor(url=part.url, media_type=part.mime_type)
                                    elif part.data:
                                        binary_part = BinaryContent(
                                            data=b64decode(part.data), media_type=part.mime_type
                                        )
                                    else:  # pragma: no cover
                                        raise ValueError('BinaryInputContent must have either a `url` or `data` field.')
                                    user_prompt_content.append(binary_part)
                                case _:  # pragma: no cover
                                    raise ValueError(f'Unsupported user message part type: {type(part)}')

                        if user_prompt_content:  # pragma: no branch
                            content_to_add = (
                                user_prompt_content[0]
                                if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str)
                                else user_prompt_content
                            )
                            builder.add(UserPromptPart(content=content_to_add))

                case SystemMessage(content=content) | DeveloperMessage(content=content):
                    builder.add(SystemPromptPart(content=content))

                case AssistantMessage(content=content, tool_calls=tool_calls_list):
                    if content:
                        builder.add(TextPart(content=content))
                    if tool_calls_list:
                        for tool_call in tool_calls_list:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, original_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=original_id,
                                        provider_name=provider_name,
                                    )
                                )
                            else:
                                builder.add(
                                    ToolCallPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_call.function.arguments,
                                    )
                                )
                case ToolMessage() as tool_msg:
                    tool_call_id = tool_msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                        _, provider_name, original_id = tool_call_id.split('|', 2)
                        builder.add(
                            BuiltinToolReturnPart(
                                tool_name=tool_name,
                                content=tool_msg.content,
                                tool_call_id=original_id,
                                provider_name=provider_name,
                            )
                        )
                    else:
                        builder.add(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=tool_msg.content,
                                tool_call_id=tool_call_id,
                            )
                        )

                case ActivityMessage():  # pragma: no cover
                    raise ValueError(f'Unsupported message type: {type(msg)}')

        return builder.messages

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        Args:
            messages: A sequence of ModelMessage objects to convert.

        Returns:
            A list of AG-UI Message objects.
        """
        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                cls._dump_request_parts(msg, result)
            elif isinstance(msg, ModelResponse):
                cls._dump_response_parts(msg, result)

        return result

    @classmethod
    def _dump_request_parts(cls, msg: ModelRequest, result: list[Message]) -> None:
        """Convert ModelRequest parts to AG-UI messages."""
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                result.append(SystemMessage(id=str(uuid4()), role='system', content=part.content))
            elif isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                result.append(UserMessage(id=str(uuid4()), role='user', content=content))
            elif isinstance(part, ToolReturnPart):
                result.append(
                    ToolMessage(
                        id=str(uuid4()),
                        role='tool',
                        tool_call_id=part.tool_call_id,
                        content=part.model_response_str(),
                    )
                )
            elif isinstance(part, RetryPromptPart) and part.tool_call_id:
                result.append(
                    ToolMessage(
                        id=str(uuid4()),
                        role='tool',
                        tool_call_id=part.tool_call_id,
                        content=part.model_response(),
                    )
                )

    @classmethod
    def _dump_response_parts(cls, msg: ModelResponse, result: list[Message]) -> None:
        """Convert ModelResponse parts to AG-UI messages."""
        current_content: str = ''
        current_tool_calls: list[ToolCall] = []

        def flush() -> None:
            nonlocal current_content, current_tool_calls
            if current_content or current_tool_calls:
                result.append(
                    AssistantMessage(
                        id=str(uuid4()),
                        role='assistant',
                        content=current_content or None,
                        tool_calls=current_tool_calls or None,
                    )
                )
                current_content = ''
                current_tool_calls = []

        for part in msg.parts:
            if isinstance(part, TextPart):
                if current_tool_calls:
                    flush()
                current_content += part.content
            elif isinstance(part, ThinkingPart):
                pass  # Skip - no AG-UI equivalent
            elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
                if current_content and not current_tool_calls:
                    flush()
                current_tool_calls.append(cls._make_tool_call(part))
            elif isinstance(part, BuiltinToolReturnPart):
                flush()
                prefixed_id = f'{BUILTIN_TOOL_CALL_ID_PREFIX}|{part.provider_name}|{part.tool_call_id}'
                result.append(
                    ToolMessage(id=str(uuid4()), role='tool', tool_call_id=prefixed_id, content=part.model_response_str())
                )
            # FilePart has no direct AG-UI equivalent, skip

        flush()

    @staticmethod
    def _make_tool_call(part: ToolCallPart | BuiltinToolCallPart) -> ToolCall:
        """Create a ToolCall from a ToolCallPart or BuiltinToolCallPart."""
        if isinstance(part, BuiltinToolCallPart):
            tool_call_id = f'{BUILTIN_TOOL_CALL_ID_PREFIX}|{part.provider_name}|{part.tool_call_id}'
        else:
            tool_call_id = part.tool_call_id
        return ToolCall(
            id=tool_call_id,
            type='function',
            function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
        )
