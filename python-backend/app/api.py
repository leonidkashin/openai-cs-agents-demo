from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
import os

from main import (
    triage_agent,
    faq_agent,
    seat_booking_agent,
    flight_status_agent,
    cancellation_agent,
    create_initial_context,
    tov_guardrail
)

# noinspection PyPackageRequirements
from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    OutputGuardrailResult,
    Handoff,
)

from config.settings import get_settings, LOGGER_NAME

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# Добавим отдельный логгер для "чистого" JSON без префиксов
json_logger = logging.getLogger(f"{LOGGER_NAME}.json")
json_logger.setLevel(logging.INFO)
_json_handler = logging.StreamHandler()
_json_handler.setLevel(logging.INFO)
_json_handler.setFormatter(logging.Formatter("%(message)s"))
json_logger.addHandler(_json_handler)
json_logger.propagate = False

"""Initialize application services"""
logger.info("Initializing application services")
settings = get_settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

app = FastAPI()

# CORS configuration (adjust as needed for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class OutputGuardrailCheck(BaseModel):
    id: str
    name: str
    input_text: str
    output: str
    reasoning: str
    final_text: str
    tripwire_triggered: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []
    output_guardrails: List[OutputGuardrailCheck] = []

# =========================
# In-memory store for conversation state
# =========================

class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

# TODO: when deploying this app in scale, switch to your own production-ready implementation
conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================

def _get_agent_by_name(name: str):
    """Возвращает агента по имени, если такого имени нет — triage_agent."""
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """Получить единый, человеко-читаемый ярлык гардрейла для UI/логов/сравнения."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
            "output_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "output_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(seat_booking_agent),
        make_agent_dict(flight_status_agent),
        make_agent_dict(cancellation_agent),
        make_agent_dict(tov_guardrail)
    ]

def _init_or_get_state(req: "ChatRequest") -> tuple[str, Dict[str, Any], Optional["ChatResponse"]]:
    """
    Создает новое состояние беседы (при необходимости) или возвращает существующее.
    Возвращает (conversation_id, state, early_response_if_any).
    """
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        conv_id: str = uuid4().hex
        ctx = create_initial_context()
        current_agent_name = triage_agent.name
        state: Dict[str, Any] = {
            "input_items": [],
            "context": ctx,
            "current_agent": current_agent_name,
        }
        if req.message.strip() == "":
            conversation_store.save(conv_id, state)
            return conv_id, state, ChatResponse(
                conversation_id=conv_id,
                current_agent=current_agent_name,
                messages=[],
                events=[],
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
                output_guardrails=[],
            )
        return conv_id, state, None
    else:
        conv_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conv_id)  # type: ignore
        return conv_id, state, None

def _run_agent_with_guardrails(conv_id: str, current_agent, state: Dict[str, Any], user_message: str) -> tuple[Optional[Any], List["GuardrailCheck"], Optional["ChatResponse"]]:
    """
    Запускает агента через Runner и обрабатывает срабатывание гардрейлов.
    Возвращает (result_or_none, guardrail_checks, early_response_if_any).
    """
    guardrail_checks: List[GuardrailCheck] = []
    try:
        return Runner.run(current_agent, state["input_items"], context=state["context"]), guardrail_checks, None
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = user_message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        refusal = "Sorry, I can only answer questions related to airline travel."
        state["input_items"].append({"role": "assistant", "content": refusal})
        early = ChatResponse(
            conversation_id=conv_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
            output_guardrails=[],
        )
        return None, guardrail_checks, early

def _parse_tool_args(raw_args: Any) -> Any:
    """Парсит аргументы инструмента: для строк пытается разобрать JSON, иначе возвращает исходное значение без изменений."""
    if isinstance(raw_args, str):
        import json
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return raw_args
    return raw_args

def _handle_message_output(item, messages: List["MessageResponse"], events: List["AgentEvent"]) -> None:
    """Обрабатывает MessageOutputItem: добавляет текст ответа и соответствующее событие."""
    text = ItemHelpers.text_message_output(item)
    messages.append(MessageResponse(content=text, agent=item.agent.name))
    events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))


def _maybe_emit_handoff_callback_event(from_agent, to_agent, events: List["AgentEvent"]) -> None:
    """Если у handoff задан коллбек (on_handoff), добавляет событие tool_call с именем коллбека."""
    ho = next(
        (h for h in getattr(from_agent, "handoffs", [])
         if isinstance(h, Handoff) and getattr(h, "agent_name", None) == to_agent.name),
        None,
    )
    if not ho:
        return
    fn = ho.on_invoke_handoff
    fv = getattr(fn, "__code__", None)
    fv = getattr(fv, "co_freevars", ()) if fv else ()
    cl = getattr(fn, "__closure__", ()) or ()
    if "on_handoff" in fv:
        idx = fv.index("on_handoff")
        if idx < len(cl) and cl[idx].cell_contents:
            cb = cl[idx].cell_contents
            cb_name = getattr(cb, "__name__", repr(cb))
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=to_agent.name,
                    content=cb_name,
                )
            )


def _handle_handoff_output(item, events: List["AgentEvent"]):
    """Обрабатывает HandoffOutputItem: добавляет событие handoff и, при наличии, событие вызова коллбека;
     возвращает нового текущего агента."""
    events.append(
        AgentEvent(
            id=uuid4().hex,
            type="handoff",
            agent=item.source_agent.name,
            content=f"{item.source_agent.name} -> {item.target_agent.name}",
            metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
        )
    )
    from_agent = item.source_agent
    to_agent = item.target_agent
    _maybe_emit_handoff_callback_event(from_agent, to_agent, events)
    return to_agent


def _handle_tool_call_item(item, messages: List["MessageResponse"], events: List["AgentEvent"]) -> None:
    """Обрабатывает ToolCallItem: логирует вызов инструмента, парсит аргументы и при необходимости добавляет служебное сообщение для UI."""
    tool_name = getattr(item.raw_item, "name", None)
    raw_args = getattr(item.raw_item, "arguments", None)
    tool_args: Any = _parse_tool_args(raw_args)
    events.append(
        AgentEvent(
            id=uuid4().hex,
            type="tool_call",
            agent=item.agent.name,
            content=tool_name or "",
            metadata={"tool_args": tool_args},
        )
    )
    if tool_name == "display_seat_map":
        messages.append(
            MessageResponse(
                content="DISPLAY_SEAT_MAP",
                agent=item.agent.name,
            )
        )


def _handle_tool_output_item(item, events: List["AgentEvent"]) -> None:
    """Обрабатывает ToolCallOutputItem: добавляет событие с результатом работы инструмента."""
    events.append(
        AgentEvent(
            id=uuid4().hex,
            type="tool_output",
            agent=item.agent.name,
            content=str(item.output),
            metadata={"tool_result": item.output},
        )
    )


def _process_result_items(result, starting_agent) -> tuple[List["MessageResponse"], List["AgentEvent"], Any]:
    """
    Преобразует result.new_items в (messages, events) и возвращает актуального агента после возможных handoff.
    """
    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []
    current_agent = starting_agent

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            _handle_message_output(item, messages, events)
        elif isinstance(item, HandoffOutputItem):
            current_agent = _handle_handoff_output(item, events)
        elif isinstance(item, ToolCallItem):
            _handle_tool_call_item(item, messages, events)
        elif isinstance(item, ToolCallOutputItem):
            _handle_tool_output_item(item, events)

    return messages, events, current_agent

def _maybe_context_update_event(state: Dict[str, Any], old_context: Dict[str, Any], current_agent) -> Optional["AgentEvent"]:
    """Формирует событие обновления контекста, если текущий контекст отличается от предыдущего; в противном случае возвращает None."""
    new_context = state["context"].dict()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        return AgentEvent(
            id=uuid4().hex,
            type="context_update",
            agent=current_agent.name,
            content="",
            metadata={"changes": changes},
        )
    return None

def _build_final_guardrails_list(current_agent, guardrail_checks: List["GuardrailCheck"], user_input: str) -> List["GuardrailCheck"]:
    """Собирает итоговый список проверок гардрейлов: добавляет проваленные из guardrail_checks и помечает прочие входные гардрейлы как пройденные."""
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=user_input,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))
    return final_guardrails

def _build_output_guardrails_list(result) -> List["OutputGuardrailCheck"]:
    """Извлекает результаты выходных гардрейлов из результата Runner."""
    output_guardrails: List[OutputGuardrailCheck] = []
    
    # Проверяем, есть ли output_guardrail_results в результате
    if hasattr(result, 'output_guardrail_results'):
        for gr_result in result.output_guardrail_results:
            try:
                # Извлекаем информацию из результата гардрейла
                output_info = gr_result.output.output_info
                reasoning = getattr(output_info, 'reasoning', '')
                final_text = getattr(output_info, 'final_text', '')
                tripwire_triggered = gr_result.output.tripwire_triggered
                
                # Получаем исходный output - это текст, который обрабатывал output guardrail
                original_output = ''
                
                # Основной источник - agent_output содержит оригинальный ответ агента
                if hasattr(gr_result, 'agent_output'):
                    if isinstance(gr_result.agent_output, str):
                        original_output = gr_result.agent_output
                    elif hasattr(gr_result.agent_output, 'response'):
                        original_output = gr_result.agent_output.response
                    elif hasattr(gr_result.agent_output, 'content'):
                        original_output = gr_result.agent_output.content
                    else:
                        original_output = str(gr_result.agent_output)
                
                # Fallback на другие возможные источники
                if not original_output and hasattr(gr_result, 'input'):
                    if isinstance(gr_result.input, str):
                        original_output = gr_result.input
                    elif hasattr(gr_result.input, 'response'):
                        original_output = gr_result.input.response
                    else:
                        original_output = str(gr_result.input)
                
                logger.debug(f"Extracted original_output from agent_output: {original_output[:100] if original_output else 'EMPTY'}")
                
                # Получаем имя гардрейла
                guardrail_name = _get_guardrail_name(gr_result.guardrail)
                
                output_guardrails.append(OutputGuardrailCheck(
                    id=uuid4().hex,
                    name=guardrail_name,
                    input_text=original_output,
                    output=original_output,
                    reasoning=reasoning,
                    final_text=final_text,
                    tripwire_triggered=tripwire_triggered,
                    timestamp=time.time() * 1000,
                ))
            except Exception as e:
                # В случае ошибки логируем и добавляем базовую запись
                logger.warning(f"Error processing output guardrail result: {e}")
                output_guardrails.append(OutputGuardrailCheck(
                    id=uuid4().hex,
                    name="Unknown Output Guardrail",
                    input_text="",
                    output="",
                    reasoning="Error processing guardrail result",
                    final_text="",
                    tripwire_triggered=False,
                    timestamp=time.time() * 1000,
                ))
    
    return output_guardrails

def _apply_output_guardrail_final_text(messages: List["MessageResponse"], result) -> None:
    """Применяет final_text из output guardrails к сообщениям агента."""
    if not hasattr(result, 'output_guardrail_results'):
        return
    
    # Находим первый output guardrail с непустым final_text
    for gr_result in result.output_guardrail_results:
        try:
            output_info = gr_result.output.output_info
            final_text = getattr(output_info, 'final_text', '')
            
            if final_text and final_text.strip():
                # Заменяем содержимое последнего сообщения агента на final_text
                for message in reversed(messages):
                    if hasattr(message, 'content') and message.content:
                        message.content = final_text
                        logger.debug(f"Applied output guardrail final_text to message: {final_text[:100]}...")
                        break
                break
        except Exception as e:
            logger.warning(f"Error applying output guardrail final_text: {e}")
            continue

# =========================
# Main Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for agent orchestration.
    Handles conversation state, agent routing, and guardrail checks.
    """
    conv_id, state, early = _init_or_get_state(req)
    if early is not None:
        return early

    current_agent = _get_agent_by_name(state["current_agent"])
    state["input_items"].append({"content": req.message, "role": "user"})
    old_context = state["context"].model_dump().copy()

    run_result, guardrail_checks, early = _run_agent_with_guardrails(conv_id, current_agent, state, req.message)
    if early is not None:
        return early

    # Awaitable Runner.run: обеспечим await, если нужно
    result = await run_result  # type: ignore

    messages, events, current_agent = _process_result_items(result, current_agent)

    # Apply output guardrail final_text to messages if available
    _apply_output_guardrail_final_text(messages, result)

    ctx_event = _maybe_context_update_event(state, old_context, current_agent)
    if ctx_event:
        events.append(ctx_event)

    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conv_id, state)

    final_guardrails = _build_final_guardrails_list(current_agent, guardrail_checks, req.message)
    output_guardrails = _build_output_guardrails_list(result)

    result_response = ChatResponse(
        conversation_id=conv_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
        output_guardrails=output_guardrails,
    )

    # Чистый JSON с отступами (Pydantic v2): без префиксов логгера
    import json
    json_logger.info(json.dumps(result_response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    
    return result_response
