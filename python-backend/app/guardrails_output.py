import os
import logging

from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    TResponseInputItem,
)

from config.settings import get_settings, LOGGER_NAME

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)

"""Initialize application services"""
logger.info("Initializing application services")
settings = get_settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


# =========================
# OUTPUT GUARDRAILS
# =========================

class TovOutput(BaseModel):
    """Schema for Tone of Voice guardrail output."""
    reasoning: str
    final_text: str

output_guardrail_agent = Agent(
    model="gpt-4.1",
    name="Tone of Voice Guardrail",
    instructions=("""
    # Исправь текст в соответствии с Общим описанием твоей роли:
Ты умный молодой парень продавец-консультант Иван в онлайн-магазине электроники restore:, https://www.re-store.ru. Используй только такое написание restore: — название маленькими латинскими буквами с двоеточием на конце.

Всегда завершай своё сообщение вопросом, после ответа на вопросы пользователя, КРОМЕ случаев, когда пользователь явно завершает разговор.
Используй только кавычки-ёлочки, «», вместо любых других.
В нужных местах используй длинное тире —.
Никогда не используй разметку markdown, только plain text. Структурируй текст только с помощью отступов и перехода на другую строку.
Никогда не используй слово "помощь", только "полезность".
Во всех случаях используйте написание Яндекс Сплит только в этой форме — Яндекс Сплит.
При показе ссылок удаляем 'https://' и 'http://'

Использование эмодзи:
Используй эмодзи умеренно (в 20-30% сообщений) для выделения важных моментов. Применяй эмодзи только в следующих случаях:
1. При упоминании конкретных товаров (🧦🧢👟🥾🎒👕🧤🩳)
2. При выражении позитивных эмоций (😄😊🙂😉😎)
3. При поздравлении с удачной покупкой (👍🙌👏)
4. При описании преимуществ товара (✨💪)

    """
    ),
    output_type=TovOutput,
)

@output_guardrail(name="Tone of Voice Guardrail")
async def tov_guardrail(
        context: RunContextWrapper[None],
        agent: Agent,
        output: "MessageOutput"
) -> GuardrailFunctionOutput:
    """Guardrail to format the output as tone of voice."""
    # Передаём в помощника именно текст ответа, а не объект MessageOutput
    result = await Runner.run(output_guardrail_agent, output.response, context=context.context)
    final = result.final_output_as(TovOutput)
    # Append approval marker to the final text
    try:
        final.final_text = f"{final.final_text}\nTone of Voice Guardrail — APPROVED"
    except Exception:
        pass
    # Это форматирующий guardrail: не блокируем ответ.
    return GuardrailFunctionOutput(output_info=final, tripwire_triggered=False)


# =========================
# CHECKING OUTPUT GUARDRAILS
# =========================

class MessageOutput(BaseModel):
    response: str

class MathOutput(BaseModel):
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[tov_guardrail],
    output_type=MessageOutput,
)

async def main():
    # This should trip the guardrail
    try:
        result = await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print(f"{result.output_guardrail_results[0].output.output_info.final_text=}")
        print(f"{result.output_guardrail_results[0].output.output_info.reasoning=}")
        # print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())