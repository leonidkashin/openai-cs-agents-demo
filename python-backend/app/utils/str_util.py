import logging
import re

from config.settings import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def remove_markdown_links(text):
    # Regular expression to match Markdown links
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

    # Replace Markdown links with their titles
    def replace_links(match):
        return match.group(1)

    # Use re.sub to remove links
    cleaned_text = re.sub(link_pattern, replace_links, text)

    return cleaned_text


def remove_markdown(text: str):
    try:
        if not text:
            return text

        res = text

        # Remove inline links
        res = re.sub(r'\[([^\]]*?)\][\[\(](.*)[\]\)]', r'\1 (\2)', res)
        # Remove atx-style headers
        res = re.sub(r"^(\n)?\s{0,}#{1,6}\s*( (.+))? +#+$|^(\n)?\s{0,}#{1,6}\s*( (.+))?$", r'\1\3\4\6', res, flags=re.MULTILINE)
        # Remove * emphasis
        res = re.sub(r'([\*]+)(\S)(.*?\S)??\1', r'\2\3', res)
        # Remove _ emphasis. Unlike *, _ emphasis gets rendered only if
        res = re.sub(r'(^|\W)([_]+)(\S)(.*?\S)??\2($|\W)', r'\1\3\4\5', res)
    except Exception as e:
        logger.exception(f"remove_markdown exception: {e}")
        return text

    return res


def remove_newline_symbol(text: str):
    if not text:
        return text

    return text.replace('\\n', '')

def remove_brackets(text: str):
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = re.sub(r'\[.*?\]', '', text)
    text = find_and_modify_links(text)
    text = text.replace('"', '')
    return text


def trim_html(source: str):
    if not source:
        return source

    source = source.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    return source


def normalize_text(text):
    text = text.replace("\r\n", "\n")  # replace windows line endings
    text = text.replace("<br/>", "\n").replace("<br>", "\n").replace("<br />", "\n")  # replace html line endings
    text = text.strip()
    text = clean_html_tags(text)
    return text


def clean_html_tags(text):
    clean_html_tags_re = re.compile('<.*?>')
    text = re.sub(clean_html_tags_re, '', text)
    return text


def find_and_modify_links(text):
    # Регулярное выражение для нахождения URL

    url_pattern = re.compile(r'(https?://[a-zA-Z0-9.\-]+(?:\.[a-zA-Z]{2,6})(?:/[^\s;,.\n]*)?)')

    # Функция замены: добавляет пробел после найденной ссылки, если его нет
    def add_space(match):
        return match.group(0) + ' '

    # Заменяем все найденные ссылки, добавляя пробел после каждой
    modified_text = url_pattern.sub(add_space, text)

    return modified_text

def find_pattern_from_list_in_text(patterns: list[str], text: str) -> str | None:
    pass
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None
