import logging
import re

import phonenumbers
from phonenumbers import NumberFormat, PhoneNumberMatcher

from config.settings import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def normalize_phone(phone: str) -> str:
    # remove all except digits
    phone = ''.join(char for char in phone if char.isdigit())
    # replace 8 to 7
    if phone.startswith('89'):
        phone = '7' + phone[1:]
    return phone


def format_phone_number(phone):
    """Format phone number to Letu API format"""
    if not phone.startswith('+'):
        phone = '+' + phone

    try:
        phone_number = phonenumbers.parse(phone, None)
    except phonenumbers.phonenumberutil.NumberParseException as e:
        logger.error(f"Invalid phone number: {phone} - {e}")
        return phone

    if len(str(phone_number.national_number)) == 10:
        num_format = NumberFormat(pattern="(\\d{3})(\\d{3})(\\d{2})(\\d{2})", format="(\\1) \\2-\\3-\\4")
        formatted = phonenumbers.format_by_pattern(phone_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL,
                                                   [num_format])
        return formatted
    else:
        formatted = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return formatted


def finding_phone_number(text):
    for match in PhoneNumberMatcher(text, "RU"):
        if match:
            return match
        else:
            return None


def finding_order_number(text):
    # phone_regex = re.compile(r'\b8(?:-?\d){8}\b')
    any_five_dig = re.compile(r'8[-\s]?\d{3}[-\s]?\d')
    # matches = phone_regex.findall(text)
    matches2 = any_five_dig.findall(text)
    if matches2:
        return matches2[0]
    else:
        return None