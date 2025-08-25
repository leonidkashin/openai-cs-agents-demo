import datetime
from typing import List, Optional
# from async_lru import alru_cache
import json

import pymongo
import pytz
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from utils.str_util import clean_html_tags, normalize_text, remove_markdown_links, remove_brackets

db: AsyncIOMotorDatabase
chat_history = "chat_history"
conversation_state = "conversation_state"
flomni_chat = "flomni_chat"
flomni_chat_session = "flomni_chat_session"
cities = "cities"
trade_in = "trade_in"
repair_price = "repair_prices"
feed_stores = "feed_stores"
shops_by_id = "shops_by_ids"


class FlomniChatStatus:
    active = 'active'
    skipped = 'skipped'
    operator = 'operator'  # перевод нами на оператора
    agent_joined = 'agent_joined'
    closed = 'closed'
    closed_frozen = 'closed_frozen'


def init_db(mongodb_dsn: str):
    global db
    client = AsyncIOMotorClient(mongodb_dsn)
    db = client.get_database()


def get_db():
    return db


async def is_history_message_exists(msg_id: str) -> bool:
    doc = await db[chat_history].find_one({"msg_id": msg_id})
    return doc is not None


async def is_old_message(conv_id: str, date: datetime.datetime) -> bool:
    doc = await db[chat_history].find_one({"conv_id": conv_id, "date": {"$gt": date}})
    return doc is not None


async def get_last_message_date(conv_id: str) -> Optional[datetime.datetime]:
    result = await db[chat_history].find({"conv_id": conv_id}).sort([("date", pymongo.DESCENDING)]).limit(1).to_list(length=None)
    return result[0]["date"] if result else None


async def add_history(conv_id: str,
                      msg_id: Optional[str],
                      user_id: Optional[str],
                      text: str,
                      role: str,
                      date: datetime.datetime,
                      has_attachments: bool,
                      completion_id: Optional[str] = None,
                      completion_date: Optional[datetime.datetime] = None) -> dict:
    item = {
        "conv_id": conv_id,
        "msg_id": msg_id,
        "user_id": user_id,
        "text": text,
        "role": role,
        "date": date,
        "date_real": datetime.datetime.utcnow(),
        "has_attachments": has_attachments,
        "completion_id": completion_id,
        "completion_shown_date": completion_date
    }

    result = await db[chat_history].insert_one(item)
    return item | {"_id": result.inserted_id}


async def update_history(
        doc: dict,
        reply: str,
        reply_status: bool,
        prompt_status: bool,
        stat_dict: dict
):
    update_data = {
        "completion_text": reply,
        "completion_date": datetime.datetime.utcnow(),
        "completion_status": reply_status,
        "prompt_status": prompt_status,
        "stats": stat_dict
    }

    await db[chat_history].update_one({"_id": doc["_id"]}, {"$set": update_data})


async def update_history_cancelled(doc: dict):
    update_data = {
        "cancelled": True
    }

    await db[chat_history].update_one({"_id": doc["_id"]}, {"$set": update_data})


async def update_history_error(doc: dict, error: str):
    update_data = {
        "error": error
    }

    await db[chat_history].update_one({"_id": doc["_id"]}, {"$set": update_data})


async def get_history(conv_id: Optional[str] = None, user_id: Optional[str] = None, limit: int = 11,
                      date_from: Optional[datetime.datetime] = None) -> List[dict]:
    if not conv_id and not user_id:
        raise ValueError("Either conv_id or user_id must be specified")

    query = {}
    if conv_id:
        query["conv_id"] = conv_id
    if user_id:
        query["user_id"] = user_id
    if date_from:
        query["date"] = {"$gt": date_from}

    cursor = db[chat_history].find(query).sort("date", pymongo.DESCENDING).limit(limit)
    result = await cursor.to_list(length=None)
    result.reverse()

    history = []
    for item in result:
        role = item.get("role")
        text = item.get("text")

        if role == "user":
            history.append({"role": "user", "content": text})
        elif role == "support":
            history.append({"role": "assistant", "content": text})

    return history


async def get_history_debug(conv_id: str):
    cursor = db[chat_history].find({"conv_id": conv_id}).sort("date", pymongo.ASCENDING)
    result = await cursor.to_list(length=None)
    return result


async def get_cities_list():
    cursor = db[cities].find({})
    result = await cursor.to_list(length=None)

    table = [item["_id"] for item in result]
    return table

async def get_shops_list():
    cursor = db[feed_stores].find({})
    stores = await cursor.to_list(length=None)

    all_shops = []
    for shop in stores:
        if not shop.get("address"):
            continue
        shop_dict = {}
        shop_dict['Название'] = shop['name']
        shop_dict['город'] = shop['city']
        shop_dict['адрес'] = shop['address'] + (
            f', метро {shop["subway"]}' if shop.get("subway") else "рядом нет")
        shop_dict['адрес'] = shop['address']
        shop_dict['service_encoding'] = shop['_id']
        all_shops.append(str(shop_dict))

    return all_shops

# @alru_cache(maxsize=1)
async def get_stores_collection():
    return await db[feed_stores].find({}).to_list(length=None)


async def get_store_info(store_description: Optional[str] = '', city_name: Optional[str] = '') -> str:
    if store_description:
        stores = await get_stores_collection()
        shop_info = []
        for shop in stores:
            if not shop.get("address"):
                continue
            shop_dict = {}
            shop_dict['Название'] = shop['name']
            shop_dict['город'] = shop['city']
            shop_dict['адрес'] = shop['address'] + (
                f', метро {shop["subway"]}' if shop.get("subway") else "рядом нет")
            if str(shop_dict) in store_description:
                shop_dict['телефон'] = shop['phone'] if shop.get('phone') else '',
                shop_dict['email'] = shop['email'] if shop.get('email') else '',
                shop_dict['Часы работы'] = normalize_text(shop['workHours']) if shop.get('workHours') else '',
                shop_dict['==='] = '==='
                shop_info.append(shop_dict)

        return json.dumps(shop_info)

    else:
        return await get_store_info_by_city(city_name)

async def get_store_info_by_city(city_name: Optional[list] = []):
    city_id = city_name[0]

    cursor = db[feed_stores].find({'city': city_id})
    result = await cursor.to_list(length=None)

    text_version = []
    for item in result:
        if item.get('address'):
            shop = {'Название Магазина': item['name'],
                    'город': item['city'],
                    'адрес': normalize_text(item['address']),
                    'телефон': item['phone'] if item.get('phone') else '',
                    'email': item['email'] if item.get('email') else '',
                    'Часы работы': normalize_text(item['workHours']) if item.get('workHours') else '',
                    '===': '==='}
            text_version.append(shop)
        else:
            continue

    result = f'''
    Список магазинов в городе:
    {text_version if text_version else "Нет магазинов в этом городе"}
    '''

    return result


async def get_trade_in():
    cursor = db[trade_in].find()
    result = await cursor.to_list(length=None)

    table = []
    for item in result:
        table.append(f"model: {item['model']}; price: {item['price']}; specifications: {item['specifications']};")

    return table

# todo
async def get_repair():
    return await db[repair_price].find().to_list(length=None)

async def get_trade_in_raw():
    return await db[trade_in].find().to_list(length=None)

# @alru_cache(maxsize=1)
async def get_shops_by_id(product_id):
    return await db[shops_by_id].find({'_id': product_id}).to_list(length=None)

# @alru_cache(maxsize=1)
async def get_shops_by_city(city_id):
    stores = await get_stores_collection()
    shops_by_city = []
    for shop in stores:
        if str(shop.get('city')) == city_id:
            shops_by_city.append(shop['_id'])

    return shops_by_city

async def get_repair_names():
    service_names = await db[repair_price].distinct("Service")
    category_names = await db[repair_price].distinct("Category")
    model_names = await db[repair_price].distinct("Model")

    return service_names, category_names, model_names

async def get_trade_in_names():
    type_names = await db[trade_in].distinct("type")
    name_names = await db[trade_in].distinct("name")
    model_names = await db[trade_in].distinct("model")

    return type_names, name_names, model_names

async def get_history_by_id(reply_id: str) -> dict:
    return await db[chat_history].find_one({"_id": ObjectId(reply_id)})


async def update_feedback_status(doc, rate_status: bool):
    feedback_data = {"rate_status": rate_status}
    await db[chat_history].update_one({"_id": doc["_id"]}, {"$set": feedback_data})


async def save_usage(conv_id: str, model: str, prompt_tokens: int, completion_tokens: int):
    timezone = pytz.timezone('Europe/Moscow')
    date_local = datetime.datetime.now(timezone).date()
    date_local = datetime.datetime.combine(date_local, datetime.time.min)

    await db.usage.insert_one({"conv_id": conv_id,
                               "model": model,
                               "prompt_tokens": prompt_tokens,
                               "completion_tokens": completion_tokens,
                               "date": datetime.datetime.utcnow(),
                               "date_local": date_local})


async def amo_lead_upsert(lead_id: int, phone: str, contact_data: Optional[dict]):
    data = {
        "_id": lead_id,
        "phone": phone,
        "contact_data": contact_data
    }
    await db.amo_leads.update_one({"_id": lead_id}, {"$set": data}, upsert=True)


async def flomni_chat_get_all() -> list[dict]:
    cursor = db[flomni_chat].find()
    return await cursor.to_list(length=None)


async def flomni_chat_get(chat_id: str) -> Optional[dict]:
    return await db[flomni_chat].find_one({"_id": chat_id})


async def flomni_chat_upsert(chat_id: str, status: str, chat_data: dict):
    data = {
        "_id": chat_id,
        "status": status
    }

    if chat_data:
        data.update(chat_data)

    await db[flomni_chat].update_one({"_id": chat_id}, {"$set": data}, upsert=True)


async def flomni_chat_mark_completed(chat_id: str, is_frozen: bool, score_value: Optional[str] = None):
    update_data = {
        "status": FlomniChatStatus.closed_frozen if is_frozen else FlomniChatStatus.closed,
        "closed_successfully": True,
        "closed_date": datetime.datetime.utcnow(),
        "is_frozen": is_frozen,
        "last_message_date": None
    }

    if score_value:
        update_data["score_value"] = score_value

    await db[flomni_chat].update_one({"_id": chat_id}, {"$set": update_data})


async def flomni_chat_disable(chat_id: str):
    await db[flomni_chat].update_one({"_id": chat_id}, {"$set": {"status": FlomniChatStatus.operator}}, upsert=True)


async def flomni_chat_set_last_completion_id(user_id: str, reply_id):
    await db[flomni_chat].update_one({"_id": user_id}, {"$set": {"last_completion_id": reply_id}}, upsert=True)


async def flomni_chat_set_last_date(user_id: str, role: str):
    await db[flomni_chat].update_one({"_id": user_id}, {"$set": {
        "last_message_date": datetime.datetime.utcnow(),
        "last_message_role": role
    }}, upsert=True)


async def flomni_chat_get_older_than(minutes: int, role: str) -> List[dict]:
    date = datetime.datetime.utcnow() - datetime.timedelta(minutes=minutes)
    cursor = db[flomni_chat].find({
        "last_message_date": {"$lt": date},
        "last_message_role": role,
        "status": FlomniChatStatus.active,
    })
    return await cursor.to_list(length=None)


async def start_new_session(user_id: str):
    await db[flomni_chat_session].insert_one({
        "user_id": user_id,
        "start_date": datetime.datetime.utcnow()
    })

async def get_latest_session(user_id: str) -> Optional[dict]:
    return await db[flomni_chat_session].find_one({"user_id": user_id}, sort=[("start_date", pymongo.DESCENDING)])


async def end_session(user_id: str, status: str, closed_successfully: bool, score_value: Optional[str] = None, change_reason: Optional[str] = None):
    now = datetime.datetime.utcnow()
    session = await get_latest_session(user_id)
    chat_user = await flomni_chat_get(user_id)

    service = chat_user.get("service") if chat_user else None

    if not session:
        await db[flomni_chat_session].insert_one({
            "user_id": user_id,
            "start_date": now,
            "end_date": now,
            "status": status,
            "closed_successfully": closed_successfully,
            "error": "No session found",
            "score_value": score_value,
            "service": service,
            "to_operator_reason": change_reason,
        })
    elif "end_date" in session: # already ended
        await db[flomni_chat_session].insert_one({
            "user_id": user_id,
            "start_date": now,
            "end_date": now,
            "status": status,
            "closed_successfully": closed_successfully,
            "error": "Session already ended",
            "score_value": score_value,
            "service": service,
            "to_operator_reason": change_reason
        })
    else:
        await db[flomni_chat_session].update_one({"_id": session["_id"]}, {"$set": {
            "end_date": now,
            "status": status,
            "closed_successfully": closed_successfully,
            "score_value": score_value,
            "service": service,
            "to_operator_reason": change_reason
        }})

async def db_save_usage(usage):
    await db["usage"].insert_one(usage)


# =========================
# Conversation State (for chat orchestration)
# =========================

async def get_conversation_state(conv_id: str) -> Optional[dict]:
    """
    Retrieve conversation state document.
    Returns None if not found.
    """
    return await db[conversation_state].find_one({"_id": conv_id})


async def save_conversation_state(conv_id: str, state: dict):
    """
    Upsert conversation state. The state should be JSON-serializable.
    """
    data = {"_id": conv_id} | state
    await db[conversation_state].update_one({"_id": conv_id}, {"$set": data}, upsert=True)


async def get_vendor_codes_by_asre():
    """
    Retrieves special IDs from the vendor_codes_by_asre collection.

    Returns:
        List[str]: A list of vendor codes
    """
    cursor = db["vendor_codes_by_asre"].find({})
    result = await cursor.to_list(length=None)

    # Extract the vendor codes from the result
    vendor_codes = []
    for item in result:
        if "_id" in item:
            vendor_codes.append(str(item["_id"]))

    return vendor_codes
