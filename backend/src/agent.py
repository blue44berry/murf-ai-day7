import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("food-grocery-agent")
load_dotenv(".env.local")

# -----------------------------
#   Paths & Simple "Database"
# -----------------------------

BASE_DIR = Path(__file__).parent.parent
CATALOG_PATH = BASE_DIR / "food_catalog.json"
ORDERS_DIR = BASE_DIR / "orders_day7"
ORDERS_DIR.mkdir(exist_ok=True)


# -----------------------------
#   Data Models
# -----------------------------

@dataclass
class CatalogItem:
    id: str
    name: str
    category: str
    price: float
    brand: Optional[str] = None
    size: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CartItem:
    item_id: str
    name: str
    quantity: int
    price: float
    notes: Optional[str] = None


@dataclass
class Order:
    order_id: str
    customer_name: Optional[str]
    address: Optional[str]
    items: List[CartItem]
    total: float
    timestamp: str


@dataclass
class SessionState:
    cart: List[CartItem] = field(default_factory=list)
    customer_name: Optional[str] = None
    address: Optional[str] = None
    # you can store last order id if you want to extend later
    last_order_id: Optional[str] = None


RunCtx = RunContext[SessionState]

# -----------------------------
#   Catalog Loading Helpers
# -----------------------------

def load_catalog() -> List[CatalogItem]:
    if not CATALOG_PATH.exists():
        logger.warning(f"Catalog file not found at {CATALOG_PATH}.")
        return []

    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[CatalogItem] = []
    for entry in raw:
        try:
            items.append(
                CatalogItem(
                    id=str(entry["id"]),
                    name=entry["name"],
                    category=entry["category"],
                    price=float(entry["price"]),
                    brand=entry.get("brand"),
                    size=entry.get("size"),
                    tags=entry.get("tags", []),
                )
            )
        except KeyError as e:
            logger.error(f"Skipping malformed catalog entry: {entry} (missing {e})")
    return items


CATALOG: List[CatalogItem] = load_catalog()

def find_items_by_name(name: str) -> List[CatalogItem]:
    """Simple case-insensitive substring match on item name."""
    name_norm = name.strip().lower()
    return [item for item in CATALOG if name_norm in item.name.lower()]


def find_item_by_id(item_id: str) -> Optional[CatalogItem]:
    for item in CATALOG:
        if item.id == item_id:
            return item
    return None


# -----------------------------
#   Simple “Recipe” Mapping
#   For “ingredients for X”
# -----------------------------
# Dish name (lowercase) -> list of (item_id, quantity)

RECIPES = {
    # you must ensure these IDs exist in your food_catalog.json
    "peanut butter sandwich": [
        {"item_id": "bread_loaf", "quantity": 1},
        {"item_id": "peanut_butter_jar", "quantity": 1},
    ],
    "simple pasta for two": [
        {"item_id": "pasta_500g", "quantity": 1},
        {"item_id": "pasta_sauce_jar", "quantity": 1},
        {"item_id": "cheese_block", "quantity": 1},
    ],
    "maggi snack": [
        {"item_id": "instant_noodles_pack", "quantity": 2},
    ],
}

def normalize_dish_name(dish: str, servings: Optional[int]) -> str:
    dish = dish.strip().lower()
    if "pasta" in dish and servings and servings <= 2:
        return "simple pasta for two"
    return dish


# -----------------------------
#   Order Saving Helper
# -----------------------------

def save_order(order: Order) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{ts}_{order.order_id}.json"
    path = ORDERS_DIR / filename

    # Convert dataclass to a JSON-serializable dict
    payload = {
        "order_id": order.order_id,
        "customer_name": order.customer_name,
        "address": order.address,
        "timestamp": order.timestamp,
        "total": order.total,
        "items": [
            {
                "item_id": ci.item_id,
                "name": ci.name,
                "quantity": ci.quantity,
                "price": ci.price,
                "notes": ci.notes,
            }
            for ci in order.items
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved food order to {path}")
    return str(path)


# -----------------------------
#   Food & Grocery Agent
# -----------------------------

class FoodOrderingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly food & grocery ordering assistant for a fictional store called "QuickCart".
You talk over voice and help the user build a cart and place an order.

Your capabilities:
- You can add items from the catalog to the cart using tools.
- You can update quantities, remove items, and list the current cart.
- You can handle "ingredients for X" requests by using the special recipe tool.
- When the user says they're done ("that's all", "place my order", "I'm done"), finalize the order.

Very important behavioral rules:
1. Always rely on tools to modify the cart or place the order. Do NOT invent cart state.
2. When the user mentions a specific item and quantity, call `add_item_to_cart`.
3. When they want to remove or change quantity, call `remove_item_from_cart` or `update_item_quantity`.
4. When they ask "what's in my cart", call `get_cart_summary`.
5. For requests like "I need ingredients for a peanut butter sandwich" or "get ingredients for pasta for two":
   - Call `add_ingredients_for_dish` with the dish name and (if provided) servings.
6. Before placing the order, make sure:
   - Cart is not empty.
   - You have at least the customer's name, and optionally address or area.
   Then call `place_order`.

Tone:
- Warm, concise, and helpful.
- Ask clarifying questions if item, brand, size, or quantity are unclear.
- Confirm each major cart change aloud.
""",
        )

    # --------- TOOLS (CART MANAGEMENT) --------- #

    @function_tool()
    async def add_item_to_cart(
        self,
        ctx: RunCtx,
        item_name: str,
        quantity: int = 1,
        notes: Optional[str] = None,
    ) -> str:
        """
        Add an item (by name) to the cart with a given quantity.
        Use this when the user says things like:
        - "Add 2 packets of bread"
        - "Get me 1 litre of milk"
        """
        matches = find_items_by_name(item_name)
        if not matches:
            return f"I couldn't find any item matching '{item_name}' in the catalog."

        if len(matches) > 1:
            # Let the LLM ask user to clarify
            names = ", ".join(f"{m.name} (id: {m.id})" for m in matches)
            return (
                "I found multiple matching items. "
                f"Ask the user to clarify which one they want: {names}"
            )

        item = matches[0]
        if quantity <= 0:
            quantity = 1

        ctx.userdata.cart.append(
            CartItem(
                item_id=item.id,
                name=item.name,
                quantity=quantity,
                price=item.price,
                notes=notes,
            )
        )
        return f"Added {quantity} x {item.name} to the cart."

    @function_tool()
    async def remove_item_from_cart(
        self,
        ctx: RunCtx,
        item_name: str,
    ) -> str:
        """
        Remove the first matching item (by name substring) from the cart.
        Use when the user says things like:
        - "Remove the bread"
        - "Take out the chips"
        """
        name_norm = item_name.strip().lower()
        cart = ctx.userdata.cart

        for i, ci in enumerate(cart):
            if name_norm in ci.name.lower():
                removed = cart.pop(i)
                return f"Removed {removed.name} from the cart."

        return f"I didn't find any item matching '{item_name}' in the cart."

    @function_tool()
    async def update_item_quantity(
        self,
        ctx: RunCtx,
        item_name: str,
        new_quantity: int,
    ) -> str:
        """
        Update the quantity of a cart item.
        Use when user says:
        - "Make it 3 milks instead of 1"
        - "Change the pasta to 2 packs"
        """
        if new_quantity <= 0:
            return "Quantity must be at least 1."

        name_norm = item_name.strip().lower()
        for ci in ctx.userdata.cart:
            if name_norm in ci.name.lower():
                old_q = ci.quantity
                ci.quantity = new_quantity
                return (
                    f"Updated {ci.name} from quantity {old_q} to {new_quantity} in the cart."
                )

        return f"I couldn't find any item matching '{item_name}' in the cart."

    @function_tool()
    async def get_cart_summary(self, ctx: RunCtx) -> str:
        """
        Return a natural language summary of the cart:
        items, quantities, and total price.
        Use this when the user asks: "What's in my cart?"
        """
        if not ctx.userdata.cart:
            return "The cart is currently empty."

        lines = []
        total = 0.0
        for ci in ctx.userdata.cart:
            line_total = ci.price * ci.quantity
            total += line_total
            lines.append(f"{ci.quantity} x {ci.name} (₹{ci.price:.2f} each)")

        summary = "; ".join(lines)
        return f"Your cart has: {summary}. The current total is approximately ₹{total:.2f}."

    # --------- TOOLS (INGREDIENTS FOR DISH) --------- #

    @function_tool()
    async def add_ingredients_for_dish(
        self,
        ctx: RunCtx,
        dish_name: str,
        servings: Optional[int] = None,
    ) -> str:
        """
        Add multiple items to the cart for simple recipes like:
        - "ingredients for a peanut butter sandwich"
        - "ingredients for pasta for two"
        The mapping is pre-defined via recipe IDs in the backend.
        """
        key = normalize_dish_name(dish_name, servings)
        recipe = RECIPES.get(key)
        if not recipe:
            return (
                f"I don't have a stored recipe for '{dish_name}'. "
                "You can ask for individual items instead."
            )

        added_lines = []
        for entry in recipe:
            item = find_item_by_id(entry["item_id"])
            if not item:
                continue
            qty = entry.get("quantity", 1)
            ctx.userdata.cart.append(
                CartItem(
                    item_id=item.id,
                    name=item.name,
                    quantity=qty,
                    price=item.price,
                    notes=f"for {dish_name}",
                )
            )
            added_lines.append(f"{qty} x {item.name}")

        if not added_lines:
            return "I couldn't map the recipe items to the catalog. Please add items manually."

        added_str = ", ".join(added_lines)
        return (
            f"For {dish_name}, I added the following to your cart: {added_str}. "
            "You can change quantities if you like."
        )

    # --------- TOOLS (PLACE ORDER) --------- #

    @function_tool()
    async def set_customer_info(
        self,
        ctx: RunCtx,
        name: Optional[str] = None,
        address: Optional[str] = None,
    ) -> str:
        """
        Store customer information like name and address.
        You can call this when the user tells you their name or address.
        """
        if name:
            ctx.userdata.customer_name = name
        if address:
            ctx.userdata.address = address

        parts = []
        if ctx.userdata.customer_name:
            parts.append(f"name: {ctx.userdata.customer_name}")
        if ctx.userdata.address:
            parts.append(f"address: {ctx.userdata.address}")

        if not parts:
            return "No customer info stored yet."
        return "Stored customer info: " + ", ".join(parts)

    @function_tool()
    async def place_order(self, ctx: RunCtx) -> str:
        """
        Finalize the current cart, compute total, and save to an order JSON file.
        Call this when the user says they are done ordering.
        """
        if not ctx.userdata.cart:
            return "The cart is empty, so I cannot place an order yet."

        customer_name = ctx.userdata.customer_name or "Guest"
        address = ctx.userdata.address

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        order_id = datetime.now().strftime("%Y%m%d%H%M%S")

        total = sum(ci.price * ci.quantity for ci in ctx.userdata.cart)
        order = Order(
            order_id=order_id,
            customer_name=customer_name,
            address=address,
            items=list(ctx.userdata.cart),
            total=total,
            timestamp=ts,
        )

        path_str = save_order(order)
        ctx.userdata.last_order_id = order_id

        # Clear the cart after placing order
        ctx.userdata.cart.clear()

        return (
            f"Order placed successfully for {customer_name}. "
            f"Total amount is approximately ₹{total:.2f}. "
            f"The order has been saved to a JSON file at: {path_str}. "
            "Let the user know their order is confirmed."
        )


# -----------------------------
#   Worker / Session Wiring
# -----------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def food_agent_entry(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session_state = SessionState()

    session = AgentSession[SessionState](
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FoodOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


async def entrypoint(ctx: JobContext):
    await food_agent_entry(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
