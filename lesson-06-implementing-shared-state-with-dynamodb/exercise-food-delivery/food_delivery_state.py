"""
food_delivery_state.py - EXERCISE STARTER (Student-Led)
=========================================================
Module 6 Exercise: Build Shared State for a Food Delivery Order System

Architecture:
    Customer places order
         │
    ┌────┴────────────────────────────────────────────────┐
    │  Shared State (DynamoDB)                            │
    │  order_id (PK) | version | restaurant | driver |    │
    │  total_price | status | ttl                         │
    │  Optimistic locking: version-based conditional      │
    │  writes (ConditionExpression) prevent lost updates  │
    │  TTL: auto-expire completed orders after 2 hours    │
    └────┬────────────────────────────────────────────────┘
         │
    Four agents update the SAME record:
    ┌────┴───────────────────────────────────────────────────┐
    │ RestaurantConfirmAgent → writes status (accept/reject) │
    │ DriverAssignAgent     → writes driver field            │
    │ PriceCalculatorAgent  → writes total_price             │
    │ StatusTrackerAgent    → writes progress updates        │
    └────────────────────────────────────────────────────────┘
         │
    Cross-Session Memory (AgentCore Memory):
    ┌────┴────────────────────────────────────────────┐
    │ SESSION_SUMMARY strategy → customer preferences │
    │ Remembers preferred driver across orders        │
    └─────────────────────────────────────────────────┘

Same shared state pattern as the demo (ride_sharing_state.py),
with two additions:
  1. STATE RECOVERY: If restaurant rejects, cleanup partial updates
  2. FOUR agents instead of three (more concurrent conflicts)

Instructions:
  - Follow the demo pattern (ride_sharing_state.py)
  - Look for TODO 1-18 below
  - State management functions: create/update/get + recovery
  - Each build_*_agent function needs: model, system_prompt, Agent()
  - DynamoDB table and customer_memory are provided

Tech Stack:
  - Python 3.11+
  - Strands Agents SDK (Agent class, @tool decorator)
  - Amazon Bedrock (Nova Lite for all agents)
  - DynamoDB shared state (real AWS resource — created by CloudFormation)
  - AgentCore Memory simulation (in-memory; production uses bedrock-agentcore-control)
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel

load_dotenv()

logging.basicConfig(level=logging.WARNING)


def to_dynamo(obj):
    """Convert Python objects to DynamoDB-compatible types (float→Decimal)."""
    return json.loads(json.dumps(obj), parse_float=Decimal)


def _dynamo_default(o):
    """Convert Decimal to int if whole number, float otherwise."""
    if isinstance(o, Decimal):
        return int(o) if o == int(o) else float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def from_dynamo(obj):
    """Convert DynamoDB types back to Python (Decimal→int or float)."""
    return json.loads(json.dumps(obj, default=_dynamo_default))


def clean_response(text: str) -> str:
    """Strip <thinking>...</thinking> tags from Nova model outputs."""
    return re.sub(
        r"<thinking>.*?</thinking>\s*", "", str(text), flags=re.DOTALL
    ).strip()


# NOTE: In production, extract shared helpers like run_agent_with_retry() and
# clean_response() to a common utils.py module to avoid code duplication.
def run_agent_with_retry(agent_builder, prompt: str, max_retries: int = 3) -> float:
    """Run an agent with retry logic for transient Bedrock errors.
    Uses exponential backoff (1s, 2s, 4s) to handle throttling."""
    for attempt in range(max_retries):
        try:
            agent = agent_builder()
            t = time.time()
            agent(prompt)
            return time.time() - t
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(
                    f"    [Retry {attempt + 1}/{max_retries}] {e.__class__.__name__}, waiting {wait}s..."
                )
                time.sleep(wait)
            else:
                print(
                    f"    [Failed] {e.__class__.__name__} after {max_retries} attempts"
                )
                raise


AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")

ORDERS = [
    {
        "order_id": "ORD-001",
        "customer_id": "CUST-42",
        "customer_name": "Alice Chen",
        "restaurant": "Tokyo Ramen House",
        "items": [
            {"name": "Tonkotsu Ramen", "price": 16.50, "qty": 1},
            {"name": "Gyoza (6pc)", "price": 8.00, "qty": 2},
        ],
        "address": "123 Main St, Apt 4B",
        "payment_method": "credit_card",
        "simulate_rejection": False,
    },
    {
        "order_id": "ORD-002",
        "customer_id": "CUST-77",
        "customer_name": "Bob Martinez",
        "restaurant": "Bella Italia",
        "items": [
            {"name": "Margherita Pizza", "price": 14.00, "qty": 1},
            {"name": "Caesar Salad", "price": 10.00, "qty": 1},
            {"name": "Tiramisu", "price": 9.00, "qty": 1},
        ],
        "address": "456 Oak Ave, Suite 200",
        "payment_method": "credit_card",
        "simulate_rejection": False,
    },
    {
        "order_id": "ORD-003",
        "customer_id": "CUST-42",
        "customer_name": "Alice Chen",
        "restaurant": "Tokyo Ramen House",
        "items": [{"name": "Spicy Miso Ramen", "price": 17.50, "qty": 1}],
        "address": "123 Main St, Apt 4B",
        "payment_method": "credit_card",
        "simulate_rejection": True,
    },
]

AVAILABLE_DRIVERS = [
    {
        "driver_id": "DRV-301",
        "name": "Carlos Rivera",
        "rating": 4.9,
        "vehicle": "Honda Civic",
    },
    {
        "driver_id": "DRV-302",
        "name": "Maria Santos",
        "rating": 4.85,
        "vehicle": "Toyota Corolla",
    },
    {
        "driver_id": "DRV-303",
        "name": "James Wilson",
        "rating": 4.7,
        "vehicle": "Ford Focus",
    },
]

DELIVERY_FEE = 4.99
TAX_RATE = 0.08


class VersionConflictError(Exception):
    """Raised when optimistic locking detects a version mismatch.
    Maps to DynamoDB ConditionalCheckFailedException."""


# STEP 1: DYNAMO_DB SHARED STATE — Real DynamoDB with Optimistic Locking
ORDER_STATE_TABLE = os.environ.get(
    "ORDER_STATE_TABLE", "lesson-06-shared-state-order-state"
)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
order_table = dynamodb.Table(ORDER_STATE_TABLE)

# Diagnostic write log (in-memory — for demo output only, not stored in DynamoDB)
_write_log = []

# AgentCore Memory simulation (in-memory; production uses bedrock-agentcore-control)
customer_memory = {}


# TODO 1: Implement create_order(order_data)
#   - Build record: order_id, customer_id, customer_name, restaurant, items, address, payment_method, driver=None,
#     total_price=None, status="pending", progress=[], version=0, ttl (2 hrs), created_at (UTC ISO)
#   - Call order_table.put_item(Item=to_dynamo(record))
#   - Log to _write_log: {"op": "put_item", "pk": order_id, "version": 0, "timestamp": time.time()}
#   - Return the record
#   Hint: Same as demo's create_trip(), but with order fields
def create_order(order_data: dict) -> dict:
    """Create a new order record in DynamoDB with initial state."""
    order_id = order_data["order_id"]
    now = datetime.now(UTC)
    record = {
        "order_id": order_id,
        "customer_id": order_data["customer_id"],
        "customer_name": order_data["customer_name"],
        "restaurant": order_data["restaurant"],
        "items": order_data["items"],
        "address": order_data["address"],
        "payment_method": order_data["payment_method"],
        "driver": None,
        "total_price": None,
        "status": "pending",
        "progress": [],
        "version": 0,
        "ttl": int(now.timestamp()) + 7200,
        "created_at": datetime.now(UTC).isoformat(),
    }
    order_table.put_item(Item=to_dynamo(record))
    _write_log.append(
        {"op": "put_item", "pk": order_id, "version": 0, "timestamp": time.time()}
    )
    return record


# TODO 2: Implement update_order(order_id, updates, max_retries=3)
#   - Loop max_retries times:
#     a) Read current with order_table.get_item(Key={"order_id": order_id})
#     b) Get expected_version from current["version"]
#     c) Apply updates, increment version
#     d) Try order_table.put_item() with ConditionExpression="version = :expected_ver"
#     e) On ClientError with "ConditionalCheckFailedException": wait + retry, else raise VersionConflictError
#   - Return the updated record (converted with from_dynamo)
#   Hint: Same as demo's update_trip() — the KEY pattern
def update_order(order_id: str, updates: dict, max_retries: int = 3) -> dict:
    """Update an existing order record with optimistic locking with retry.

    Every agent call this to write its piece of shared order state.
    Three steps: READ the current version → MODIFY locally → WRITE with a
    version condition. If another agent wrote first, DynamoDB rejects the
    write and we retry from step 1 with the fresh version.
    """
    for attempt in range(max_retries):
        response = order_table.get_item(Key={"order_id": order_id})
        current = response.get("Item")
        if not current:
            raise ValueError(f"Order {order_id} not found")
        expected_version = current["version"]

        current.update(to_dynamo(updates))
        current["version"] = expected_version + 1
        current["updated_at"] = datetime.now(UTC).isoformat()

        try:
            order_table.put_item(
                Item=current,
                ConditionExpression="version = :expected_ver",
                ExpressionAttributeValues={":expected_ver": expected_version},
            )
            _write_log.append(
                {
                    "op": "update_item",
                    "pk": order_id,
                    "version": f"{expected_version} → {expected_version + 1}",
                    "fields": list(updates.keys()),
                    "timestamp": time.time(),
                }
            )
            return from_dynamo(current)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                _write_log.append(
                    {
                        "op": "CONFLICT",
                        "pk": order_id,
                        "expected_version": expected_version,
                        "timestamp": time.time(),
                    }
                )
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    print(
                        f"    [Conflict] Order {order_id} version mismatch, retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise VersionConflictError(
                        f"Order {order_id} version conflict after {max_retries} attempts"
                    )
            else:
                raise


def get_order(order_id: str) -> dict | None:
    """Read an order record from DynamoDB."""
    response = order_table.get_item(Key={"order_id": order_id})
    item = response.get("Item")
    return from_dynamo(item) if item else None


def recover_order(order_id: str) -> dict:
    """State Recovery: Clean up after restaurant rejection (driver, price, status, progress)."""
    print(f"[Recovery] Cleaning up partial state for order {order_id}...")
    result = update_order(
        order_id,
        {
            "driver": None,
            "total_price": None,
            "status": "cancelled",
            "progress": ["Order rejected by restaurant", "Partial updates cleaned up"],
        },
    )
    print(f"[Recovery] Order {order_id} Cancelled. (v{result['version']})")
    return result


# TODO 5-7: Build RestaurantConfirmAgent
def build_restaurant_confirm_agent(simulate_rejection: bool = False) -> Agent:
    """Worker: Restaurant confirms or rejects the order."""

    model = bedrock_model = BedrockModel(
        model_id=NOVA_LITE_MODEL, temperature=0.0, region_name=AWS_REGION
    )
    system_prompt = """You are a restaurant order confirmation agent.
    You will receive an order ID and must confirm or reject the order.
    Your ONLY job:
    1. Call confirm_order with the order_id
    2. Report: Order <order_id> <confirmed/rejected> by restaurant
    Do NOT add any other commentary."""

    @tool
    def confirm_order(order_id: str) -> str:
        """Restaurant confirms or rejects the order."""
        order = get_order(order_id)
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})
        status = "rejected" if simulate_rejection else "confirmed"
        reason = (
            "Restaurant is closing early today"
            if simulate_rejection
            else "Order accepted, preparing now"
        )
        update_order(order_id, {"status": status})
        result = {
            "order_id": order_id,
            "restaurant": order["restaurant"],
            "status": status,
            "reason": reason,
        }
        return json.dumps(result, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[confirm_order])


# TODO 8-10: Build DriverAssignAgent
def build_driver_assign_agent() -> Agent:
    """Worker: Assigns a delivery driver."""

    model = BedrockModel(
        model_id=NOVA_LITE_MODEL, temperature=0.0, region_name=AWS_REGION
    )

    system_prompt = """You are a driver assignment agent.
    You will receive an order ID and must assign the best available driver.
    Your ONLY job:
    1. Call assign_driver with the order_id
    2. Report: Driver <name> assigned to order <order_id>
    Do NOT add any other commentary."""

    @tool
    def assign_driver(order_id: str) -> str:
        """Assign the best available driver to the order."""
        order = get_order(order_id)
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})
        cust_id = order.get("customer_id")
        preferred = customer_memory.get(cust_id, {}).get("preferred_driver")
        if preferred and any(d["driver_id"] == preferred for d in AVAILABLE_DRIVERS):
            best = next(d for d in AVAILABLE_DRIVERS if d["driver_id"] == preferred)
            reason = "preferred driver (from memory)"
        else:
            best = max(AVAILABLE_DRIVERS, key=lambda d: d["rating"])
            reason = "highest rated available"
        driver_info = {
            "driver_id": best["driver_id"],
            "name": best["name"],
            "rating": best["rating"],
            "vehicle": best["vehicle"],
            "match_reason": reason,
        }
        update_order(order_id, {"driver": driver_info})
        if cust_id not in customer_memory:
            customer_memory[cust_id] = {}
        customer_memory[cust_id]["preferred_driver"] = best["driver_id"]
        customer_memory[cust_id]["last_driver"] = best["name"]
        customer_memory[cust_id]["favorite_restaurant"] = order["restaurant"]
        customer_memory[cust_id]["usual_address"] = order["address"]
        return json.dumps({**driver_info, "order_id": order_id}, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[assign_driver])


# TODO 11-13: Build PriceCalculatorAgent
def build_price_calculator_agent() -> Agent:
    """Worker: Calculates total price with delivery fee and tax."""
    # TODO 11: Create BedrockModel with NOVA_LITE_MODEL, temperature=0.0
    model = BedrockModel(
        model_id=NOVA_LITE_MODEL, temperature=0.0, region_name=AWS_REGION
    )

    # TODO 12: Write system prompt — agent calls calculate_price, reports "Total for <order_id>: $<amount>"
    system_prompt = """You are a price calculation agent.
    You will receive an order ID and must calculate the total price including delivery fee and tax.
    Your ONLY job:
    1. Call calculate_price with the order_id
    2. Report: Total for <order_id>: $<amount>
    Do NOT add any other commentary."""

    @tool
    def calculate_price(order_id: str) -> str:
        """Calculate total order price including delivery fee and tax."""
        order = get_order(order_id)
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})
        subtotal = sum(item["price"] * item["qty"] for item in order["items"])
        tax = round(subtotal * TAX_RATE, 2)
        total = round(subtotal + tax + DELIVERY_FEE, 2)
        price_info = {
            "subtotal": subtotal,
            "tax": tax,
            "delivery_fee": DELIVERY_FEE,
            "total": total,
        }
        update_order(order_id, {"total_price": price_info})
        return json.dumps({**price_info, "order_id": order_id}, indent=2)

    return Agent(model=model, system_prompt=system_prompt, tools=[calculate_price])


# TODO 14-16: Build StatusTrackerAgent
def build_status_tracker_agent() -> Agent:
    """Worker: Updates order progress/status."""
    # TODO 14: Create BedrockModel with NOVA_LITE_MODEL, temperature=0.0
    model = BedrockModel(
        model_id=NOVA_LITE_MODEL, temperature=0.0, region_name=AWS_REGION
    )

    # TODO 15: Write system prompt — agent calls update_status, reports "Status for <order_id>: <status>"
    system_prompt = """You are a status tracking agent.
    You will receive an order ID and must update its progress and status.
    Your ONLY job:
    1. Call update_status with the order_id
    2. Report: Status for <order_id>: <status>
    Do NOT add any other commentary."""

    @tool
    def update_status(order_id: str) -> str:
        """Update order progress tracking."""
        order = get_order(order_id)
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})
        progress = order.get("progress", [])
        progress.append(
            f"Order processed at {datetime.now(UTC).strftime('%H:%M:%S UTC')}"
        )
        new_status = (
            "ready"
            if (order.get("driver") and order.get("total_price"))
            else order.get("status", "pending")
        )
        update_order(order_id, {"progress": progress, "status": new_status})
        return json.dumps(
            {
                "order_id": order_id,
                "status": new_status,
                "progress_count": len(progress),
            },
            indent=2,
        )

    return Agent(model=model, system_prompt=system_prompt, tools=[update_status])


# TODO 17-18: Wire up scenarios in main()
def main():
    print("=" * 70)
    print("  Food Delivery Order System — Module 6 Exercise")
    print("  Shared State with Optimistic Locking + State Recovery | 4 Agents")
    print("=" * 70)

    # TODO 17: Sequential scenario (Scenario 1)
    #   a) Create order1, print version + TTL
    #   b) Run 4 agents sequentially: RestaurantConfirm → DriverAssign → PriceCalculator → StatusTracker
    #   c) Print final state between steps
    #   d) Print summary box with order state
    #   Hint: Same flow as demo's Scenario 1, but with 4 agents instead of 3
    order1 = ORDERS[0]
    print(f"\n{'━' * 70}")
    print("  SCENARIO 1: Sequential Updates (no conflicts)")
    print(f"  Order: {order1['order_id']} — {order1['restaurant']}")
    items_str = ", ".join(f"{i['name']} x{i['qty']}" for i in order1["items"])
    print(f"  Items: {items_str} | Customer: {order1['customer_name']}")
    print(f"{'━' * 70}")

    record1 = create_order(order1)
    print(
        f"\n  Created: {order1['order_id']} (version {record1['version']}, "
        f"TTL: {datetime.fromtimestamp(record1['ttl'], tz=UTC).strftime('%H:%M:%S UTC')})"
    )

    sequential_steps = [
        ("RestaurantConfirmAgent", build_restaurant_confirm_agent, "Confirm order"),
        ("DriverAssignAgent", build_driver_assign_agent, "Assign driver for order"),
        (
            "PriceCalculatorAgent",
            build_price_calculator_agent,
            "Calculate price for order",
        ),
        ("StatusTrackerAgent", build_status_tracker_agent, "Update status for order"),
    ]
    for agent_name, agent_builder, instruction in sequential_steps:
        print(f"\n  [{agent_name}]")
        run_agent_with_retry(agent_builder, f"{instruction} {order1['order_id']}")
        state1 = get_order(order1["order_id"])
        print(f"    State: status={state1['status']} | version={state1['version']}")

    final1 = get_order(order1["order_id"])
    print(f"\n  {'─' * 52}")
    print(f"  FINAL STATE — {final1['order_id']}")
    print(f"    Version: {final1['version']} | Status: {final1['status']}")
    print(f"    Driver: {final1['driver']['name'] if final1['driver'] else 'None'}")
    print(
        f"    Total: ${final1['total_price']['total']:.2f}"
        if final1["total_price"]
        else "    Total: None"
    )
    print(f"    Progress: {final1['progress']}")
    print(f"  {'─' * 52}")

    # TODO 18: Concurrent scenario (Scenario 2)
    #   a) Create order2, print version
    #   b) Count conflicts before: sum(1 for e in db._write_log if e["op"] == "CONFLICT")
    #   c) Use ThreadPoolExecutor(max_workers=4) to run all 4 agents in parallel
    #   d) Count new conflicts (after - before)
    #   e) Print final state with conflict count + parallel time
    #   Hint: Same as demo's concurrent scenario, but with 4 futures instead of 3
    order2 = ORDERS[1]
    print(f"\n{'━' * 70}")
    print("  SCENARIO 2: Concurrent Updates (with conflicts)")
    print(f"  Order: {order2['order_id']} — {order2['restaurant']}")
    print("  All 4 agents run in PARALLEL — expect version conflicts!")
    print(f"{'━' * 70}")

    record2 = create_order(order2)
    print(
        f"\n  Created: {order2['order_id']} (version {record2['version']}, "
        f"TTL: {datetime.fromtimestamp(record2['ttl'], tz=UTC).strftime('%H:%M:%S UTC')})"
    )

    conflicts_before = sum(1 for e in _write_log if e["op"] == "CONFLICT")
    concurrent_steps = [
        ("RestaurantConfirmAgent", build_restaurant_confirm_agent, "Confirm order"),
        ("DriverAssignAgent", build_driver_assign_agent, "Assign driver for order"),
        (
            "PriceCalculatorAgent",
            build_price_calculator_agent,
            "Calculate price for order",
        ),
        ("StatusTrackerAgent", build_status_tracker_agent, "Update status for order"),
    ]
    start_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_agent_with_retry, builder, f"{instruction} {order2['order_id']}"
            ): name
            for name, builder, instruction in concurrent_steps
        }
        for future in as_completed(futures):
            name = futures[future]
            elapsed = (
                future.result()
            )  # Propagate failures rather than report false success.
            print(f"  [{name}] completed in {elapsed:.2f}s")
    parallel_time = time.perf_counter() - start_parallel
    conflicts_after = sum(1 for e in _write_log if e["op"] == "CONFLICT")
    final2 = get_order(order2["order_id"])
    print(f"\n  {'─' * 52}")
    print(f"  FINAL STATE — {final2['order_id']}")
    print(f"    Version: {final2['version']} | Status: {final2['status']}")
    print(f"    Driver: {final2['driver']['name'] if final2['driver'] else 'None'}")
    print(
        f"    Total: ${final2['total_price']['total']:.2f}"
        if final2["total_price"]
        else "    Total: None"
    )
    print(f"    Progress: {final2['progress']}")
    print(f"    Version conflicts: {conflicts_after - conflicts_before}")
    print(f"    Parallel time: {parallel_time:.2f}s")
    print(f"  {'─' * 52}")

    # SCENARIO 3: State Recovery (restaurant rejection) — provided for you
    order3 = ORDERS[2]
    print(f"\n{'━' * 70}")
    print("  SCENARIO 3: State Recovery (restaurant rejection + returning customer)")
    print(f"  Order: {order3['order_id']} — {order3['restaurant']}")
    print(
        f"  Restaurant will REJECT → triggers cleanup | Customer: {order3['customer_name']} (same as ORD-001)"
    )
    print(f"{'━' * 70}")

    cust_id = order3["customer_id"]
    mem = customer_memory.get(cust_id, {})
    print(f"\n  Customer memory for {cust_id}: {json.dumps(mem, indent=4)}")

    record3 = create_order(order3)
    print(f"  Created: {order3['order_id']} (version {record3['version']})")

    print("\n  [Partial] DriverAssignAgent (should use preferred driver)...")
    run_agent_with_retry(
        build_driver_assign_agent, f"Assign driver for order {order3['order_id']}"
    )
    state3 = get_order(order3["order_id"])
    driver3 = state3.get("driver", {})
    print(
        f"    Driver: {driver3.get('name', '?')} — {driver3.get('match_reason', '?')} (v{state3['version']})"
    )

    print("  [Partial] PriceCalculatorAgent...")
    run_agent_with_retry(
        build_price_calculator_agent, f"Calculate price for order {order3['order_id']}"
    )
    state3 = get_order(order3["order_id"])
    print(f"    Price: ${state3['total_price']['total']:.2f} (v{state3['version']})")

    print("\n  [REJECT] RestaurantConfirmAgent — restaurant rejects order!")
    run_agent_with_retry(
        lambda: build_restaurant_confirm_agent(True),
        f"Confirm order {order3['order_id']}",
    )
    state3 = get_order(order3["order_id"])
    print(f"    Status: {state3['status']} (v{state3['version']})")

    print("\n  [RECOVERY] Cleaning up partial state...")
    recovered = recover_order(order3["order_id"])

    print(f"\n  Final State: {recovered['order_id']} v{recovered['version']}")
    print(
        f"    Status: {recovered['status']} | Driver: {recovered['driver']} | Price: {recovered['total_price']}"
    )
    print(f"    Progress: {recovered['progress']}")

    print(f"\n{'═' * 70}")
    print("  SHARED STATE SUMMARY")
    print(f"{'═' * 70}")

    total_writes = sum(1 for e in _write_log if e["op"] in ("put_item", "update_item"))
    total_conflicts = sum(1 for e in _write_log if e["op"] == "CONFLICT")

    print(f"  Total writes: {total_writes} | Total conflicts: {total_conflicts}")
    if (total_writes + total_conflicts) > 0:
        print(
            f"  Conflict rate: {total_conflicts / (total_writes + total_conflicts) * 100:.0f}%"
        )
    print("  All resolved: YES (via optimistic locking + retry)")

    print("\n  Customer Memory (AgentCore Memory SESSION_SUMMARY simulation):")
    for cid, mem in customer_memory.items():
        print(
            f"    {cid}: preferred_driver={mem.get('preferred_driver', '?')}, "
            f"last_driver={mem.get('last_driver', '?')}, favorite_restaurant={mem.get('favorite_restaurant', '?')}"
        )

    print("\n  Key Insight: Exercise adds STATE RECOVERY to the demo's pattern:")
    print("  1. OPTIMISTIC LOCKING — version + ConditionExpression (same as demo)")
    print("  2. RETRY ON CONFLICT — re-read → new version → retry (same as demo)")
    print("  3. STATE RECOVERY — reject → cleanup partial updates → cancel (NEW)")
    print("  4. TTL — auto-expire completed orders (2 hours)")
    print("  5. AGENTCORE MEMORY — customer preferences persist across orders\n")


if __name__ == "__main__":
    main()
