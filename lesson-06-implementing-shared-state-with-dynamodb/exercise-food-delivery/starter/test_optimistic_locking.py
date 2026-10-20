"""Controlled DynamoDB optimistic-locking collision test.

Run beside food_delivery_state.py using the same environment and AWS credentials.
Creates a uniquely named temporary order, forces two writers to read version 0,
then verifies conflict detection, retry, and preservation of both updates.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from uuid import uuid4

import food_delivery_state as app


class CoordinatedTable:
    """Synchronize only the first two reads; all writes go to real DynamoDB."""

    def __init__(self, real_table):
        self.real_table = real_table
        self.barrier = Barrier(2, timeout=20)
        self.lock = Lock()
        self.read_count = 0

    def get_item(self, **kwargs):
        # Strongly consistent read ensures both writers observe the newly created item.
        result = self.real_table.get_item(**{**kwargs, "ConsistentRead": True})
        with self.lock:
            self.read_count += 1
            first_round = self.read_count <= 2
        if first_round:
            version = result.get("Item", {}).get("version")
            print(f"  [READ] version={version}; waiting for the other writer", flush=True)
            self.barrier.wait()
        return result

    def put_item(self, **kwargs):
        return self.real_table.put_item(**kwargs)


def main():
    order_id = f"LOCK-TEST-{uuid4().hex[:12].upper()}"
    real_table = app.order_table
    order = {**app.ORDERS[0], "order_id": order_id}
    app._write_log.clear()
    created = False

    try:
        # Avoid replacing an existing order even in the extremely unlikely event of ID reuse.
        record = {
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "customer_name": order["customer_name"],
            "restaurant": order["restaurant"],
            "items": order["items"],
            "address": order["address"],
            "payment_method": order["payment_method"],
            "driver": None,
            "total_price": None,
            "status": "pending",
            "progress": [],
            "version": 0,
            "ttl": int(app.time.time()) + 7200,
            "created_at": app.datetime.now(app.UTC).isoformat(),
        }
        real_table.put_item(
            Item=app.to_dynamo(record),
            ConditionExpression="attribute_not_exists(order_id)",
        )
        created = True
        print(f"Temporary order: {order_id} (version 0)")

        app.order_table = CoordinatedTable(real_table)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(app.update_order, order_id, {"restaurant_test": "accepted"}),
                pool.submit(app.update_order, order_id, {"driver_test": "DRV-301"}),
            ]
            for future in futures:
                future.result()

        app.order_table = real_table
        final = real_table.get_item(Key={"order_id": order_id}, ConsistentRead=True)["Item"]
        conflicts = [e for e in app._write_log if e["op"] == "CONFLICT"]
        print(f"\nFinal version: {final['version']}")
        print(f"Restaurant update: {final.get('restaurant_test')}")
        print(f"Driver update: {final.get('driver_test')}")
        print(f"Detected conflicts: {len(conflicts)}")
        assert final["version"] == 2, "Expected two successful version increments"
        assert final.get("restaurant_test") == "accepted", "Restaurant update was lost"
        assert final.get("driver_test") == "DRV-301", "Driver update was lost"
        assert len(conflicts) >= 1, "Expected at least one version conflict"
        print("PASS: collision detected, retry succeeded, both updates preserved")
    finally:
        app.order_table = real_table
        if created:
            real_table.delete_item(Key={"order_id": order_id})
            print(f"Temporary order {order_id} deleted")


if __name__ == "__main__":
    main()
