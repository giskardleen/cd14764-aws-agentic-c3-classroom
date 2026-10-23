"""
analytics_gateway.py - EXERCISE STARTER (Student-Led)
==============================================================
Module 11 Exercise: Register and Invoke Tools through AgentCore Gateway

Same Gateway pattern as the demo (supply_chain_gateway.py),
with additions:
  1. MIXED FUNCTIONALITY: 2 analytical tools + 1 news tool
  2. DIFFERENT DOMAIN: Analytics utilities
  3. SEMANTIC ROUTING: Agent selects tool by query meaning

Instructions:
  - Follow the demo pattern (supply_chain_gateway.py)
  - Look for TODO 1-9 below
  - Register 3 targets on the Gateway
  - Build an agent that discovers and uses Gateway tools

Tech Stack:
  - Python 3.11+
  - Strands Agents SDK (Agent class, @tool decorator)
  - Amazon Bedrock (Nova Lite for the agent)
  - AWS Lambda (tool backends via boto3)
  - LambdaGateway (registry pattern for Lambda functions)

Production equivalent: Amazon Bedrock AgentCore Gateway (MCP protocol)
"""

import json
import logging
import os
import re
import time

import boto3
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel

load_dotenv()
logging.basicConfig(level=logging.WARNING)


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


# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")

# Lambda client for invoking tool backends
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

# Lambda function names (from CloudFormation)
WEATHER_FUNCTION = os.environ.get("WEATHER_FUNCTION", "lesson-11-exercise-weather")
CURRENCY_FUNCTION = os.environ.get("CURRENCY_FUNCTION", "lesson-11-exercise-currency")
NEWS_FUNCTION = os.environ.get("NEWS_FUNCTION", "lesson-11-exercise-news")
STOCK_PRICE_FUNCTION = os.environ.get(
    "STOCK_PRICE_FUNCTION", "lesson-11-exercise-stock-price"
)


# LAMBDA GATEWAY
class LambdaGateway:
    """Gateway that routes tool calls to AWS Lambda functions.

    This gateway implements the registration → discovery → invocation pattern
    using AWS Lambda as the tool backend. Each registered tool maps to a
    Lambda function that is invoked via boto3.

    Production: Amazon Bedrock AgentCore Gateway (MCP protocol)
    """

    def __init__(self, name: str, description: str):
        self.gateway_id = f"GW-{name.upper().replace(' ', '-')[:20]}"
        self.name = name
        self.description = description
        self.targets = {}  # name → {description, function_name, target_type}
        self.invocation_log = []

    def register_target(
        self,
        name: str,
        description: str,
        function_name: str,
        target_type: str = "lambda",
    ):
        """Register a Lambda function as a gateway target."""
        self.targets[name] = {
            "description": description,
            "function_name": function_name,
            "target_type": target_type,
        }

    def discover_tools(self, query: str = None) -> list[dict]:
        """List all registered tools (or filter by query)."""
        tools = []
        for name, config in self.targets.items():
            if (
                query is None
                or query.lower() in name.lower()
                or query.lower() in config["description"].lower()
            ):
                tools.append(
                    {
                        "name": name,
                        "description": config["description"],
                        "type": config["target_type"],
                    }
                )
        return tools

    def invoke_tool(self, tool_name: str, params: dict) -> dict:
        """Invoke a registered tool via Lambda."""
        if tool_name not in self.targets:
            return {
                "status": "error",
                "message": f"Tool '{tool_name}' not found in gateway",
            }

        target = self.targets[tool_name]
        function_name = target["function_name"]

        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(params),
        )

        result = json.loads(response["Payload"].read().decode("utf-8"))

        self.invocation_log.append(
            {
                "tool": tool_name,
                "function": function_name,
                "params": params,
                "result_status": result.get("status", "unknown"),
                "timestamp": time.time(),
            }
        )

        return result


# Note: Handler functions now live in AWS Lambda (see infrastructure/stack.yaml)
# The data that was previously in-memory is now stored in Lambda functions,
# which are invoked through the gateway's invoke_tool() method.


# ANALYTICS AGENT
def build_analytics_agent(gateway: LambdaGateway) -> Agent:
    """Build an analytics agent connected to the Gateway."""

    # TODO 1: Create a BedrockModel
    model = BedrockModel(
        model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.1
    )

    # TODO 2: Build system prompt listing available Gateway tools
    available_tools = gateway.discover_tools()
    tool_list = "\n".join(
        f"  - {t['name']}: {t['description']}" for t in available_tools
    )
    system_prompt = f"""You are a data analytics agent.

    You have access to the following tools via the Gateway:
    {tool_list}

    Responsibilities:
    - Select the appropriate tool based on the user's request.
    - Use tool results as the source of truth for reported data.
    - Provide concise, factual answers.

    Boundaries:
    - Do not fabricate data, tool results, or sources.
    - Do not present assumptions or inferences as verified facts.
    - Do not claim access to real-time information unless supported by the tool results.
    - If the available tools cannot answer a request, clearly state the limitation.
    - Do not perform actions beyond the capabilities provided by your tools.
    """

    # TODO 3: Create @tool function for weather lookup
    @tool
    def get_weather(city: str) -> str:
        """Look up current weather conditions for a city.
        Args:
            city: City name (e.g., "Tokyo", "London", "New York")
        Returns:
            JSON with weather data(temperature, conditions, humidity, wind)
        """
        result = gateway.invoke_tool("weather_lambda", {"city": city})
        return json.dumps(result, indent=2)

    # TODO 4: Create @tool function for currency conversion
    @tool
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert an amount between currencies, using real-time exchange rates.
        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., "USD")
            to_currency: Target currency code (e.g., "EUR")
        Returns:
            JSON with conversion result and exchange rate
        """
        result = gateway.invoke_tool(
            "currency_lambda",
            {"amount": amount, "from": from_currency, "to": to_currency},
        )
        return json.dumps(result, indent=2)

    # TODO 5: Create the get_news @tool function
    @tool
    def get_news(topic: str) -> str:
        """Get the latest news headlines for a topic.
        Args:
            topic: Topic to search for (e.g., "AI", "technology", "finance")
        Returns:
            JSON with news headlines for the requested topic and sources
        """
        result = gateway.invoke_tool("news_api", {"topic": topic})
        return json.dumps(result, indent=2)

    @tool
    def stock_price(symbol: str) -> str:
        """Get stock price information for a ticker symbol.

        Args:
            symbol: Stock ticker symbol, such as AMZN or AAPL.

        Returns:
            JSON with stock price, daily change, and company name.
        """
        result = gateway.invoke_tool("stock_price", {"symbol": symbol})
        return json.dumps(result, indent=2)

    # TODO 6: Return Agent with model, system_prompt, and all 3 tools
    tools = [get_weather, convert_currency, get_news]

    if "stock_price" in gateway.targets:
        tools.append(stock_price)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
    )


TEST_QUERIES = [
    {
        "query": "What is the weather in Tokyo right now?",
        "expected_tool": "weather_lambda",
        "description": "Weather lookup — routes to Weather Lambda",
    },
    {
        "query": "Convert 500 USD to EUR",
        "expected_tool": "currency_lambda",
        "description": "Currency conversion — routes to Currency Lambda",
    },
    {
        "query": "What are the latest AI news headlines?",
        "expected_tool": "news_api",
        "description": "News headlines — routes to News REST API",
    },
]


def main():
    print("=" * 70)
    print("  Analytics Gateway — Module 11 Exercise")
    print("  Agent discovers tools via Gateway (Lambda-backed)")
    print("=" * 70)

    # ── Create Gateway ──
    # TODO 7: Register 3 targets on the Gateway
    gateway = LambdaGateway(
        name="analytics-gateway",
        description="Lambda-backed tool gateway for analytics utilities",
    )
    print(f"\n  Gateway: {gateway.gateway_id}")
    gateway.register_target(
        name="weather_lambda",
        description="Look up weather conditions for a given city",
        function_name=WEATHER_FUNCTION,
    )

    gateway.register_target(
        name="currency_lambda",
        description="Convert amounts between currencies",
        function_name=CURRENCY_FUNCTION,
    )

    gateway.register_target(
        name="news_api",
        description="Get news headlines by topic",
        function_name=NEWS_FUNCTION,
    )

    print(f"  Registered {len(gateway.targets)} targets:")
    for t in gateway.discover_tools():
        print(f"    [{t['type']:8s}] {t['name']}")

    # TODO 8: Run test queries through the agent
    for i, test in enumerate(TEST_QUERIES):
        print(f"\n{'━' * 70}")
        print(f'  QUERY {i + 1}: "{test["query"]}"')
        print(f"  Expected tool: {test['expected_tool']}")
        print(f"  {test['description']}")
        print(f"{'━' * 70}")

        elapsed = run_agent_with_retry(
            lambda: build_analytics_agent(gateway), test["query"]
        )
        print(f"    Time: {elapsed:.1f}s")
    print(f"\n{'═' * 70}")
    print("INVOCATION LOG")
    print(f"{'═' * 70}")
    for entry in gateway.invocation_log:
        print(
            f"  Tool: {entry['tool']:20s} Lambda: {entry['function']:30s} Status: {entry['result_status']}"
        )

    print("\n  Key: 1) MIXED TARGETS — 2 Lambda + 1 REST API 2) SEMANTIC ROUTING")
    print("       3) ZERO CODE CHANGES — new API = config only\n")

    # ═══════════════════════════════════════════════════════
    #  EXTENSION: DYNAMIC TOOL REGISTRATION
    #  The gateway pattern's key advantage: add tools WITHOUT changing agent code.
    # ═══════════════════════════════════════════════════════

    print(f"{'═' * 70}")
    print("  EXTENSION: Adding Dynamic Tool Registration (no code changes)")
    print(f"{'═' * 70}")
    gateway.register_target(
        name="stock_price",
        description="Get the current stock price for a ticker symbol, including price and daily change",
        function_name=STOCK_PRICE_FUNCTION,
    )
    print(f"\n  {len(gateway.targets)} tools available:")
    for t in gateway.discover_tools():
        print(f"    [{t['type']:8s}] {t['name']}")

    test_query = "What is the current stock price of AMZN?"
    print(f"\n{'━' * 70}")
    print(f'  DYNAMIC QUERY: "{test_query}"')
    print(f"{'━' * 70}")
    elapsed = run_agent_with_retry(lambda: build_analytics_agent(gateway), test_query)
    print(f"    Time: {elapsed:.1f}s")

    # Final LOG ---- (provided)
    print(f"\n{'═' * 70}")
    print("FINAL INVOCATION LOG")
    print(f"{'═' * 70}")
    for entry in gateway.invocation_log:
        print(
            f"  Tool: {entry['tool']:20s} Lambda: {entry['function']:30s} Status: {entry['result_status']}"
        )

    print(
        "\n  Gateway Pattern Advantage: register a 4th tool (stock_price) with NO changes to:"
    )
    print("    - Agent invocation logic  (no re-deployment)")
    print(
        "    - System prompt           (rebuilt automatically from gateway.discover_tools())\n"
    )

    # ── AgentCore Gateway API (Production) ────────────────────────────────────
    print(f"{'═' * 70}")
    print("AgentCore Gateway API (Production)")
    print(f"{'═' * 70}")
    print("  Replace LambdaGateway with Amazon Bedrock AgentCore Gateway.")
    print(
        "  These exact API calls register the same tools on a managed MCP endpoint:\n"
    )

    agentcore_role = os.environ.get("AGENTCORE_ROLE_ARN", "")
    if not agentcore_role or agentcore_role.startswith("PASTE"):
        print("  AGENTCORE_ROLE_ARN not set in .env")
        print("  Run:  python infrastructure/deploy_stack.py")
        print("  Then paste the printed role ARN into .env as AGENTCORE_ROLE_ARN\n")
    else:
        try:
            print(
                "  AGENTCORE_ROLE_ARN detected. Production AgentCore registration can be enabled here.\n"
            )
        except Exception as e:
            print(f"  [{e.__class__.__name__}] {e}")
            print(
                "  (Check AGENTCORE_ROLE_ARN and that the Lambda stack is deployed)\n"
            )


if __name__ == "__main__":
    main()
