"""
vectrabank_architecture.py - EXERCISE STARTER (Student-Led)
==============================================================
Module 10 Exercise: Plan a Production Deployment Architecture for VectraBank

Create a deployment architecture plan for VectraBank's financial services
multi-agent system. Define runtime configuration, monitoring strategy,
cost estimates, and operational runbooks.

Same planning pattern as the demo (deployment_walkthrough.py),
with additions:
  1. FINANCIAL DOMAIN — VectraBank-specific agents and KBs
  2. OPERATIONAL RUNBOOK (NEW) — deploy, rollback, kill switch, latency
  3. COMPLIANCE REQUIREMENTS — VPC network mode, stricter thresholds
  4. COST OPTIMIZATION — model selection recommendations

Instructions:
  - Follow the demo pattern (deployment_walkthrough.py)
  - Look for TODO 1-8 below
  - Define configs as Python dicts; the provided helper handles deployment
  - Focus on WHAT to configure and WHY
"""

import json
import os
from pathlib import Path
import agentcore_cli
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Supporting resources are provisioned separately from the CLI runtime stack.
_RESOURCES = agentcore_cli.load_resources("lesson-10-exercise")
_ROLE_ARN = _RESOURCES["AgentCoreRoleArn"]
_GUARDRAIL_ID = _RESOURCES["GuardrailId"]
_GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT")


# ═══════════════════════════════════════════════════════
#  VECTRABANK RUNTIME CONFIGURATION
# ═══════════════════════════════════════════════════════

VECTRABANK_RUNTIME_CONFIG = {
    "agentRuntimeName": "vectrabank-financial-services",
    "description": "Multi-agent financial services system with RAG, guardrails, and compliance",
    "networkConfiguration": {
        # Production should use VPC mode with private subnet and security group IDs.
        "networkMode": "PUBLIC",
    },
    "protocolConfiguration": {
        "serverProtocol": "HTTP",
    },

    "guardrailConfiguration": {
        "guardrailIdentifier": _GUARDRAIL_ID,
        "guardrailVersion": _GUARDRAIL_VERSION,
    },

    "environmentVariables": {
        "MARKET_DATA_KB_ID": "KB-MARKET-DATA-001",
        "COMPLIANCE_KB_ID": "KB-COMPLIANCE-002",
        "PRODUCTS_KB_ID": "KB-PRODUCTS-003",
        "STATE_TABLE_NAME": "vectrabank-agent-state",
        "AUDIT_TABLE_NAME": "vectrabank-audit-log",
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "LOG_LEVEL": "INFO",
        "ENVIRONMENT": "production",
        "ENABLE_XRAY": "true",
    },
}


# ═══════════════════════════════════════════════════════
#  AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════

VECTRABANK_AGENTS = [
    {
        "name": "QueryRouter",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.0,
        "role": "Classify incoming financial queries and route to specialist",
        "tools": ["classify_query", "check_priority"],
        "estimated_tokens_per_request": 400,
        "requests_per_day": 10000,
    },
    {
        "name": "MarketDataRetriever",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.0,
        "role": "Retrieve market data and financial reports from KB",
        "tools": ["retrieve_market_data"],
        "estimated_tokens_per_request": 600,
        "requests_per_day": 6000,
    },
    {
        "name": "ComplianceRetriever",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.0,
        "role": "Retrieve SEC/FINRA regulations and compliance guidelines from KB",
        "tools": ["retrieve_compliance_docs"],
        "estimated_tokens_per_request": 600,
        "requests_per_day": 4000,
    },
    {
        "name": "FinancialAdvisor",
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "temperature": 0.1,
        "role": "Synthesize retrieved data into grounded financial analysis with citations",
        "tools": [],
        "estimated_tokens_per_request": 2500,
        "requests_per_day": 10000,
    },
]


# ═══════════════════════════════════════════════════════
#  MONITORING STRATEGY
# ═══════════════════════════════════════════════════════

VECTRABANK_MONITORING = {
    "dashboard_name": "vectrabank-financial-services",
    "widgets": [
        {"title": "Total Queries", "type": "line", "metric": "AgentCore/Invocations",
         "period": 60, "stat": "Sum"},
        {"title": "Latency P50/P99", "type": "line",
         "metrics": [
             {"name": "AgentCore/Latency", "stat": "p50"},
             {"name": "AgentCore/Latency", "stat": "p99"},
         ], "period": 60},
        {"title": "Error Rate", "type": "number", "metric": "AgentCore/Errors",
         "period": 300, "stat": "Average", "threshold": 0.02},
        {"title": "Guardrail Blocks by Policy", "type": "stacked_bar",
         "metrics": [
             {"name": "Guardrail/PIIBlocks", "label": "PII"},
             {"name": "Guardrail/TopicBlocks", "label": "Topic"},
             {"name": "Guardrail/ContentBlocks", "label": "Content"},
         ], "period": 300},
        {"title": "RAG Retrieval Quality", "type": "line",
         "metric": "Custom/AvgRelevanceScore", "period": 300, "stat": "Average"},
        {"title": "Kill Switch Status", "type": "indicator",
         "metric": "Custom/KillSwitchActive", "period": 60},
    ],
    "alarms": [
        {
            "name": "HighErrorRate",
            "metric": "AgentCore/Errors",
            "threshold": 0.02,
            "period": 300,
            "action": "SNS → kill-switch-topic → Lambda disables runtime",
        },
        {
            "name": "HighLatencyP99",
            "metric": "AgentCore/Latency",
            "stat": "p99",
            "threshold": 8.0,
            "period": 300,
            "action": "SNS → ops-team-pager",
        },
        {
            "name": "GuardrailViolationSpike",
            "metric": "Guardrail/TotalBlocks",
            "threshold": 50,
            "period": 300,
            "action": "SNS → security-team + rate-limit-increase",
        },
    ],
    "xray_tracing": {
        "enabled": True,
        "sampling_rate": 0.10,
        "annotations": ["query_type", "agent_name", "model_id", "customer_tier"],
    },
}


# ═══════════════════════════════════════════════════════
#  COST ESTIMATION (provided)
# ═══════════════════════════════════════════════════════

MODEL_PRICING = {
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 0.003, "output": 0.015},
}


def estimate_monthly_costs(agents: list, days: int = 30) -> dict:
    """Estimate monthly costs for VectraBank."""
    costs = {}
    total = 0

    for agent in agents:
        model = agent["model"]
        pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.005})
        tokens = agent["estimated_tokens_per_request"]
        daily_requests = agent["requests_per_day"]

        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        daily_cost = ((input_tokens / 1000) * pricing["input"] +
                      (output_tokens / 1000) * pricing["output"]) * daily_requests
        monthly_cost = daily_cost * days

        costs[agent["name"]] = {
            "model": model, "daily_requests": daily_requests,
            "monthly_cost": round(monthly_cost, 2),
        }
        total += monthly_cost

    infrastructure_costs = {
        "DynamoDB (2 tables)": 50.0,
        "Knowledge Bases (3 KBs)": 30.0,
        "CloudWatch + X-Ray": 25.0,
        "VPC/Networking": 10.0,
    }
    for name, monthly_cost in infrastructure_costs.items():
        costs[name] = {"monthly_cost": monthly_cost}
        total += monthly_cost

    costs["TOTAL"] = {"monthly_cost": round(total, 2)}
    return costs


# ═══════════════════════════════════════════════════════
#  OPERATIONAL RUNBOOK
# ═══════════════════════════════════════════════════════

OPERATIONAL_RUNBOOK = {
    "deploy": {
        "name": "Production Deployment",
        "steps": [
            "Run the full test suite",
            "Validate the guardrail configuration",
            "Deploy the new runtime version",
            "Run smoke tests on 10 test transactions",
            "Monitor error rate and latency for 5 minutes",
        ],
        "rollback_trigger": "Error rate > 2%",
        "estimated_duration_minutes": 15,  # Estimate time to complete
    },

    "rollback": {
        "name": "Emergency Rollback",
        "steps": [
            "Identify the previous stable runtime version",
            "Revert the runtime to the previous version",
            "Verify agents are responding to transactions",
            "Conduct a post-incident review to determine what went wrong",
        ],
        "estimated_duration_minutes": 10,
    },

    "kill_switch": {
        "name": "Kill Switch Activation",
        "steps": [
            "Acknowledge the alert in PagerDuty",
            "Check AWS X-Ray for error patterns",
            "Check CloudWatch audit logs for anomalies",
            "Disable the runtime if an attack or fraud is detected",
            "Notify the compliance team",
        ],
        "threshold": "Guardrail blocks > 50/5min",
        "requires_approval": True,  # Financial system requires human approval
        "estimated_duration_minutes": 5,
    },

    "latency_investigation": {
        "name": "High Latency Investigation",
        "steps": [
            "Pull the X-Ray service map to identify the slow service",
            "Check per-agent latency in CloudWatch",
            "Identify whether the bottleneck is an agent or knowledge base retrieval",
            "Check for DynamoDB throttling or knowledge base latency",
            "Scale the bottleneck component by increasing provisioned capacity",
        ],
        "trigger_threshold": "P99 latency > 8 seconds",
        "estimated_duration_minutes": 20,
    },
}


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def deploy_to_agentcore() -> str:
    """Provided infrastructure; student architecture TODOs remain above."""
    required = ("networkConfiguration", "protocolConfiguration", "environmentVariables")
    if any(key not in VECTRABANK_RUNTIME_CONFIG for key in required) or len(VECTRABANK_AGENTS) != 4:
        raise ValueError("Complete the runtime configuration and four agent definitions before deployment.")
    resources = agentcore_cli.load_resources("lesson-10-exercise")
    config = {**VECTRABANK_RUNTIME_CONFIG,
              "roleArn": resources["AgentCoreRoleArn"],
              "guardrailConfiguration": {
                  "guardrailIdentifier": resources["GuardrailId"],
                  "guardrailVersion": os.environ.get("GUARDRAIL_VERSION", "DRAFT")}}
    return agentcore_cli.deploy(config)


def main():
    print("=" * 70)
    print("  VectraBank Deployment Architecture — Module 10 Exercise")
    print("  Runtime Config + Monitoring + Cost + Operational Runbook")
    print("=" * 70)

    # ── Runtime Configuration ──
    print(f"\n{'━' * 70}")
    print("  1. AgentCore Runtime Configuration")
    print(f"{'━' * 70}")

    cfg = VECTRABANK_RUNTIME_CONFIG
    print(f"  Runtime:   {cfg['agentRuntimeName']}")
    print(f"  Network:   {cfg['networkConfiguration']['networkMode']}")
    print(f"  Protocol:  {cfg['protocolConfiguration']['serverProtocol']}")
    guardrail = cfg["guardrailConfiguration"]
    print(f"  Guardrail: {guardrail['guardrailIdentifier']} "
          f"v{guardrail['guardrailVersion']}")
    print("  Environment Variables:")
    for key, value in cfg["environmentVariables"].items():
        print(f"    {key}: {value}")

    print(f"\n{'━' * 70}")
    print(f"  2. Agent Definitions ({len(VECTRABANK_AGENTS)} agents)")
    print(f"{'━' * 70}")
    for agent in VECTRABANK_AGENTS:
        print(f"\n  {agent['name']}:")
        print(f"    Model:       {agent['model']}")
        print(f"    Temperature: {agent['temperature']}")
        print(f"    Role:        {agent['role']}")
        print(f"    Daily reqs:  {agent['requests_per_day']:,}")

    print(f"\n{'━' * 70}")
    print("  3. Monitoring Strategy")
    print(f"{'━' * 70}")
    monitoring = VECTRABANK_MONITORING
    print(f"\n  Dashboard: {monitoring['dashboard_name']} "
          f"({len(monitoring['widgets'])} widgets)")
    for widget in monitoring["widgets"]:
        print(f"    [{widget['type']:12s}] {widget['title']}")
    print(f"\n  Alarms ({len(monitoring['alarms'])}):")
    for alarm in monitoring["alarms"]:
        print(f"    {alarm['name']}: threshold={alarm['threshold']}, "
              f"period={alarm['period']}s")
        print(f"      -> {alarm['action']}")
    tracing = monitoring["xray_tracing"]
    print(f"\n  X-Ray: sampling={tracing['sampling_rate'] * 100:.0f}%, "
          f"annotations={', '.join(tracing['annotations'])}")

    if VECTRABANK_AGENTS:
        # ── Cost Estimation ──
        print(f"\n{'━' * 70}")
        print("  4. Monthly Cost Estimation (10,000 requests/day)")
        print(f"{'━' * 70}")
        costs = estimate_monthly_costs(VECTRABANK_AGENTS)
        print(f"\n  {'Component':<25s} {'Model':<40s} {'Monthly':>10s}")
        print(f"  {'─' * 75}")
        for name, data in costs.items():
            if name == "TOTAL":
                print(f"  {'─' * 75}")
            model = data.get("model", "—")
            cost = data["monthly_cost"]
            print(f"  {name:<25s} {model:<40s} ${cost:>9.2f}")

    print(f"\n{'━' * 70}")
    print(f"  5. Operational Runbook ({len(OPERATIONAL_RUNBOOK)} procedures)")
    print(f"{'━' * 70}")
    for runbook in OPERATIONAL_RUNBOOK.values():
        print(f"\n  {runbook['name']}")
        for step in runbook["steps"]:
            print(f"    {step}")

    print(f"\n  Key Takeaways:")
    print(f"  1. VPC NETWORK MODE — financial services agents stay internal")
    print(f"  2. MULTI-MODEL COST OPTIMIZATION — Lite for routing/retrieval, Sonnet for synthesis")
    print(f"  3. STRICTER THRESHOLDS — 2% error rate for financial compliance")
    print(f"  4. OPERATIONAL RUNBOOK — deploy, rollback, kill switch, latency procedures (NEW)")
    print(f"  5. AUDIT TRAIL — X-Ray at 10% sampling + full guardrail audit log\n")

    deploy_to_agentcore()


if __name__ == "__main__":
    main()
