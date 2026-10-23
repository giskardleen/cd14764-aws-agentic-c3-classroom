# Multi-Agent Systems Engineering | AWS Agentic AI Nanodegree

Hands-on implementations and engineering exercises exploring multi-agent architectures, orchestration patterns, shared state, distributed coordination, retrieval-augmented generation, guardrails, and production deployment on AWS.

This repository documents my practical work throughout the multi-agent systems and protocol integration course of the **AWS Agentic AI Nanodegree**, developed by Udacity in collaboration with AWS.

## Overview

The course explores how to design and implement systems in which multiple AI agents collaborate, delegate responsibilities, coordinate execution, and interact with external services.

My work focuses on understanding the engineering decisions behind these systems, including:

- Agent specialization and responsibility boundaries
- Multi-model and multi-agent composition
- Parallel execution and orchestration
- Routing and delegation strategies
- Shared state and concurrency control
- Distributed transactions and the Saga pattern
- Multi-agent retrieval-augmented generation (RAG)
- Guardrails, evaluation, and operational controls
- Deployment, monitoring, and inter-agent communication

## Repository Structure

```text
.
├── lesson-01-implementing-multi-agent-architecture/
├── lesson-02-implementing-multi-model-multi-agent-systems/
├── lesson-03-implementing-agentic-parallelization-workflows/
├── lesson-04-implementing-agent-orchestration/
├── lesson-05-implementing-routing-in-multi-agent-systems/
├── lesson-06-implementing-shared-state-with-dynamodb/
├── lesson-07-saga-pattern-and-state-coordination/
├── lesson-08-implementing-multi-agent-rag/
├── lesson-09-implementing-guardrails-and-evaluation/
├── lesson-10-production-deployment-and-monitoring/
├── lesson-11-inter-agent-communication-via-gateway/
└── project/
```

Each lesson contains two complementary components:

- **`demo-*`** — Instructor-provided demonstrations retained as course references.
- **`exercise-*`** — My hands-on exercise workspace, including implementations, configuration, architecture diagrams, and supporting documentation.

The `project/starter/` directory retains the course-provided project scaffold.

## Engineering Topics and Exercises

| Lesson | Engineering topic | Exercise |
|---|---|---|
| 01 | Multi-agent architecture | Smart Home Device Management |
| 02 | Multi-model agent systems | Content Moderation Pipeline |
| 03 | Parallelization workflows | Contract Compliance |
| 04 | Agent orchestration | Package Delivery |
| 05 | Routing and delegation | Telecom Router |
| 06 | Shared state with DynamoDB | Food Delivery |
| 07 | Saga pattern and state coordination | E-commerce Checkout |
| 08 | Multi-agent RAG | Clinical Literature |
| 09 | Guardrails and evaluation | Trading Compliance |
| 10 | Production deployment and monitoring | VectraBank Architecture |
| 11 | Inter-agent communication via AgentCore Gateway | Analytics Gateway |

Each lesson's README provides additional context, architecture details, and instructions for its associated activities.

## Implementation Notes

The exercise directories contain the work performed during the course, organized for direct navigation rather than retaining separate starter and reference-solution directories.

The implementations explore patterns using Python, AWS services, and agentic development frameworks. Depending on the lesson, activities may require AWS credentials, provisioned infrastructure, environment configuration, or additional dependencies.

Consult the corresponding lesson and exercise READMEs before running an implementation.

## Attribution

This repository is a personal fork of [Udacity's AWS Agentic C3 Classroom repository](https://github.com/udacity/cd14764-aws-agentic-c3-classroom).

Original course materials, instructional demonstrations, exercise scaffolds, and reference architectures are attributed to **Udacity and the original course contributors**.

The exercise directories document my practical work within the course. They should not be interpreted as independently conceived projects or as entirely original course material.

The original repository structure and course reference materials remain available through the upstream repository and Git history.

## Related Work

For a separate agent engineering project developed around context-aware technical diagnosis, see:

[Elementary — Context-Aware Diagnostic Support Agent](https://github.com/giskardleen/elementary-context-aware-diagnostic-agent)

This project explores evidence-grounded reasoning, information authority boundaries, and the separation between diagnosis and authorized action.
