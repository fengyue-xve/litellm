# Live provider route tests

Each route crate owns `tests/live.rs`. Use named `rstest` provider cases for the behaviors that route supports. Shared setup lives in `litellm_inference_testing::live`, with production HTTP settings, authentication services, environment secrets and a deadline

These tests execute the Rust route against a real provider. They do not prove Python parity, proxy authorization, billing persistence or agent compatibility. Those belong in the corresponding boundary and gateway suites

All live cases use `#[ignore = "calls a real provider, requires credentials and a live model"]`. Normal Cargo runs report them as ignored. An explicitly selected case fails on missing credentials, unavailable models, provider errors or timeouts. Never return early and count missing coverage as a pass

Set `LITELLM_LIVE_<ROUTE>_<PROVIDER>_MODEL` to the latest model available to the test account. The model has no compiled default because vendor availability changes independently of LiteLLM. Credentials use the provider's existing configuration rather than a second test-only key convention. Export credentials in the shell before running tests. The harness does not load `.env` or modify process environment while tests run

## Messages

Completion checks text and usage. Streaming consumes SSE to completion and checks text, usage and the terminal event. Tool round trips request a tool call and send its returned ID back with a result. Each case checks provider execution facts and rejects cache results

Messages setup and assertions live in `inference-messages/tests/live/support.rs`. Provider modules under `tests/live/` select their payloads and tool policy. The shared helpers consume streams completely and reconstruct text, thinking signatures and tool arguments for the follow-up request

The initial provider cases are `github_copilot`, `bedrock` and `edenai`. A registered ignored case is available coverage, not evidence that it passed against that provider

Run one provider's cases explicitly

```bash
LITELLM_LIVE_MESSAGES_GITHUB_COPILOT_MODEL=github_copilot/<enabled-claude-model> \
  cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live github_copilot \
  -- --ignored --nocapture --test-threads=1
```

Replace `<enabled-claude-model>` with a current model enabled on the account. Copilot uses the existing device login under `~/.config/litellm/github_copilot`, or `GITHUB_COPILOT_TOKEN_DIR`. It acquires and refreshes the inference session through the production Rust authentication service. Merely appearing in Copilot's `/models` response does not mean a model's policy is enabled

For Bedrock, set `LITELLM_LIVE_MESSAGES_BEDROCK_MODEL` to a supported Claude Invoke model or inference profile. Configure AWS credentials through the existing credential chain, or `AWS_BEARER_TOKEN_BEDROCK`, and the required region. For EdenAI, set `LITELLM_LIVE_MESSAGES_EDENAI_MODEL` and `EDENAI_API_KEY`. Select `bedrock` or `edenai` instead of `github_copilot` in the command. EdenAI requires a production token for live coverage, because sandbox tokens return dummy responses

EdenAI cases live in `inference-messages/tests/live/edenai.rs`. They cover system messages, portable cache hints, provider-reported cost handoff, text streaming, and complete and streaming tool round trips. The cost assertion compares the public response with the execution facts instead of pinning a vendor price

Bedrock cases live in `inference-messages/tests/live/bedrock.rs`. They cover completion with plain and system messages, unknown field filtering, plain and explicit Invoke model paths, text streaming, and complete and streaming tool round trips. Streaming checks the returned SSE content type, terminal event, usage and tool argument reconstruction

Tool cases require a model that supports forced `tool_choice`. A provider rejection fails the selected case instead of skipping it

Export `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION_NAME` and the model before running the Bedrock cases. An optional endpoint override uses `AWS_BEDROCK_RUNTIME_ENDPOINT`, not `AWS_BEDROCK_ENDPOINT`

```bash
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live bedrock \
  -- --ignored --nocapture --test-threads=1
```

Select one behavior by its test name, for example `completion::case_1_github_copilot`. To list every available cell without contacting any provider

```bash
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-inference-messages --test live -- --list
```

For another route, reuse `LiveResources`, `model` and `within_deadline`. Keep payload construction and response assertions in that route's test file. Add only supported provider behaviors, and document any gap instead of substituting mocks or skipping failures. Run cases serially to keep spending and rate limits predictable
