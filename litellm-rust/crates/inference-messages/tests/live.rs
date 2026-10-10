use std::{sync::OnceLock, time::Duration};

use futures_util::{FutureExt, TryStreamExt};
use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
};
use litellm_inference_messages::{Error, MessagesCall, MessagesCallResponse, MessagesRoute};
use litellm_inference_testing::live::{LiveResources, model, within_deadline};
use litellm_llms_types::formats::messages::MessagesResponse;
use rstest::rstest;
use serde_json::{Value, json};

struct LiveCall {
    provider: &'static str,
    facts: OnceLock<ExecutionFacts>,
}

impl LiveCall {
    fn new(provider: &'static str) -> Self {
        Self {
            provider,
            facts: OnceLock::new(),
        }
    }

    fn assert_provider_result(&self) {
        let facts = self
            .facts
            .get()
            .expect("the route must deliver execution facts");
        assert_eq!(facts.provider.provider, self.provider);
        assert_eq!(facts.source, ResultSource::Provider);
        println!(
            "{}",
            json!({"route": "messages", "provider": self.provider, "source": "provider"})
        );
    }
}

impl Interceptors<Error> for LiveCall {
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        assert!(
            self.facts.set(facts).is_ok(),
            "result must be accepted exactly once"
        );
        Ok(())
    }

    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        assert_eq!(context.custom_llm_provider, self.provider);
        println!(
            "{}",
            json!({"route": "messages", "provider": self.provider, "model": wire.body["model"]})
        );
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Ok(())
    }
}

#[rstest::fixture]
fn route() -> MessagesRoute {
    let resources = LiveResources::default();
    MessagesRoute::new(resources.http, resources.auth, resources.secrets)
}

async fn request(
    route: &MessagesRoute,
    call: MessagesCall,
    host: &LiveCall,
) -> MessagesCallResponse {
    route
        .execute(call, host, None)
        .boxed()
        .await
        .expect("live request failed")
}

fn call(provider: &str, payload: Value) -> MessagesCall {
    let Value::Object(fields) = payload else {
        panic!("live payload must be an object")
    };
    let body = serde_json::from_value(Value::Object(
        fields
            .into_iter()
            .chain([("model".into(), json!(model("messages", provider)))])
            .collect(),
    ))
    .expect("live payload must be a valid Messages request");
    MessagesCall {
        body,
        api_key: None,
        api_base: None,
        custom_llm_provider: Some(provider.into()),
        litellm_params: Default::default(),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(60)),
        shaping: Default::default(),
    }
}

fn complete(response: MessagesCallResponse) -> Box<MessagesResponse> {
    match response {
        MessagesCallResponse::Complete(message) => message.body,
        MessagesCallResponse::Stream { .. } => panic!("a complete request returned a stream"),
    }
}

fn assert_text(message: &MessagesResponse) {
    assert!(!message.id.is_empty());
    assert_eq!(message.role, "assistant");
    assert!(message.stop_reason.is_some());
    assert!(
        message
            .content
            .iter()
            .any(|block| block["text"].as_str().is_some_and(|text| !text.is_empty()))
    );
    assert!(
        message
            .usage
            .as_ref()
            .and_then(|usage| usage["output_tokens"].as_u64())
            .is_some_and(|count| count > 0)
    );
}

#[rstest]
#[case::bedrock("bedrock")]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion(route: MessagesRoute, #[case] provider: &'static str) {
    within_deadline(async {
        let host = LiveCall::new(provider);
        let response = request(&route, call(provider, json!({
            "max_tokens": 128, "messages": [{"role": "user", "content": "Say hello in one sentence."}]
        })), &host).await;
        let message = complete(response);
        assert_text(&message);
        host.assert_provider_result();
        println!("{}", serde_json::to_string(&message).unwrap());
    }).await;
}

#[rstest]
#[case::bedrock("bedrock")]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn streaming(route: MessagesRoute, #[case] provider: &'static str) {
    within_deadline(async {
        let host = LiveCall::new(provider);
        let response = request(
            &route,
            call(
                provider,
                json!({
                    "max_tokens": 128, "stream": true,
                    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
                }),
            ),
            &host,
        )
        .await;
        let MessagesCallResponse::Stream { head, chunks } = response else {
            panic!("stream request returned a complete response")
        };
        assert!(
            head.headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("content-type")
                    && value.contains("text/event-stream"))
        );
        let frames = litellm_framer::frames(chunks, litellm_framer::sse::SseCodec::default())
            .try_collect::<Vec<_>>()
            .await
            .expect("live SSE framing failed");
        let events: Vec<Value> = frames
            .iter()
            .map(|frame| serde_json::from_str(&frame.data).expect("SSE data must be JSON"))
            .collect();
        assert_eq!(
            events.first().map(|event| event["type"].as_str()),
            Some(Some("message_start"))
        );
        assert_eq!(
            events.last().map(|event| event["type"].as_str()),
            Some(Some("message_stop"))
        );
        assert!(!events.iter().any(|event| event["type"] == "error"));
        assert!(events.iter().any(|event| {
            event["delta"]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        }));
        assert!(events.iter().any(|event| {
            event["usage"]["output_tokens"]
                .as_u64()
                .is_some_and(|count| count > 0)
        }));
        host.assert_provider_result();
        println!("{}", json!({"events": events}));
    })
    .await;
}

#[rstest]
#[case::bedrock("bedrock")]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn tool_round_trip(route: MessagesRoute, #[case] provider: &'static str) {
    within_deadline(async {
        let first_host = LiveCall::new(provider);
        let prompt = json!({"role": "user", "content": "Use the echo tool with value live-check, then tell me the result."});
        let tools = json!([{"name": "echo", "description": "Return a value", "input_schema": {
            "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]
        }}]);
        let first = complete(request(&route, call(provider, json!({
            "max_tokens": 128, "messages": [prompt], "tools": tools,
            "tool_choice": {"type": "tool", "name": "echo"}
        })), &first_host).await);
        assert_eq!(first.stop_reason.as_deref(), Some("tool_use"));
        let tool = first.content.iter().find(|block| block["type"] == "tool_use").expect("forced tool must be returned");
        assert_eq!(tool["name"], "echo");
        assert_eq!(tool["input"]["value"], "live-check");
        let tool_id = tool["id"].as_str().filter(|id| !id.is_empty()).expect("tool must have an id");
        first_host.assert_provider_result();
        let second_host = LiveCall::new(provider);
        let second = complete(request(&route, call(provider, json!({
            "max_tokens": 128, "tools": tools, "messages": [prompt,
                {"role": "assistant", "content": first.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "live-check"}]}]
        })), &second_host).await);
        assert_text(&second);
        assert!(second.content.iter().any(|block| block["text"].as_str().is_some_and(|text| text.contains("live-check"))));
        second_host.assert_provider_result();
        println!("{}", serde_json::to_string(&second).unwrap());
    }).await;
}
