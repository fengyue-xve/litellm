use futures_util::TryStreamExt;
use litellm_inference_testing::live::within_deadline;
use litellm_llms_types::formats::messages::MessagesRequest;
use rstest::rstest;

use super::*;

const PROVIDER: &str = "bedrock";

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}))]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[case::system_role(json!({"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello."}
]}))]
#[case::unknown_field(json!({
    "unknown_extension": true,
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion(route: MessagesRoute, #[case] payload: Value) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let message = complete(request(&route, bedrock_call(payload), &host).await);
        assert_text(&message);
        host.assert_provider_result();
        println!("{}", serde_json::to_string(&message).unwrap());
    })
    .await;
}

fn bedrock_call(payload: Value) -> MessagesCall {
    let Value::Object(fields) = payload else {
        panic!("live payload must be an object")
    };
    call(
        PROVIDER,
        Value::Object(
            fields
                .into_iter()
                .chain([("max_tokens".into(), json!(128))])
                .collect(),
        ),
    )
}

async fn stream_events(response: MessagesCallResponse) -> Vec<Value> {
    let MessagesCallResponse::Stream { head, chunks } = response else {
        panic!("stream request returned a complete response")
    };
    let content_types: Vec<_> = head
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.as_str())
        .collect();
    assert_eq!(content_types, ["text/event-stream"]);
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
        event["usage"]["output_tokens"]
            .as_u64()
            .is_some_and(|count| count > 0)
    }));
    println!("{}", json!({"events": events}));
    events
}

#[rstest]
#[case::model(false)]
#[case::invoke_prefix(true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn streaming(route: MessagesRoute, #[case] invoke_prefix: bool) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let original = bedrock_call(json!({
            "stream": true,
            "messages": [{"role": "user", "content": "Say hello."}]
        }));
        let model_id = original
            .body
            .model
            .strip_prefix("bedrock/")
            .unwrap_or(&original.body.model);
        let model = match invoke_prefix {
            false => original.body.model.clone(),
            true => format!(
                "bedrock/invoke/{}",
                model_id.strip_prefix("invoke/").unwrap_or(model_id)
            ),
        };
        let request_call = MessagesCall {
            body: MessagesRequest {
                model,
                ..original.body
            },
            ..original
        };
        let events = stream_events(request(&route, request_call, &host).await).await;
        assert!(events.iter().any(|event| {
            event["delta"]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        }));
        host.assert_provider_result();
    })
    .await;
}

fn prompt() -> Value {
    json!({"role": "user", "content": "Use the echo tool with value live-check, then tell me the result."})
}

fn tools() -> Value {
    json!([{"name": "echo", "description": "Return a value", "input_schema": {
        "type": "object", "properties": {"value": {"type": "string", "enum": ["live-check"]}}, "required": ["value"]
    }}])
}

#[rstest]
#[case::complete(false)]
#[case::streaming(true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn tool_round_trip(route: MessagesRoute, #[case] stream: bool) {
    within_deadline(async {
        let first_host = LiveCall::new(PROVIDER);
        let first = request(&route, bedrock_call(json!({
            "messages": [prompt()], "tools": tools(), "stream": stream,
            "tool_choice": {"type": "tool", "name": "echo"}
        })), &first_host).await;
        let content = match stream {
            false => {
                let message = complete(first);
                assert_eq!(message.stop_reason.as_deref(), Some("tool_use"));
                message.content
            }
            true => {
                let events = stream_events(first).await;
                assert!(events.iter().any(|event| event["delta"]["stop_reason"] == "tool_use"));
                let tool = events.iter().find(|event| event["content_block"]["type"] == "tool_use")
                    .expect("forced tool must be returned");
                let index = tool["index"].as_u64().expect("tool block must have an index");
                let input: String = events.iter().filter(|event| event["index"].as_u64() == Some(index))
                    .filter_map(|event| event["delta"]["partial_json"].as_str()).collect();
                let parsed: Value = serde_json::from_str(&input).expect("streamed tool arguments must be JSON");
                vec![json!({"type": "tool_use", "id": tool["content_block"]["id"],
                    "name": tool["content_block"]["name"], "input": parsed})]
            }
        };
        let tool = content.iter().find(|block| block["type"] == "tool_use").expect("forced tool must be returned");
        assert_eq!(tool["name"], "echo");
        assert_eq!(tool["input"]["value"], "live-check");
        let tool_id = tool["id"].as_str().filter(|id| !id.is_empty()).expect("tool must have an id");
        first_host.assert_provider_result();
        let second_host = LiveCall::new(PROVIDER);
        let second = complete(request(&route, bedrock_call(json!({
            "tools": tools(), "messages": [prompt(),
                {"role": "assistant", "content": content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "live-check"}]}]
        })), &second_host).await);
        assert_text(&second);
        second_host.assert_provider_result();
        println!("{}", serde_json::to_string(&second).unwrap());
    })
    .await;
}
