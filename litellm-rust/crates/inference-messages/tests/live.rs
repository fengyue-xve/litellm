use std::{sync::OnceLock, time::Duration};

use futures_util::FutureExt;
use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
};
use litellm_inference_messages::{Error, MessagesCall, MessagesCallResponse, MessagesRoute};
use litellm_inference_testing::live::{LiveResources, model};
use litellm_llms_types::formats::messages::MessagesResponse;
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
            json!({"route": "messages", "provider": self.provider, "model": context.model})
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

#[path = "live/bedrock.rs"]
mod bedrock;
