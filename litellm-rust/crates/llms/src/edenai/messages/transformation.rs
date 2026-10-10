use litellm_llms_types::{
    billing::BilledAmount,
    formats::messages::{MessagesRequest, MessagesResponse},
    providers::edenai::EdenAIResponseExtension,
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    anthropic::messages::transformation::{DEFAULT_HEADERS, update_headers_with_anthropic_beta},
    base_llm::messages::{
        context::MessagesTransformContext,
        transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
    },
    openai_like::messages::transformation::{
        compatible_host_environment, compatible_host_url, portable_cache_control,
        without_billing_blocks,
    },
};

const API_KEY_ENV: &str = "EDENAI_API_KEY";
const API_BASE_ENV: &str = "EDENAI_API_BASE";
const DEFAULT_BASE: &str = "https://api.edenai.run/v3";

/// Eden AI resells Claude behind its own bearer key, accepts only portable cache hints, and
/// reports the amount it billed as a top-level `cost` on the response.
pub struct EdenAIAnthropicMessagesConfig;
pub const EDENAI_MESSAGES_CONFIG: EdenAIAnthropicMessagesConfig = EdenAIAnthropicMessagesConfig;

impl BaseMessagesConfig for EdenAIAnthropicMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(compatible_host_url(
            api_base,
            API_BASE_ENV,
            DEFAULT_BASE,
            env,
        ))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(without_billing_blocks(request))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[API_KEY_ENV, API_BASE_ENV]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        compatible_host_environment(headers, api_key, "Eden AI", API_KEY_ENV, env)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }

    fn wire_body(&self, body: Value, headers: Headers) -> (Value, Headers) {
        (portable_cache_control(body), headers)
    }

    fn reported_cost(&self, response: &MessagesResponse) -> Option<serde_json::Number> {
        let extension: EdenAIResponseExtension =
            serde_json::from_value(Value::Object(response.extra.clone())).ok()?;
        extension.cost.map(BilledAmount::into_number)
    }
}
