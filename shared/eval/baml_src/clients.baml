// LLM client configurations for the BIM QA judge.

// ============================================================================
// PRIMARY JUDGE: MiniMax M2.7 via MiniMax API (OpenAI-compatible endpoint)
// Used by EvaluateResponse (default judge)
// ============================================================================
client<llm> MinimaxM2_Fireworks {
  provider openai
  options {
    base_url "https://api.minimax.io/v1"
    model "MiniMax-M2.7"
    api_key env.MINIMAX_API_KEY
    temperature 0
    max_tokens 8192
    http {
      connect_timeout_ms 10000
      request_timeout_ms 120000
    }
  }
}

// ============================================================================
// SECOND JUDGE: Gemini 3 Pro (second opinion, via Google GenAI)
// Used by EvaluateResponseGemini
// ============================================================================
client<llm> Gemini_3_Pro {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-3-pro-preview"
    api_key env.GEMINI_API_KEY
  }
}

// ============================================================================
// THIRD JUDGE: OpenAI GPT (sanity-check judge)
// Used by EvaluateResponseGPT
// ============================================================================
client<llm> GPT_Model {
  provider openai
  options {
    model "gpt-5"
    api_key env.OPENAI_API_KEY
  }
}

// ============================================================================
// RETRY POLICIES
// ============================================================================
retry_policy Constant {
  max_retries 0
}

retry_policy Exponential {
  max_retries 0
}
