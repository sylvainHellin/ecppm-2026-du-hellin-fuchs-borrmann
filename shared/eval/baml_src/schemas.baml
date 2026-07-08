// Schemas for BIM QA LLM Judge
// Only includes types required by the answer verifier.

// Criterion evaluation result (yes/no/na)
enum CriterionResult {
  Yes
  No
  Na
}

// Answer evaluation result
class AnswerEvaluationResult {
  abstention bool @description("true if system explicitly declined to answer, false if answer provided")

  faithfulness CriterionResult @description(#"
    Are all claims grounded in valid sources for this question category?
    - Category 1: BIM data only
    - Category 2: Simple computations on BIM data
    - Category 3: Complex geometric computations from BIM data
    - Category 4: BIM data + explicitly stated assumptions
    Use 'na' only if abstention is true.
  "#)

  completeness CriterionResult @description(#"
    Are all relevant facts included to fully answer the question?
    Use 'na' if abstention is true OR question is open-ended with no objective completeness standard.
  "#)

  transparency CriterionResult @description(#"
    Are sources/methods explicitly disclosed for each claim?
    Use 'na' only if abstention is true.
  "#)

  relevance CriterionResult @description(#"
    Does the answer directly address the question asked?
    Use 'na' only if abstention is true.
  "#)

  justification string @description(#"
    Brief explanation (2-3 sentences) of the evaluation, especially for edge cases or borderline decisions.
    Focus on explaining any 'no' verdicts or unusual circumstances.
  "#)
}
