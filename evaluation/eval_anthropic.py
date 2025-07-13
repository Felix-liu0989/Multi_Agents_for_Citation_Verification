import json

import anthropic


class RelatedWorkEvaluation:
    def __init__(self, client=None):
        """
        Initialize the evaluation framework

        Args:
            client (anthropic.Anthropic, optional): Anthropic client for API calls
        """
        self.client = client or anthropic.Anthropic()

    def evaluate_relevance(self, identified_works, input_abstract):
        """
        Evaluate relevance of identified related works

        Args:
            identified_works (list): List of related works to evaluate
            input_abstract (str): Original abstract for context

        Returns:
            dict: Evaluation results with relevance scores
        """
        prompt = f"""
        You are an expert academic researcher tasked with evaluating the relevance of related works to an original abstract.

        Input Abstract: {input_abstract}

        Related Works to Evaluate:{json.dumps(identified_works, indent=2)}

        Please evaluate the relevance of each identified related work to the input abstract. Provide a numeric relevance score from 0-10, where:
        - 0-3: No meaningful connection
        - 4-6: Weak or tangential connection
        - 7-8: Moderate relevance with some key insights
        - 9-10: Highly relevant, directly addresses core research themes

        Respond in the following JSON format:
        {{
            "relevance_scores": {{
                "work_1": numeric_score,
                "work_2": numeric_score,
                ...
            }},
            "overall_relevance_summary": "Brief explanation of scoring"
        }}
        """
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1000, messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content[0].text)
        except Exception as e:
            return {"error": str(e)}

    def compare_related_works(self, input_section, synthetic_section, input_abstract):
        """
        Compare input and synthetic related works sections

        Args:
            input_section (str): Original related works section
            synthetic_section (str): Synthetic generated related works section
            input_abstract (str): Original abstract for context

        Returns:
            dict: Comparison results
        """
        prompt = f"""
        You are an expert researcher comparing two related works sections for a research paper.

        Input Abstract: {input_abstract}

        Original Related Works Section:{input_section}

        Synthetic Related Works Section:{synthetic_section}

        Comprehensively evaluate and compare these sections across multiple dimensions:

        1. Technical Accuracy (0-10): 
        2. Writing Quality (0-10):
        3. Comprehensiveness (0-10):
        4. Scholarly Tone (0-10):
        5. Overall Preference (Which section would you recommend?)

        Provide a one-sentence rationale for your overall section preference.

        Respond in the following JSON format:
        {{
            "technical_accuracy": numeric_score,
            "writing_quality": numeric_score, 
            "comprehensiveness": numeric_score,
            "scholarly_tone": numeric_score,
            "preferred_section": "input" or "synthetic",
            "rationale": "One-sentence explanation"
        }}
        """
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1000, messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content[0].text)
        except Exception as e:
            return {"error": str(e)}


def main():
    # Example usage
    evaluator = RelatedWorkEvaluation()

    # Sample data (replace with actual data)
    input_abstract = "A novel approach to machine learning..."
    identified_works = ["Paper on deep learning techniques", "Research about neural network architectures"]
    input_section = "Traditional related works discussion..."
    synthetic_section = "Synthetic generated related works section..."

    # Relevance Evaluation
    relevance_results = evaluator.evaluate_relevance(identified_works, input_abstract)
    print("Relevance Evaluation:", json.dumps(relevance_results, indent=2))

    # Related Works Comparison
    comparison_results = evaluator.compare_related_works(input_section, synthetic_section, input_abstract)
    print("Related Works Comparison:", json.dumps(comparison_results, indent=2))


if __name__ == "__main__":
    main()
